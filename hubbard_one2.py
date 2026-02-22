import matplotlib

matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import multiprocessing as mp
import time
import os
from scipy.integrate import simpson
from scipy.optimize import brentq
from tqdm import tqdm

t = 1.0
U = 4.0 * t
BROADENING = 0.05 * t
T_SMEARING = 0.005 * t


def epsilon_k_continuous(kx, ky, tp):
    return -2 * t * (np.cos(kx) + np.cos(ky)) - 4 * tp * np.cos(kx) * np.cos(ky)
     #return -2 * t * (np.cos(kx) + np.cos(ky)) - 2* tp * np.cos(kx + ky)

def fermi_smooth(energies, mu):
    val = (energies - mu) / T_SMEARING
    val = np.clip(val, -100, 100)
    return 1.0 / (np.exp(val) + 1.0)


def integrate_2d_simpson(f_val, x_grid, y_grid):
    return simpson(simpson(f_val, x=x_grid, axis=-1), x=y_grid, axis=-1)


def compute_chemical_potential(target_density, tp, n_grid=200):
    kx = np.linspace(-np.pi, np.pi, n_grid)
    ky = np.linspace(-np.pi, np.pi, n_grid)
    KX, KY = np.meshgrid(kx, ky)

    eps = epsilon_k_continuous(KX, KY, tp)
    norm = (2 * np.pi) ** 2

    def density_diff(mu):
        f = fermi_smooth(eps, mu)
        den = integrate_2d_simpson(f, kx, ky) / norm
        return den - target_density

    mu_min = np.min(eps) - 2.0
    mu_max = np.max(eps) + 2.0

    try:
        return brentq(density_diff, mu_min, mu_max, xtol=1e-8)
    except ValueError:
        return mu_min if density_diff(mu_min) > 0 else mu_max


def compute_free_energy(tp, mu, n_grid=200):
    kx = np.linspace(-np.pi, np.pi, n_grid)
    ky = np.linspace(-np.pi, np.pi, n_grid)
    KX, KY = np.meshgrid(kx, ky)

    eps = epsilon_k_continuous(KX, KY, tp)
    f = fermi_smooth(eps, mu)

    energy = integrate_2d_simpson(eps * f, kx, ky) / (2 * np.pi) ** 2
    return energy


def compute_interaction_energy_vectorized(tp, mu_up, mu_down, n_q=50, n_k=48):
    q_vec = np.linspace(-np.pi, np.pi, n_q)
    QX, QY = np.meshgrid(q_vec, q_vec, indexing='ij')

    k_vec = np.linspace(-np.pi, np.pi, n_k)
    KX, KY = np.meshgrid(k_vec, k_vec)

    p_vec = k_vec
    PX, PY = KX, KY

    eps_k = epsilon_k_continuous(KX, KY, tp)
    f_k_up = fermi_smooth(eps_k, mu_up)

    integrand_Q_values = np.zeros((n_q, n_q))

    for i in range(n_q):
        for j in range(n_q):
            Qx_val = QX[i, j]
            Qy_val = QY[i, j]

            KX_prime = (Qx_val - KX + np.pi) % (2 * np.pi) - np.pi
            KY_prime = (Qy_val - KY + np.pi) % (2 * np.pi) - np.pi

            eps_kprime = epsilon_k_continuous(KX_prime, KY_prime, tp)
            f_kprime_down = fermi_smooth(eps_kprime, mu_down)

            omega_grid = eps_k + eps_kprime

            PX_prime = (Qx_val - PX + np.pi) % (2 * np.pi) - np.pi
            PY_prime = (Qy_val - PY + np.pi) % (2 * np.pi) - np.pi

            eps_p = epsilon_k_continuous(PX, PY, tp)
            eps_p_prime = epsilon_k_continuous(PX_prime, PY_prime, tp)

            num_p = (1.0 - fermi_smooth(eps_p, mu_up)) * (1.0 - fermi_smooth(eps_p_prime, mu_down))

            denom_p_sum = eps_p + eps_p_prime

            omega_4d = omega_grid[:, :, np.newaxis, np.newaxis]

            denom_4d = denom_p_sum[np.newaxis, np.newaxis, :, :]
            num_4d = num_p[np.newaxis, np.newaxis, :, :]

            D = denom_4d - omega_4d

            kernel = D / (D ** 2 + BROADENING ** 2)

            integrand_chi = num_4d * kernel

            chi_int_y = simpson(integrand_chi, x=p_vec, axis=-1)
            chi_grid = simpson(chi_int_y, x=p_vec, axis=-1)

            chi_grid /= (2 * np.pi) ** 2

            numerator_grid = f_k_up * f_kprime_down

            denominator_grid = 1.0 + U * chi_grid

            final_integrand = numerator_grid / denominator_grid

            res_Q = integrate_2d_simpson(final_integrand, k_vec, k_vec) / (2 * np.pi) ** 2
            integrand_Q_values[i, j] = res_Q

    total_energy = integrate_2d_simpson(integrand_Q_values, q_vec, q_vec) / (2 * np.pi) ** 2

    return total_energy


def solve_phase_point_deltaE(args):
    tp_over_t, density, n_grid_mu = args
    tp =  tp_over_t * t

    if density <= 1e-5:
        return 0.0

    try:
        mu_para = compute_chemical_potential(density / 2.0, tp, n_grid=n_grid_mu)
        E_free_para = 2.0 * compute_free_energy(tp, mu_para, n_grid=n_grid_mu)

        E_int_para = compute_interaction_energy_vectorized(tp, mu_para, mu_para, n_q=48, n_k=48)

        E_total_para = E_free_para + U * E_int_para

        mu_ferro = compute_chemical_potential(density, tp, n_grid=n_grid_mu)
        E_free_ferro = compute_free_energy(tp, mu_ferro, n_grid=n_grid_mu)
        E_total_ferro = E_free_ferro

        return E_total_para - E_total_ferro

    except Exception as e:
        return 0.0


def main():
    tp_vals = np.linspace(-0.45, -0.55, 10)
    dens_vals = np.linspace(0.01, 0.05, 10)

    n_grid_mu = 2000

    tasks = [(tp, dens, n_grid_mu) for tp in tp_vals for dens in dens_vals]

    try:
        n_cores = int(os.environ['SLURM_CPUS_PER_TASK'])
    except:
        n_cores = mp.cpu_count()

    print(f"=== Hubbard Phase Diagram (Vectorized & Exact Omega) ===")
    print(f"Cores: {n_cores}")
    print(f"Grid: {len(tp_vals)}x{len(dens_vals)} = {len(tasks)} points")
    print(f"Integration Mesh: n_q=24, n_k=24 (Vectorized)")
    print(f"Physics: U={U / t}t, Broadening={BROADENING}, T_smear={T_SMEARING}")

    start_time = time.time()

    with mp.Pool(n_cores) as pool:
        results = list(tqdm(pool.imap(solve_phase_point_deltaE, tasks, chunksize=1), total=len(tasks)))

    elapsed = (time.time() - start_time) / 60
    print(f"Calculation finished in {elapsed:.2f} minutes.")

    delta_E_grid = np.array(results).reshape(len(tp_vals), len(dens_vals))

    filename_base = f"two_U{U:.1f}_tp{tp_vals[0]:.1f}_{tp_vals[-1]:.1f}"
    np.savez(f"results_{filename_base}.npz", tp=tp_vals, dens=dens_vals, delta_E=delta_E_grid)

    plt.figure(figsize=(10, 8))
    X, Y = np.meshgrid(dens_vals, tp_vals)

    limit = np.max(np.abs(delta_E_grid)) * 0.8
    if limit == 0: limit = 0.1
    norm = mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)

    cf = plt.contourf(X, Y, delta_E_grid, levels=100, cmap='RdBu_r', norm=norm, extend='both')
    cbar = plt.colorbar(cf)
    cbar.set_label(r'$\Delta E = E_{para} - E_{ferro}$')

    plt.contour(X, Y, delta_E_grid, levels=[0], colors='black', linewidths=2)

    plt.xlabel("Density n")
    plt.ylabel("t'/t")
    plt.title(f"Phase Diagram (Vectorized High-Precision)\nRed=Ferro, Blue=Para")

    plt.savefig(f"PhaseDiagram_{filename_base}.png", dpi=300)
    print("Done.")


if __name__ == "__main__":
    main()
