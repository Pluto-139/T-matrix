import matplotlib
# === ¹Ø¼üÐÞ¸Ä 1: ÉèÖÃºó¶ËÎª·Ç½»»¥Ê½£¬±ØÐëÔÚ import pyplot Ö®Ç° ===
matplotlib.use('Agg') 

from scipy.integrate import trapz
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
from scipy.optimize import brentq
from tqdm import tqdm
import os

# === Constants ===
t = 1.0
U = 200.0 * t

# === Dispersion ===
def epsilon_k_continuous(kx, ky, tp):
    return -2 * t * (np.cos(kx) + np.cos(ky)) -2 * tp * np.cos(kx+ky)

def fermi_step_vectorized(energies, mu):
    return (energies <= mu).astype(float)

# === ÌÝÐÎ»ý·Ö·½·¨¼ÆËã»¯Ñ§ÊÆ ===
def compute_chemical_potential_trapz(target_density, tp, n_grid):
    kx = np.linspace(-np.pi, np.pi, n_grid)
    ky = np.linspace(-np.pi, np.pi, n_grid)
    KX, KY = np.meshgrid(kx, ky)
    eps = epsilon_k_continuous(KX, KY, tp)

    def total_density(mu):
        f = fermi_step_vectorized(eps, mu)
        integral = trapz(trapz(f, kx, axis=1), ky)
        return integral / (2 * np.pi) ** 2

    mu_min = np.min(eps) - 1.0
    mu_max = np.max(eps) + 1.0
    
    try:
        return brentq(lambda mu: total_density(mu) - target_density, mu_min, mu_max, xtol=1e-8)
    except ValueError:
        # Èç¹ûÕÒ²»µ½¸ù£¬·µ»Ø±ß½çÖµ×÷Îªfallback
        return mu_min if total_density(mu_min) > target_density else mu_max

# === ÌÝÐÎ»ý·Ö·½·¨¼ÆËã×ÔÓÉÄÜ ===
def compute_free_energy_trapz(tp, mu, n_grid):
    kx = np.linspace(-np.pi, np.pi, n_grid)
    ky = np.linspace(-np.pi, np.pi, n_grid)
    KX, KY = np.meshgrid(kx, ky)
    eps = epsilon_k_continuous(KX, KY, tp)
    f = fermi_step_vectorized(eps, mu)
    integrand = eps * f
    integral = trapz(trapz(integrand, kx, axis=1), ky)
    return integral / (2 * np.pi) ** 2

# === Monte Carlo ¦Ö_pp ===
def compute_chi_pp_monte_carlo(qx, qy, omega, mu_up, mu_down, tp, n_samples=70000):
    # Ê¹ÓÃËæ»úÖÖ×ÓÈ·±£¿É¸´ÏÖÐÔ£¬µ«ÔÚ²¢ÐÐÖÐÐèÒª±ä»¯
    np.random.seed(int((qx * 1000 + qy * 1000 + time.time()) % 2 ** 32))
    px = np.random.uniform(-np.pi, np.pi, n_samples)
    py = np.random.uniform(-np.pi, np.pi, n_samples)

    px_q = (qx - px) % (2 * np.pi)
    py_q = (qy - py) % (2 * np.pi)

    eps_p = epsilon_k_continuous(px, py, tp)
    eps_pq = epsilon_k_continuous(px_q, py_q, tp)

    f_p = fermi_step_vectorized(eps_p, mu_up)
    f_pq = fermi_step_vectorized(eps_pq, mu_down)

    num = (1 - f_p) * (1 - f_pq)
    denom = (eps_p - mu_up) + (eps_pq - mu_down) - omega
    mask = np.abs(denom) > 1e-10

    if np.sum(mask) == 0:
        return 0.0

    return np.mean((num / denom)[mask])

# === ÃÉÌØ¿¨Âå·½·¨¼ÆËãÏà»¥×÷ÓÃÄÜ ===
def compute_interaction_monte_carlo(tp, mu_up, mu_down, n_samples=10000):
    np.random.seed(None) # ÈÃÃ¿¸ö½ø³ÌÓÉÏµÍ³Ê±¼ä¾ö¶¨ÖÖ×Ó

    k1x = np.random.uniform(-np.pi, np.pi, n_samples)
    k1y = np.random.uniform(-np.pi, np.pi, n_samples)
    k2x = np.random.uniform(-np.pi, np.pi, n_samples)
    k2y = np.random.uniform(-np.pi, np.pi, n_samples)

    eps1 = epsilon_k_continuous(k1x, k1y, tp)
    eps2 = epsilon_k_continuous(k2x, k2y, tp)
    f1 = fermi_step_vectorized(eps1, mu_up)
    f2 = fermi_step_vectorized(eps2, mu_down)

    qx = (k1x + k2x) % (2 * np.pi)
    qy = (k1y + k2y) % (2 * np.pi)

    omega = (eps1 - mu_up) + (eps2 - mu_down)
    mask = (f1 > 0.5) & (f2 > 0.5)
    if np.sum(mask) == 0:
        return 0.0

    values = np.zeros(n_samples)
    # ÓÅ»¯£ºÖ»¶Ô mask Îª True µÄ²¿·Ö½øÐÐÑ­»·»òÏòÁ¿»¯¼ÆËã
    indices = np.where(mask)[0]
    # ÕâÀïÎªÁËËÙ¶È£¬¿ÉÒÔ¼õÉÙÄÚ²ãÑ­»·µÄ sample Êý»òÕß±£³ÖÏÖ×´
    for i in indices:
        chi = compute_chi_pp_monte_carlo(qx[i], qy[i], omega[i], mu_up, mu_down, tp, n_samples=5000)
        values[i] = f1[i] * f2[i] / (1.0 + U * chi)
    return np.mean(values)

# === Ê¹ÓÃÌÝÐÎ»ý·Ö+ÃÉÌØ¿¨Âå»ìºÏ·½·¨¼ÆËã×ÜÄÜÁ¿ ===
def compute_total_energy_hybrid(density_up, density_down, tp, n_grid, N=32):
    mu_up = compute_chemical_potential_trapz(density_up, tp, n_grid)
    mu_down = compute_chemical_potential_trapz(density_down, tp, n_grid)

    E_free_up = compute_free_energy_trapz(tp, mu_up, n_grid)
    E_free_down = compute_free_energy_trapz(tp, mu_down, n_grid)
    E_free = E_free_up + E_free_down

    E_int = compute_interaction_monte_carlo(tp, mu_up, mu_down)

    return E_free + U * E_int

# === Phase determination ===
def determine_phase_hybrid(tp_over_t, density, n_grid, N=32):
    tp = tp_over_t * t
    if density == 0:
        return 0

    try:
        E_para = compute_total_energy_hybrid(density / 2, density / 2, tp, n_grid, N)
        E_ferro = compute_total_energy_hybrid(density, 0.0, tp, n_grid, N)
        return 1 if E_para >= E_ferro else 0
    except Exception as e:
        # print(f"Error at tp={tp}, den={density}: {e}")
        return 0

def determine_phase_wrapper(args):
    tp_over_t, density, n_grid, N = args
    return determine_phase_hybrid(tp_over_t, density, n_grid, N)

# === Phase diagram ===
def generate_phase_diagram_hybrid(n_grid=100, N=32):
    # ¸ù¾ÝÐèÒªµ÷Õû²ÎÊý·¶Î§
    tp_values = np.linspace(-1, 0, 30)
    density_values = np.linspace(0, 0.04, 30)
    
    # ¹¹½¨ÈÎÎñÁÐ±í
    params = [(tp, dens, n_grid, N) for tp in tp_values for dens in density_values]
    total_points = len(params)
    
    # === ¹Ø¼üÐÞ¸Ä 2: »ñÈ¡ SLURM ·ÖÅäµÄºËÊý ===
    # Èç¹ûÔÚ SLURM »·¾³ÖÐ£¬Ê¹ÓÃÉêÇëµÄ CPU ºËÊý£»·ñÔòÊ¹ÓÃ±¾µØËùÓÐºË
    num_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', mp.cpu_count()))
    print(f"Running on {num_workers} cores. Total points: {total_points}")

    with mp.Pool(processes=num_workers) as pool:
        # ×¢Òâ£ºÔÚ Cluster ÉÏ tqdm ¿ÉÄÜ»áµ¼ÖÂÈÕÖ¾ÎÄ¼þ±äµÃ¾Þ´óÇÒ»ìÂÒ
        # Èç¹ûÈÕÖ¾ÂÒÂë£¬¿ÉÒÔ°Ñ tqdm È¥µô£¬»»³É¼òµ¥µÄ print
        results = list(tqdm(
            pool.imap(determine_phase_wrapper, params),
            total=total_points,
            desc="Calculating",
            unit="pts"
        ))

    return tp_values, density_values, np.array(results).reshape(len(tp_values), len(density_values))

def plot_phase_diagram(tp_values, density_values, phase_grid):
    plt.figure(figsize=(12, 8))
    X, Y = np.meshgrid(density_values, tp_values)

    ferro_mask = phase_grid == 1
    plt.scatter(X[ferro_mask], Y[ferro_mask], c='black', s=15, label='Ferromagnetic', alpha=0.8)

    plt.ylabel("t'/t", fontsize=14)
    plt.xlabel("Density", fontsize=14)
    plt.title(f"Phase Diagram (U={U / t:.1f}t)", fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=16)
    
    plt.xlim(np.min(density_values), np.max(density_values))
    plt.ylim(np.min(tp_values), np.max(tp_values))
    
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # === ¹Ø¼üÐÞ¸Ä 3: ±£´æÍ¼Æ¬¶ø²»ÊÇÏÔÊ¾ ===
    output_filename = "phase_diagram.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Plot saved to {output_filename}")

def main():
    print("=== Monte Carlo Hubbard Calculation on Cluster ===")
    N = 32
    start = time.time()

    # ÕâÀïµÄ n_grid ºÍ N ¿ÉÒÔ¸ù¾Ý cluster µÄËãÁ¦ÊÊµ±µ÷´ó
    tp_vals, dens_vals, phase = generate_phase_diagram_hybrid(n_grid=500, N=N)

    # === ¹Ø¼üÐÞ¸Ä 4: ±£´æÔ­Ê¼Êý¾Ý ===
    # ÕâÑù¼´Ê¹»­Í¼²»ÂúÒâ£¬ÒÔºóÒ²¿ÉÒÔÖ±½Ó¶ÁÈ¡ .npz ÎÄ¼þÖØ»­£¬²»ÓÃÖØÐÂ¼ÆËã
    np.savez("simulation_result.npz", tp=tp_vals, density=dens_vals, phase=phase)
    print("Raw data saved to simulation_result.npz")

    print(f"Computed {phase.size} points in {time.time() - start:.2f}s")
    plot_phase_diagram(tp_vals, dens_vals, phase)

if __name__ == "__main__":
    main()
