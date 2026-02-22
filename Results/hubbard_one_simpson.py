#This  a first version  code for calculation in the cluster ,it use simpson instead of accurate integral,and for the calculation results, it return the 0 or 1 instead of \delta E
# ,so in next verison ,the reults will return the differnece .
import matplotlib

matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
import os
from scipy.integrate import simpson
from scipy.optimize import brentq
from tqdm import tqdm

# === Physical Parameters ===
t = 1.0
U = 200.0 * t
# BROADENING (eta) is the key to the thermodynamic limit.
# In continuous integration, the pole in the denominator requires an infinitesimal imaginary part i*eta.
# Numerically we take a small value (e.g., 0.02t) to ensure smoothness while approaching the physical limit.
BROADENING = 0.02 * t 

# === Dispersion Relation ===
def epsilon_k_continuous(kx, ky, tp):
    return -2 * t * (np.cos(kx) + np.cos(ky)) - 2 * tp * np.cos(kx + ky)

def fermi_step_vectorized(energies, mu):
    # Step function at T=0
    return (energies <= mu).astype(float)

# === Core Tool: 2D Simpson Integration ===
# This implements integral dk_x dk_y / (2pi)^2
def integrate_2d_simpson(f_val, x_grid, y_grid):
    # First integration (over the last axis, typically kx)
    integral_x = simpson(f_val, x=x_grid, axis=-1)
    # Second integration (over the remaining axis, typically ky)
    integral_y = simpson(integral_x, x=y_grid)
    return integral_y

# === 1. Chemical Potential Calculation (Integration Method) ===
def compute_chemical_potential(target_density, tp, n_grid=400):
    kx = np.linspace(-np.pi, np.pi, n_grid)
    ky = np.linspace(-np.pi, np.pi, n_grid)
    KX, KY = np.meshgrid(kx, ky)
    eps = epsilon_k_continuous(KX, KY, tp)
    normalization = (2 * np.pi) ** 2

    def total_density(mu):
        f = fermi_step_vectorized(eps, mu)
        # Integration instead of summation
        return integrate_2d_simpson(f, kx, ky) / normalization

    mu_min, mu_max = np.min(eps) - 1.0, np.max(eps) + 1.0
    try:
        return brentq(lambda mu: total_density(mu) - target_density, mu_min, mu_max, xtol=1e-10)
    except ValueError:
        return mu_min if total_density(mu_min) > target_density else mu_max

# === 2. Free Energy Calculation (Integration Method) ===
def compute_free_energy(tp, mu, n_grid=400):
    kx = np.linspace(-np.pi, np.pi, n_grid)
    ky = np.linspace(-np.pi, np.pi, n_grid)
    KX, KY = np.meshgrid(kx, ky)
    eps = epsilon_k_continuous(KX, KY, tp)
    f = fermi_step_vectorized(eps, mu)
    
    # E = integral( eps * f )
    return integrate_2d_simpson(eps * f, kx, ky) / (2 * np.pi) ** 2

# === 3. Polarizability Chi_pp (Integration Method) ===
def compute_chi_pp_integral(Qx, Qy, mu_up, mu_down, tp, k_vec, KX, KY):
    """
    Compute Chi_pp(Q).
    To achieve the thermodynamic limit, this must be an integral over k, not a simple summation.
    """
    # Q - k
    KX_prime = (Qx - KX + np.pi) % (2 * np.pi) - np.pi
    KY_prime = (Qy - KY + np.pi) % (2 * np.pi) - np.pi
    
    eps_k = epsilon_k_continuous(KX, KY, tp)
    eps_kprime = epsilon_k_continuous(KX_prime, KY_prime, tp)
    
    f_k = fermi_step_vectorized(eps_k, mu_up)
    f_kprime = fermi_step_vectorized(eps_kprime, mu_down)
    
    # Static Approximation (Omega=0)
    # denominator = E_k + E_{Q-k} - (mu_up + mu_down)
    denominator = (eps_k - mu_up) + (eps_kprime - mu_down)
    
    # Integrand (including Lorentzian broadening)
    integrand = (1 - f_k) * (1 - f_kprime) * (denominator / (denominator**2 + BROADENING**2))
    
    # Use Simpson integration instead of averaging
    # Result divided by (2pi)^2 for normalization
    chi_val = integrate_2d_simpson(integrand, k_vec, k_vec) / (2 * np.pi)**2
    return chi_val

# === 4. Interaction Energy (Double Integration Method) ===
def compute_interaction_energy_integral(tp, mu_up, mu_down, n_q=50, n_k=60):
    """
    True thermodynamic limit calculation:
    E_int ~ Integral_Q [ Integral_k ( ... ) ]
    """
    # 1. Prepare Q-space grid (for outer integration)
    q_vec = np.linspace(-np.pi, np.pi, n_q)
    QX, QY = np.meshgrid(q_vec, q_vec, indexing='ij') # indexing='ij' for simpson axis alignment
    
    # 2. Prepare k-space grid (for inner Chi calculation)
    # n_k is slightly larger to ensure accuracy of Chi
    k_vec = np.linspace(-np.pi, np.pi, n_k)
    KX, KY = np.meshgrid(k_vec, k_vec)
    
    # We need to compute the integrand value at each Q point, store in an array, and finally integrate the array.
    # Integrand(Q) = [Integral_P f*f] / (1 + U * Chi(Q))
    
    integrand_Q_values = np.zeros((n_q, n_q))
    
    # This is a relatively heavy loop, but acceptable for a Cluster.
    # We could compute only the irreducible zone, but full grid is safest.
    for i in range(n_q):
        for j in range(n_q):
            Qx_val = QX[i, j]
            Qy_val = QY[i, j]
            
            # --- A. Compute Chi(Q) in denominator (inner integration) ---
            chi_val = compute_chi_pp_integral(Qx_val, Qy_val, mu_up, mu_down, tp, k_vec, KX, KY)
            
            # --- B. Compute numerator integral (inner integration) ---
            # numerator = Integral_k [ f(k) * f(Q-k) ]
            # Note: the k-grid here can be the same as for Chi calculation.
            KX_prime = (Qx_val - KX + np.pi) % (2 * np.pi) - np.pi
            KY_prime = (Qy_val - KY + np.pi) % (2 * np.pi) - np.pi
            
            eps_k = epsilon_k_continuous(KX, KY, tp)
            eps_kprime = epsilon_k_continuous(KX_prime, KY_prime, tp)
            f1 = fermi_step_vectorized(eps_k, mu_up)
            f2 = fermi_step_vectorized(eps_kprime, mu_down)
            
            # Compute the integral of the Cooper Pair Bubble
            numerator_integral = integrate_2d_simpson(f1 * f2, k_vec, k_vec) / (2 * np.pi)**2
            
            # --- C. Combine to get integrand value ---
            # Meaningful only when numerator is non-zero
            if numerator_integral > 1e-10:
                integrand_Q_values[i, j] = numerator_integral / (1.0 + U * chi_val)
            else:
                integrand_Q_values[i, j] = 0.0

    # 3. Final integration over Q space
    # Result = Integral_Q [ ... ]
    total_energy = integrate_2d_simpson(integrand_Q_values, q_vec, q_vec) / (2 * np.pi)**2
    
    return total_energy

# === Parallel Task Wrapper ===
def solve_phase_point(args):
    tp_over_t, density, n_grid_base = args
    tp = tp_over_t * t
    
    if density <= 1e-6: return 0

    try:
        # Cluster configuration:
        # n_grid_base for chemical potential (suggest 400+)
        # n_q, n_k for interaction energy (suggest 50, 60. This means each phase point does 50*50*60*60 ≈ 9 million calculations)
        # A 32-core Cluster can handle this without issue.
        
        # 1. Paramagnetic calculation
        mu_para = compute_chemical_potential(density/2.0, tp, n_grid=n_grid_base)
        E_free_para = 2 * compute_free_energy(tp, mu_para, n_grid=n_grid_base)
        E_int_para = compute_interaction_energy_integral(tp, mu_para, mu_para, n_q=50, n_k=60)
        E_para = E_free_para + U * E_int_para
        
        # 2. Ferromagnetic calculation (E_int = 0)
        mu_ferro = compute_chemical_potential(density, tp, n_grid=n_grid_base)
        E_ferro = compute_free_energy(tp, mu_ferro, n_grid=n_grid_base)
        
        return 1 if E_para >= E_ferro else 0
        
    except Exception as e:
        return 0

# === Main Program ===
def main():
    # Resolution settings
    tp_vals = np.linspace(-0.6,-0.3,30)
    dens_vals = np.linspace(0,0.15,30)
    
    # Task list
    tasks = [(tp, dens, 400) for tp in tp_vals for dens in dens_vals]
    
    # Environment detection
    try:
        n_cores = int(os.environ['SLURM_CPUS_PER_TASK'])
    except:
        n_cores = mp.cpu_count()
    
    print(f"=== Thermodynamic Limit Calculation ===")
    print(f"Cores: {n_cores}")
    print(f"Broadening (eta): {BROADENING}")
    print(f"Integrator: Simpson's Rule (4th Order)")
    
    with mp.Pool(n_cores) as pool:
        results = list(tqdm(pool.imap(solve_phase_point, tasks), total=len(tasks)))
        
    phase_mat = np.array(results).reshape(len(tp_vals), len(dens_vals))
    
    np.savez("thermodynamic_limit_results.npz", tp=tp_vals, dens=dens_vals, phase=phase_mat)
    
    # Plotting
    plt.figure(figsize=(10, 8))
    X, Y = np.meshgrid(dens_vals, tp_vals)
    plt.contourf(X, Y, phase_mat, levels=[-0.1, 0.5, 1.1], cmap='bwr', alpha=0.3)
    plt.scatter(X[phase_mat==1], Y[phase_mat==1], c='red', s=10, label='Ferro')
    plt.scatter(X[phase_mat==0], Y[phase_mat==0], c='blue', s=2, label='Para')
    plt.xlabel("Density")
    plt.ylabel("t'/t")
    plt.title("Phase Diagram (Integral Method)")
    plt.savefig("phase_diagram_integral.png", dpi=300)

if __name__ == "__main__":
    main()
