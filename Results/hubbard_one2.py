import matplotlib
# Set backend to Agg, suitable for Cluster headless mode
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

# ==========================================
# 1. Physical Constants and Numerical Parameters
# ==========================================
t = 1.0
U = 200.0 * t

# [Key Parameter 1] Broadening Factor (Broadening)
# Used to handle the pole in the Green's function denominator (1/E), preventing division by zero.
# Physically corresponds to quasiparticle lifetime or scattering rate.
BROADENING = 0.05 * t 

# [Key Parameter 2] Numerical Smearing Temperature (Smearing Temperature)
# Used to smooth the T=0 step function, allowing Simpson integration to achieve high-order accuracy.
# Choose a very small value, ensuring physical approximation to T=0 while maintaining numerical smoothness.
T_SMEARING = 0.001 * t 

# ==========================================
# 2. Basic Physical Functions
# ==========================================

def epsilon_k_continuous(kx, ky, tp):
    """Tight-binding model dispersion relation"""
    return -2 * t * (np.cos(kx) + np.cos(ky)) - 2 * tp * np.cos(kx + ky)

def fermi_smooth(energies, mu):
    """
    Smooth Fermi-Dirac distribution function.
    Replaces the original (energies <= mu).astype(float)
    """
    # Compute (E - mu) / T
    # Use clip to prevent exp overflow
    val = (energies - mu) / T_SMEARING
    val = np.clip(val, -100, 100) 
    return 1.0 / (np.exp(val) + 1.0)

def integrate_2d_simpson(f_val, x_grid, y_grid):
    """2D Simpson Integration Tool"""
    # Integrate over the last dimension (kx)
    int_x = simpson(f_val, x=x_grid, axis=-1)
    # Integrate over the second-to-last dimension (ky)
    int_y = simpson(int_x, x=y_grid)
    return int_y

# ==========================================
# 3. High-Precision Calculation Module
# ==========================================

def compute_chemical_potential(target_density, tp, n_grid=1000):
    """
    Compute chemical potential mu.
    Because T_SMEARING and Simpson are used, n_grid doesn't need to be extremely large for good accuracy here.
    """
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
        return brentq(density_diff, mu_min, mu_max, xtol=1e-10)
    except ValueError:
        # Rare case where root is not found (e.g., extremely low density), return boundary
        return mu_min if density_diff(mu_min) > 0 else mu_max

def compute_free_energy(tp, mu, n_grid=1000):
    """Compute free energy (kinetic term contribution)"""
    kx = np.linspace(-np.pi, np.pi, n_grid)
    ky = np.linspace(-np.pi, np.pi, n_grid)
    KX, KY = np.meshgrid(kx, ky)
    
    eps = epsilon_k_continuous(KX, KY, tp)
    f = fermi_smooth(eps, mu)
    
    # Free energy density = Integral( epsilon * f )
    energy = integrate_2d_simpson(eps * f, kx, ky) / (2 * np.pi) ** 2
    return energy

def compute_chi_pp_integral(Qx, Qy, mu_up, mu_down, tp, k_vec, KX, KY):
    """
    Compute polarizability Chi_pp (Pair Susceptibility).
    Use Simpson integration instead of summation.
    """
    # Momentum conservation k' = Q - k
    KX_prime = (Qx - KX + np.pi) % (2 * np.pi) - np.pi
    KY_prime = (Qy - KY + np.pi) % (2 * np.pi) - np.pi
    
    eps_k = epsilon_k_continuous(KX, KY, tp)
    eps_kprime = epsilon_k_continuous(KX_prime, KY_prime, tp)
    
    f_k = fermi_smooth(eps_k, mu_up)
    f_kprime = fermi_smooth(eps_kprime, mu_down)
    
    # Denominator (Static approximation, Omega=0)
    denom = (eps_k - mu_up) + (eps_kprime - mu_down)
    
    # Integrand: Include Lorentzian Broadening to prevent denominator from being 0
    # Re[ 1 / (E - i*eta) ] = E / (E^2 + eta^2)
    numerator = (1.0 - f_k) * (1.0 - f_kprime)
    kernel = denom / (denom**2 + BROADENING**2)
    
    integrand = numerator * kernel
    
    # Integration
    chi_val = integrate_2d_simpson(integrand, k_vec, k_vec) / (2 * np.pi)**2
    return chi_val

def compute_interaction_energy(tp, mu_up, mu_down, n_q=100, n_k=100):
    """
    Compute interaction energy Correction.
    Double integration: outer over Q, inner over k.
    """
    # 1. Q Grid (outer integration grid)
    q_vec = np.linspace(-np.pi, np.pi, n_q)
    QX, QY = np.meshgrid(q_vec, q_vec, indexing='ij')
    
    # 2. k Grid (inner integration grid, used for computing Chi and numerator)
    k_vec = np.linspace(-np.pi, np.pi, n_k)
    KX, KY = np.meshgrid(k_vec, k_vec)
    
    integrand_Q_values = np.zeros((n_q, n_q))
    
    # Loop over Q space
    for i in range(n_q):
        for j in range(n_q):
            Qx_val = QX[i, j]
            Qy_val = QY[i, j]
            
            # Compute Chi(Q)
            chi_val = compute_chi_pp_integral(Qx_val, Qy_val, mu_up, mu_down, tp, k_vec, KX, KY)
            
            # Compute numerator integral: Integral[ f_k * f_{Q-k} ]
            KX_prime = (Qx_val - KX + np.pi) % (2 * np.pi) - np.pi
            KY_prime = (Qy_val - KY + np.pi) % (2 * np.pi) - np.pi
            
            eps_k = epsilon_k_continuous(KX, KY, tp)
            eps_kprime = epsilon_k_continuous(KX_prime, KY_prime, tp)
            
            f1 = fermi_smooth(eps_k, mu_up)
            f2 = fermi_smooth(eps_kprime, mu_down)
            
            numerator_integral = integrate_2d_simpson(f1 * f2, k_vec, k_vec) / (2 * np.pi)**2
            
            # Combine RPA formula term
            if numerator_integral > 1e-12:
                integrand_Q_values[i, j] = numerator_integral / (1.0 + U * chi_val)
            else:
                integrand_Q_values[i, j] = 0.0
                
    # Integrate over Q space to get final energy
    total_energy = integrate_2d_simpson(integrand_Q_values, q_vec, q_vec) / (2 * np.pi)**2
    return total_energy

# ==========================================
# 4. Parallel Task Wrapper
# ==========================================

def solve_phase_point_deltaE(args):
    """
    Compute energy difference for a single (tp, density) point.
    Returns: Delta_E = E_para - E_ferro
    """
    tp_over_t, density, n_grid_mu = args
    tp = tp_over_t * t
    
    # Handle extremely low density
    if density <= 1e-6:
        return 0.0

    try:
        # --- 1. Compute paramagnetic state (Paramagnetic) ---
        # Spin up and down densities equal n/2
        mu_para = compute_chemical_potential(density/2.0, tp, n_grid=n_grid_mu)
        
        # Free energy (double, because of up and down)
        E_free_para = 2.0 * compute_free_energy(tp, mu_para, n_grid=n_grid_mu)
        
        # Interaction energy (time-consuming step)
        # On Cluster, suggest n_q=50, n_k=60. For extremely high accuracy, can set n_q=60, n_k=80.
        E_int_para = compute_interaction_energy(tp, mu_para, mu_para, n_q=50, n_k=60)
        
        E_total_para = E_free_para + U * E_int_para
        
        # --- 2. Compute ferromagnetic state (Ferromagnetic) ---
        # Fully polarized: n_up = n, n_down = 0
        mu_ferro = compute_chemical_potential(density, tp, n_grid=n_grid_mu)
        
        # Free energy (only up, down is empty)
        E_free_ferro = compute_free_energy(tp, mu_ferro, n_grid=n_grid_mu)
        
        # Ferromagnetic interaction energy is 0 (Pauli Exclusion)
        E_total_ferro = E_free_ferro
        
        # --- 3. Return energy difference ---
        # Delta > 0: Para energy higher -> system tends to Ferro
        # Delta < 0: Para energy lower -> system tends to Para
        return E_total_para - E_total_ferro
        
    except Exception as e:
        # print(f"Error: {e}")
        return 0.0

# ==========================================
# 5. Main Program and Plotting
# ==========================================

def main():
    # ---------------- Configuration Parameters ----------------
    # Increase resolution for a better-looking gradient plot
    tp_vals = np.linspace(-0.58, -0.45, 30)
    dens_vals = np.linspace(0.001, 0.1,40)
    
    n_grid_mu = 2000 # Chemical potential integration grid
    
    # Build tasks
    tasks = [(tp, dens, n_grid_mu) for tp in tp_vals for dens in dens_vals]
    
    # Get Cluster core count
    try:
        n_cores = int(os.environ['SLURM_CPUS_PER_TASK'])
    except:
        n_cores = mp.cpu_count()
        
    print(f"=== Gradient Phase Diagram Calculation ===")
    print(f"Cores: {n_cores}")
    print(f"Resolution: {len(tp_vals)}x{len(dens_vals)} points")
    print(f"Smoothing: T={T_SMEARING}, Broadening={BROADENING}")

    # ---------------- Parallel Computation ----------------
    start_time = time.time()
    with mp.Pool(n_cores) as pool:
        # Use imap to show progress bar, list enforces execution and collects results in order
        results = list(tqdm(pool.imap(solve_phase_point_deltaE, tasks), total=len(tasks)))
        
    print(f"Calculation finished in {(time.time()-start_time)/60:.2f} minutes.")

    # ---------------- Data Processing ----------------
    delta_E_grid = np.array(results).reshape(len(tp_vals), len(dens_vals))
    
    # Save raw data
    np.savez("gradient_results.npz", tp=tp_vals, dens=dens_vals, delta_E=delta_E_grid)
    print("Data saved to gradient_results.npz")

    # ---------------- Gradient Plot (Gradient Plot) ----------------
    plt.figure(figsize=(12, 9))
    X, Y = np.meshgrid(dens_vals, tp_vals)
    
    # Set colormap center
    # Use Diverging Colormap (RdBu_r): Red-White-Blue
    # Red = Positive value (Delta E > 0) = Ferro Stable
    # Blue = Negative value (Delta E < 0) = Para Stable
    # White = 0 = Phase Boundary
    
    # For better visual effect, we typically apply a small truncation to Delta E to prevent extreme values from obscuring details.
    limit = np.max(np.abs(delta_E_grid)) * 0.6 # Take 60% of the maximum value as the display range
    norm = mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)

    # Draw smooth filled contour plot (Contourf)
    # levels=100 ensures very smooth color transition
    cf = plt.contourf(X, Y, delta_E_grid, levels=100, cmap='RdBu_r', norm=norm, extend='both')
    
    # Add color bar
    cbar = plt.colorbar(cf)
    cbar.set_label(r'$\Delta E = E_{para} - E_{ferro}$', fontsize=14)
    # Mark phase regions on Colorbar
    cbar.ax.text(0.5, 0.95, 'Ferro', transform=cbar.ax.transAxes, ha='center', color='black', fontsize=10, fontweight='bold')
    cbar.ax.text(0.5, 0.05, 'Para', transform=cbar.ax.transAxes, ha='center', color='black', fontsize=10, fontweight='bold')

    # Add phase transition boundary line (Delta E = 0)
    # linewidths=2, black solid line
    boundary = plt.contour(X, Y, delta_E_grid, levels=[0], colors='black', linewidths=2.5)
    plt.clabel(boundary, inline=True, fmt='Boundary', fontsize=12)

    # Axis labels
    plt.xlabel(r"Density $n$", fontsize=16)
    plt.ylabel(r"$t'/t$", fontsize=16)
    plt.title(f"Hubbard Phase Diagram (U={U/t}t)\nColor Intensity = Energetic Stability", fontsize=16)
    
    # Save image
    plt.tight_layout()
    plt.savefig("phase_diagram_gradient.png", dpi=300)
    print("Plot saved to phase_diagram_gradient.png")

if __name__ == "__main__":
    main()
