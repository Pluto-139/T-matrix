import numpy as np
import time
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import matplotlib.patches as mpatches
import os
from tqdm import tqdm
t = 1.0
U = 4.0 * t
L = 32
Omega = L * L    

def compute_phase_point(args):
    
    rho, t_prime = args
    N_total = int(rho * Omega)
    

    if N_total % 2 != 0:
        N_total -= 1

    if N_total == 0 or N_total >= Omega:
        return rho, t_prime, 0.0, 0.0, 0 

    
    kx_idx, ky_idx = np.meshgrid(np.arange(L), np.arange(L))
    kx_idx = kx_idx.flatten()
    ky_idx = ky_idx.flatten()
    

    kx_val = kx_idx * 2 * np.pi / L - np.pi
    ky_val = ky_idx * 2 * np.pi / L - np.pi
    
  
    eps = -2 * t * (np.cos(kx) + np.cos(ky)) - 2* tp * np.cos(kx + ky)
    

    sorted_order = np.argsort(eps)
    eps_sorted = eps[sorted_order]
 
    E_fm = np.sum(eps_sorted[:N_total]) / Omega
    
 
    N_half = N_total // 2
    occ_pm = sorted_order[:N_half]     
    unocc_pm = sorted_order[N_half:]   
    
    mu = eps[occ_pm[-1]]
    eps_tilde = eps - mu
    
    E_pm_kin = 2 * np.sum(eps[occ_pm]) / Omega

    px_idx = unocc_pm % L
    py_idx = unocc_pm // L
    eps_tilde_p = eps_tilde[unocc_pm]
    
    is_unocc = np.zeros((L, L), dtype=bool)
    is_unocc[py_idx, px_idx] = True
    
    interaction_sum = 0.0
 
    occ_kx = kx_idx[occ_pm]
    occ_ky = ky_idx[occ_pm]
    eps_tilde_occ = eps_tilde[occ_pm]
    
    for i in range(N_half):
        kx_i = occ_kx[i]
        ky_i = occ_ky[i]
        eps_k = eps_tilde_occ[i]
        
        for j in range(N_half):
            kx_j = occ_kx[j]
            ky_j = occ_ky[j]
            eps_kp = eps_tilde_occ[j]
            

            omega = eps_k + eps_kp
  
            qx = (kx_i + kx_j) % L
            qy = (ky_i + ky_j) % L
            
            minus_p_q_x = (qx - px_idx) % L
            minus_p_q_y = (qy - py_idx) % L
            
    
            valid_mask = is_unocc[minus_p_q_y, minus_p_q_x]
            
            if not np.any(valid_mask):
       
                interaction_sum += 1.0
                continue
            
            p_valid_idx = np.where(valid_mask)[0]
            eps_p_valid = eps_tilde_p[p_valid_idx]
      
            minus_p_q_flat = minus_p_q_y[p_valid_idx] * L + minus_p_q_x[p_valid_idx]
            eps_minus_p_q_valid = eps_tilde[minus_p_q_flat]
            
            denom = eps_p_valid + eps_minus_p_q_valid - omega

            denom[denom < 1e-12] = 1e-12    

            chi_pp = np.sum(1.0 / denom) / Omega
            
            interaction_sum += 1.0 / (1.0 + U * chi_pp)

    E_pm_int = (U / (Omega * Omega)) * interaction_sum
    E_pm = E_pm_kin + E_pm_int
    

    is_fm = 1 if E_fm < E_pm else 0
    
    return rho, t_prime, E_fm, E_pm, is_fm
    
def plot_phase_diagram(data, rho_range, tp_range, filename_prefix):

    rho = data[:, 0]
    t_prime = data[:, 1]
    is_fm = data[:, 4]
    

    rho_grid, tp_grid = np.meshgrid(
        np.linspace(rho.min(), rho.max(), 200),
        np.linspace(t_prime.min(), t_prime.max(), 200)
    )
    

    points = np.column_stack((rho, t_prime))
    phase_grid = griddata(points, is_fm, (rho_grid, tp_grid), method='linear')
    

    phase_grid_smooth = gaussian_filter(phase_grid, sigma=1.0)
    phase_grid_binary = (phase_grid_smooth > 0.5).astype(float)
    

    fig, ax = plt.subplots(figsize=(10, 8))

    cmap = ListedColormap(['white', 'lightgray'])
    
    im = ax.pcolormesh(rho_grid, tp_grid, phase_grid_binary, 
                       cmap=cmap, shading='auto', alpha=0.8)
    
    contours = ax.contour(rho_grid, tp_grid, phase_grid_smooth, 
                          levels=[0.5], colors='black', linewidths=2, alpha=0.8)
    

    ax.set_xlabel(r'Electron density $n$', fontsize=14)
    ax.set_ylabel(r'$t^{\prime}/t$', fontsize=14)
    

    ax.set_xlim(rho.min(), rho.max())
    ax.set_ylim(t_prime.min(), t_prime.max())
    
    fm_patch = mpatches.Patch(color='lightgray', label='Ferromagnetic (FM)')
    pm_patch = mpatches.Patch(color='white', label='Paramagnetic (PM)')
    ax.legend(handles=[fm_patch, pm_patch], loc='upper right', fontsize=12)
    
    ax.set_title(f'Phase diagram for U = 4t\n'
                 f'$t^{{\prime}}/t$: [{tp_range[0]:.2f}, {tp_range[1]:.2f}], '
                 f'n: [{rho_range[0]:.2f}, {rho_range[1]:.2f}]', 
                 fontsize=14)
    
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plot_filename = f"{filename_prefix}_phase.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Phase diagram saved as '{plot_filename}' and '{pdf_filename}'")
    
    return plot_filename

if __name__ == '__main__':
    rho_min, rho_max = 0.01, 1
    tp_min, tp_max = -1, 0
    n_rho = 50
    n_tp = 50
    
    rho_vals = np.linspace(rho_min, rho_max, n_rho)
    t_prime_vals = np.linspace(tp_min, tp_max, n_tp)

    tasks = [(r, tp) for tp in t_prime_vals for r in rho_vals]
    
    print(f"Total points to compute: {len(tasks)}")
    print(f"Parameter range: n ∈ [{rho_min:.2f}, {rho_max:.2f}], "
          f"t'/t ∈ [{tp_min:.2f}, {tp_max:.2f}]")
    start_time = time.time()
    
    try:
        n_cores = int(os.environ['SLURM_CPUS_PER_TASK'])
    except:
        n_cores = mp.cpu_count()
    
    with Pool(processes=n_cores) as pool:
        results = []
        with tqdm(total=len(tasks), desc="Computing phase points") as pbar:
            for result in pool.imap_unordered(compute_phase_point, tasks):
                results.append(result)
                pbar.update(1)
    
    end_time = time.time()
    computation_time = (end_time - start_time) / 60
    print(f"Computation completed in {computation_time:.2f} minutes.")
    
    results_array = np.array(results)
    
    filename_prefix = f"phase_U4_tp{tp_min:.2f}-{tp_max:.2f}_n{rho_min:.2f}-{rho_max:.2f}"
    data_filename = f"{filename_prefix}_data.dat"
    
    np.savetxt(data_filename, results_array, fmt="%.6f", 
               header="rho t_prime E_fm E_pm is_fm")
    print(f"Data saved to {data_filename}")
    plot_filename = plot_phase_diagram(results_array, 
                                      (rho_min, rho_max), 
                                      (tp_min, tp_max), 
                                      filename_prefix)
