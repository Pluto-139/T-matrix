import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import multiprocessing as mp
import time
import os
from scipy.integrate import simpson
from tqdm import tqdm

# --- 物理参数 ---
t = 1.0
U = 4.0 * t

def get_fractional_occupation(eps_flat, N_req, tol=1e-10):
    """
    计算保留晶格对称性的分数占据数 (T=0 的正则系综平均)
    解决有限网格下简并壳层未填满导致的巨大数值波动
    """
    if N_req <= 0:
        return np.zeros_like(eps_flat), np.min(eps_flat)
        
    sort_idx = np.argsort(eps_flat)
    eps_sorted = eps_flat[sort_idx]
    
    # 找到第 N_req 个电子所处的能级作为费米能 (化学势)
    mu = eps_sorted[N_req - 1]
    
    # 严格低于 mu 的态完全占据
    below_mu = eps_flat < mu - tol
    # 刚好等于 mu 的简并态
    at_mu = np.abs(eps_flat - mu) <= tol
    
    n_below = np.sum(below_mu)
    n_at = np.sum(at_mu)
    n_needed = N_req - n_below
    
    f = np.zeros_like(eps_flat)
    f[below_mu] = 1.0
    
    # 将剩余的电子均匀分配给处于费米面上的所有简并态
    if n_at > 0:
        f[at_mu] = n_needed / float(n_at)
        
    return f, mu

def solve_phase_point_eq4_discrete(args):
    tp_over_t, target_density, L = args
    tp = tp_over_t * t
    Omega = L * L

    N_tot = int(target_density * Omega)
    if N_tot % 2 != 0:
        N_tot -= 1
        
    if N_tot <= 0 or N_tot >= Omega:
        return 0.0
        
    actual_density = N_tot / Omega
    N_half = N_tot // 2

    kx = np.arange(L) * 2 * np.pi / L - np.pi
    ky = np.arange(L) * 2 * np.pi / L - np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    
    eps = -2 * t * (np.cos(KX) + np.cos(KY)) - 4 * tp * np.cos(KX) * np.cos(KY)
    eps_flat = eps.flatten()

    # --- 核心优化：使用分数阶闭壳层填充，消除低密度波动 ---
    f_PM, mu_PM = get_fractional_occupation(eps_flat, N_half)
    f_FM, _ = get_fractional_occupation(eps_flat, N_tot)
    
    eps_tilde = eps_flat - mu_PM

    E_kin_PM = 2.0 * np.sum(f_PM * eps_flat) / Omega
    E_FM = np.sum(f_FM * eps_flat) / Omega

    N_omega = 60
    Gamma = 4.0 * t
    x_grid = np.linspace(0, np.pi/2 - 1e-6, N_omega)
    omega = Gamma * np.tan(x_grid)
    d_omega_dx = Gamma / (np.cos(x_grid)**2)
    omega_2d = omega[np.newaxis, :]  

    px = np.arange(L)
    py = np.arange(L)
    PX, PY = np.meshgrid(px, py, indexing='ij')
    PX_flat = PX.flatten()
    PY_flat = PY.flatten()

    integral_q_sum = 0.0

    for q_flat in range(Omega):
        qx = q_flat // L
        qy = q_flat % L

        minus_p_q_x = (qx - PX_flat) % L
        minus_p_q_y = (qy - PY_flat) % L
        minus_p_q_idx = minus_p_q_x * L + minus_p_q_y

        f_minus_p_q = f_PM[minus_p_q_idx]
        eps_tilde_minus_p_q = eps_tilde[minus_p_q_idx]

        E_pq = eps_tilde + eps_tilde_minus_p_q
        N_pq = 1.0 - f_PM - f_minus_p_q

        E_pq_2d = E_pq[:, np.newaxis]
        N_pq_2d = N_pq[:, np.newaxis]

        denom = E_pq_2d**2 + omega_2d**2
        denom = np.where(denom < 1e-12, 1e-12, denom)

        chi1_integrand = N_pq_2d * E_pq_2d / denom
        chi2_integrand = N_pq_2d * omega_2d / denom

        chi1 = np.sum(chi1_integrand, axis=0) / Omega
        chi2 = np.sum(chi2_integrand, axis=0) / Omega

        log_term = np.log((1.0 + U * chi1)**2 + (U * chi2)**2)
        integral_omega = simpson(log_term * d_omega_dx, x=x_grid) / (2 * np.pi)
        
        integral_q_sum += integral_omega

    E_PM_int = - (U / 2.0) * (1.0 - actual_density) + (integral_q_sum / Omega)
    E_PM = E_kin_PM + E_PM_int

    return E_PM - E_FM

def main():
    tp_vals = np.linspace(-0.40, -0.50, 20)
    dens_vals = np.linspace(0.15, 0.65, 20)
    
    L_lattice = 32  

    tasks = [(tp, dens, L_lattice) for tp in tp_vals for dens in dens_vals]

    try:
        n_cores = int(os.environ.get('SLURM_CPUS_PER_TASK', mp.cpu_count()))
    except:
        n_cores = mp.cpu_count()

    print(f"=== Hubbard Phase Diagram (Symmetry Preserved Discrete) ===")
    print(f"Cores: {n_cores}")
    print(f"Grid: {len(tp_vals)}x{len(dens_vals)} = {len(tasks)} points")

    start_time = time.time()

    with mp.Pool(n_cores) as pool:
        results = list(tqdm(pool.imap(solve_phase_point_eq4_discrete, tasks, chunksize=1), total=len(tasks)))

    elapsed = (time.time() - start_time) / 60
    print(f"Calculation finished in {elapsed:.2f} minutes.")

    delta_E_grid = np.array(results).reshape(len(tp_vals), len(dens_vals))

    filename_base = f"SymmetryEq4_L{L_lattice}_U{U:.1f}_tp{tp_vals[0]:.1f}_{tp_vals[-1]:.1f}"
    np.savez(f"results_{filename_base}.npz", tp=tp_vals, dens=dens_vals, delta_E=delta_E_grid)

    # --- 绘图逻辑优化：严格二值化 (白与灰) ---
    plt.figure(figsize=(10, 8))
    X, Y = np.meshgrid(dens_vals, tp_vals)

    # 用 contourf 画出两块纯色区域：负值(顺磁)用白色，正值(铁磁)用浅灰色
    cmap = mcolors.ListedColormap(['white', 'lightgray'])
    bounds = [-np.inf, 0, np.inf]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    plt.contourf(X, Y, delta_E_grid, levels=bounds, colors=['white', 'lightgray'])
    
    # 绘制严格的零等能线作为相界边界线
    plt.contour(X, Y, delta_E_grid, levels=[0], colors='black', linewidths=2)

    # 添加自定义图例
    fm_patch = mpatches.Patch(color='lightgray', label='Ferromagnetic (FM)')
    pm_patch = mpatches.Patch(color='white', label='Paramagnetic (PM)')
    plt.legend(handles=[fm_patch, pm_patch], loc='upper right', framealpha=0.9)

    plt.xlabel("Density $n$", fontsize=14)
    plt.ylabel("$t^{\prime}/t$", fontsize=14)
    plt.title(f"Phase Diagram ($U={U/t}t$, $L={L_lattice}$ Grid)\nEq 4 with Fractional Shell Filling", fontsize=15)

    # 加上浅色网格线以便观察坐标
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.savefig(f"PhaseDiagram_{filename_base}.png", dpi=300, bbox_inches='tight')
    print(f"Plot saved as PhaseDiagram_{filename_base}.png")

if __name__ == "__main__":
    main()
