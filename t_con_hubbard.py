import numpy as np
import time
import multiprocessing as mp
from multiprocessing import Pool
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import matplotlib.patches as mpatches
from scipy.optimize import brentq
import os
from tqdm import tqdm
from numba import njit

# --- 物理参数设置 ---
t_hop = 1.0
U_int = 4.0 * t_hop
T_smear = 0.05 * t_hop  # 有限温度展宽 (Smearing)

# --- 积分节点设置 (Gauss-Legendre) ---
# N_quad 决定了热力学极限的逼近精度，集群运行建议 48 或更高
N_quad = 48 
nodes, weights = np.polynomial.legendre.leggauss(N_quad)
nodes_pi = nodes * np.pi
weights_pi = weights * np.pi / (2 * np.pi) 

# --- Numba 高速计算核心 (公式保持严格不变) ---

@njit(fastmath=True)
def dispersion(kx, ky, tp):
    return -2.0 * t_hop * (np.cos(kx) + np.cos(ky)) - 4.0 * tp * np.cos(kx) * np.cos(ky)

@njit(fastmath=True)
def fermi_dirac(eps, mu):
    x = (eps - mu) / T_smear
    if x > 50: return 0.0
    if x < -50: return 1.0
    return 1.0 / (np.exp(x) + 1.0)

@njit(fastmath=True)
def compute_chi_pp_core(qx, qy, omega, mu, tp, nodes, weights):
    """严格按照 chi_pp 积分公式计算"""
    chi = 0.0
    for i in range(N_quad):
        px = nodes[i]
        for j in range(N_quad):
            py = nodes[j]
            eps_p = dispersion(px, py, tp)
            eps_minus_p_q = dispersion(qx - px, qy - py, tp)
            
            f_up = fermi_dirac(eps_p, mu)
            f_down = fermi_dirac(eps_minus_p_q, mu)
            
            numerator = (1.0 - f_up) * (1.0 - f_down)
            denom = eps_p + eps_minus_p_q - omega
            if abs(denom) < 1e-10: denom = 1e-10
            chi += (weights[i] * weights[j]) * (numerator / denom)
    return chi

@njit(fastmath=True)
def compute_total_energy_pm(mu, tp, U, nodes, weights):
    """PM 总能量积分: E_kin + E_int"""
    E_kin = 0.0
    for i in range(N_quad):
        for j in range(N_quad):
            kx, ky = nodes[i], nodes[j]
            eps_k = dispersion(kx, ky, tp)
            E_kin += 2.0 * (weights[i] * weights[j]) * fermi_dirac(eps_k, mu) * eps_k

    E_int_sum = 0.0
    for i1 in range(N_quad):
        for j1 in range(N_quad):
            kx, ky = nodes[i1], nodes[j1]
            f_k = fermi_dirac(dispersion(kx, ky, tp), mu)
            if f_k < 1e-7: continue
            for i2 in range(N_quad):
                for j2 in range(N_quad):
                    kpx, kpy = nodes[i2], nodes[j2]
                    f_kp = fermi_dirac(dispersion(kpx, kpy, tp), mu)
                    if f_kp < 1e-7: continue
                    
                    qx, qy = kx + kpx, ky + kpy
                    omega = dispersion(kx, ky, tp) + dispersion(kpx, kpy, tp)
                    chi = compute_chi_pp_core(qx, qy, omega, mu, tp, nodes, weights)
                    E_int_sum += (weights[i1]*weights[j1]*weights[i2]*weights[j2]) * (f_k * f_kp) / (1.0 + U * chi)
                    
    return E_kin + U * E_int_sum

# --- 化学势求解 ---
def get_mu(rho, tp, spin_deg):
    @njit
    def calc_n(mu_guess):
        n = 0.0
        for i in range(N_quad):
            for j in range(N_quad):
                eps = dispersion(nodes_pi[i], nodes_pi[j], tp)
                n += (weights_pi[i] * weights_pi[j]) * fermi_dirac(eps, mu_guess)
        return n * spin_deg
    return brentq(lambda m: calc_n(m) - rho, -8.0, 8.0)

def compute_phase_point(args):
    rho, tp = args
    try:
        mu_fm = get_mu(rho, tp, 1)
        E_fm = 0.0
        for i in range(N_quad):
            for j in range(N_quad):
                eps = dispersion(nodes_pi[i], nodes_pi[j], tp)
                E_fm += (weights_pi[i] * weights_pi[j]) * fermi_dirac(eps, mu_fm) * eps

        mu_pm = get_mu(rho, tp, 2)
        E_pm = compute_total_energy_pm(mu_pm, tp, U_int, nodes_pi, weights_pi)
        is_fm = 1 if E_fm < E_pm else 0
        return rho, tp, E_fm, E_pm, is_fm
    except:
        return rho, tp, 0.0, 0.0, 0

# --- 画图功能补全 ---
def plot_phase_diagram(results_array, rho_range, tp_range, filename_prefix):
    rho = results_array[:, 0]
    t_prime = results_array[:, 1]
    is_fm = results_array[:, 4]

    # 创建高密度网格用于插值
    rho_grid, tp_grid = np.meshgrid(
        np.linspace(rho.min(), rho.max(), 200),
        np.linspace(t_prime.min(), t_prime.max(), 200)
    )

    # 线性插值
    points = np.column_stack((rho, t_prime))
    phase_grid = griddata(points, is_fm, (rho_grid, tp_grid), method='linear')

    # 高斯平滑处理边界锯齿
    phase_grid_smooth = gaussian_filter(phase_grid, sigma=1.0)
    phase_grid_binary = (phase_grid_smooth > 0.5).astype(float)

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = ListedColormap(['white', 'lightgray'])

    # 绘制色块
    ax.pcolormesh(rho_grid, tp_grid, phase_grid_binary, 
                  cmap=cmap, shading='auto', alpha=0.8)

    # 绘制相界黑线
    ax.contour(rho_grid, tp_grid, phase_grid_smooth, 
               levels=[0.5], colors='black', linewidths=2, alpha=0.8)

    # 设置坐标轴与标签
    ax.set_xlabel(r'Electron density $n$', fontsize=14)
    ax.set_ylabel(r'$t^{\prime}/t$', fontsize=14)
    ax.set_xlim(rho_range)
    ax.set_ylim(tp_range)

    # 添加图例
    fm_patch = mpatches.Patch(color='lightgray', label='Ferromagnetic (FM)')
    pm_patch = mpatches.Patch(color='white', label='Paramagnetic (PM)')
    ax.legend(handles=[fm_patch, pm_patch], loc='upper right', fontsize=12)

    # 标题标注热力学极限与展宽
    ax.set_title(rf'Thermodynamic Limit Phase Diagram ($U={U_int}t$, $T={T_smear}t$)', fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{filename_prefix}.png", dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Figures saved: {filename_prefix}.png")

# --- 主程序入口 ---
if __name__ == '__main__':
    # 定义计算范围
    rho_min, rho_max = 0.1, 0.7
    tp_min, tp_max = -0.6, -0.4
    n_rho, n_tp = 30, 30  # 网格点密度
    
    rho_vals = np.linspace(rho_min, rho_max, n_rho)
    tp_vals = np.linspace(tp_min, tp_max, n_tp)
    tasks = [(r, tp) for tp in tp_vals for r in rho_vals]

    print(f"Starting computation on {len(tasks)} points (N_quad={N_quad})...")
    start_time = time.time()

    # SLURM 环境适配
    try:
        cpus = int(os.environ['SLURM_CPUS_PER_TASK'])
    except:
        cpus = mp.cpu_count()

    with Pool(cpus) as pool:
        results = list(tqdm(pool.imap(compute_phase_point, tasks), total=len(tasks)))

    results_array = np.array(results)
    
    # 存盘
    prefix = f"thermo_U4_T{T_smear}_n{n_rho}x{n_tp}"
    np.savetxt(f"{prefix}_data.dat", results_array, header="rho tp E_fm E_pm is_fm")
    
    # 调用补全的绘图功能
    plot_phase_diagram(results_array, (rho_min, rho_max), (tp_min, tp_max), prefix)
    
    print(f"Total time: {(time.time()-start_time)/60:.2f} mins.")
