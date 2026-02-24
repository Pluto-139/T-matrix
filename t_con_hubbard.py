import numpy as np
import time
import multiprocessing as mp
from multiprocessing import Pool
import os
from tqdm import tqdm
from numba import njit, prange

# --- 物理参数 ---
t_hop = 1.0
U_int = 4.0 * t_hop
T_smear = 0.05 * t_hop  # 有限温度展宽

# --- 积分节点设置 ---
# 使用高斯-勒让德点数。对于平滑函数，N_quad=40~60 已经非常接近热力学极限
N_quad = 48 
nodes, weights = np.polynomial.legendre.leggauss(N_quad)
# 将 [-1, 1] 映射到 [-pi, pi]
nodes_pi = nodes * np.pi
weights_pi = weights * np.pi / (2 * np.pi) # 归一化因子包含在测度中

# --- Numba 加速的高速物理函数 ---

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
    """
    严格按照公式计算 chi_pp(q, omega)
    积分测度为 d^2p / (2pi)^2
    """
    chi = 0.0
    for i in range(N_quad):
        px = nodes[i]
        for j in range(N_quad):
            py = nodes[j]
            
            eps_p = dispersion(px, py, tp)
            # p' = q - p
            eps_minus_p_q = dispersion(qx - px, qy - py, tp)
            
            f_up = fermi_dirac(eps_p, mu)
            f_down = fermi_dirac(eps_minus_p_q, mu)
            
            numerator = (1.0 - f_up) * (1.0 - f_down)
            denom = eps_p + eps_minus_p_q - omega
            
            # 引入微小虚部或截断避免发散
            if abs(denom) < 1e-10:
                denom = 1e-10
            
            # weights[i]*weights[j] 对应 (dp_x/2pi)*(dp_y/2pi)
            chi += (weights[i] * weights[j]) * (numerator / denom)
    return chi

@njit(fastmath=True, parallel=False)
def compute_total_energy_pm(mu, tp, U, nodes, weights):
    """
    计算 PM 态的总能量 E = E_kin + E_int
    包含 4 维外层积分 (k, k') 和 2 维内层积分 (chi_pp)
    """
    E_kin = 0.0
    # 1. 动能项积分
    for i in range(N_quad):
        for j in range(N_quad):
            kx, ky = nodes[i], nodes[j]
            eps_k = dispersion(kx, ky, tp)
            f_k = fermi_dirac(eps_k, mu)
            E_kin += 2.0 * (weights[i] * weights[j]) * f_k * eps_k

    # 2. 相互作用项积分 (4维嵌套)
    E_int_sum = 0.0
    for i1 in range(N_quad):
        for j1 in range(N_quad):
            kx, ky = nodes[i1], nodes[j1]
            f_k = fermi_dirac(dispersion(kx, ky, tp), mu)
            if f_k < 1e-7: continue # 优化：远离费米海的态贡献极小
            
            for i2 in range(N_quad):
                for j2 in range(N_quad):
                    kpx, kpy = nodes[i2], nodes[j2]
                    f_kp = fermi_dirac(dispersion(kpx, kpy, tp), mu)
                    if f_kp < 1e-7: continue
                    
                    # 按照公式：q = k + k', omega = eps_k + eps_kp
                    qx, qy = kx + kpx, ky + kpy
                    omega = dispersion(kx, ky, tp) + dispersion(kpx, kpy, tp)
                    
                    chi = compute_chi_pp_core(qx, qy, omega, mu, tp, nodes, weights)
                    
                    term = (f_k * f_kp) / (1.0 + U * chi)
                    E_int_sum += (weights[i1]*weights[j1]*weights[i2]*weights[j2]) * term
                    
    return E_kin + U * E_int_sum

# --- 化学势求解 (保持连续积分) ---

def get_mu(rho, tp, spin_deg):
    from scipy.optimize import brentq
    
    @njit
    def calc_n(mu_guess):
        n = 0.0
        for i in range(N_quad):
            for j in range(N_quad):
                eps = dispersion(nodes_pi[i], nodes_pi[j], tp)
                n += (weights_pi[i] * weights_pi[j]) * fermi_dirac(eps, mu_guess)
        return n * spin_deg

    return brentq(lambda m: calc_n(m) - rho, -8.0, 8.0)

# --- 主计算任务 ---

def compute_phase_point(args):
    rho, tp = args
    
    # 1. FM 能量 (仅动能，无 U 项贡献在 Nagaoka 限制下)
    mu_fm = get_mu(rho, tp, 1)
    E_fm = 0.0
    for i in range(N_quad):
        for j in range(N_quad):
            eps = dispersion(nodes_pi[i], nodes_pi[j], tp)
            E_fm += (weights_pi[i] * weights_pi[j]) * fermi_dirac(eps, mu_fm) * eps

    # 2. PM 能量 (包含 U 项的 6 维积分)
    mu_pm = get_mu(rho, tp, 2)
    E_pm = compute_total_energy_pm(mu_pm, tp, U_int, nodes_pi, weights_pi)
    
    is_fm = 1 if E_fm < E_pm else 0
    return rho, tp, E_fm, E_pm, is_fm

if __name__ == '__main__':
    # 参数空间
    rho_vals = np.linspace(0.1, 0.7, 10) # 建议先从小规模测试
    tp_vals = np.linspace(-0.6, -0.4, 10)
    tasks = [(r, t) for t in tp_vals for r in rho_vals]
    
    print(f"Starting Optimized Thermodynamic Limit Calculation...")
    print(f"Integration Method: Gauss-Legendre Quadrature (N={N_quad})")
    
    start = time.time()
    
    # 在集群上利用多进程分发任务
    try:
        cpus = int(os.environ['SLURM_CPUS_PER_TASK'])
    except:
        cpus = mp.cpu_count()

    with Pool(cpus) as p:
        results = list(tqdm(p.imap(compute_phase_point, tasks), total=len(tasks)))
    
    # 保存结果（逻辑同前）
    results_array = np.array(results)
    np.savetxt("optimized_thermo_data.dat", results_array, header="rho tp E_fm E_pm is_fm")
    print(f"Done in {(time.time()-start)/60:.2f} mins.")
