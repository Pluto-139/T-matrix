import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import multiprocessing as mp
import time
import os
from scipy.integrate import simpson
from tqdm import tqdm

# --- 物理参数 ---
t = 1.0
U = 4.0 * t

def solve_phase_point_eq4_discrete(args):
    tp_over_t, target_density, L = args
    tp = tp_over_t * t
    Omega = L * L

    # 1. 严格控制离散晶格的粒子数，保证闭壳层及自旋对称性
    N_tot = int(target_density * Omega)
    if N_tot % 2 != 0:
        N_tot -= 1
        
    if N_tot <= 0 or N_tot >= Omega:
        return 0.0
        
    actual_density = N_tot / Omega
    N_half = N_tot // 2

    # 2. 构建离散动量网格与色散关系
    kx = np.arange(L) * 2 * np.pi / L - np.pi
    ky = np.arange(L) * 2 * np.pi / L - np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    
    eps = -2 * t * (np.cos(KX) + np.cos(KY)) - 4 * tp * np.cos(KX) * np.cos(KY)
    eps_flat = eps.flatten()

    # 3. 绝对 T=0 的基态填充 (通过能级排序)
    sort_idx = np.argsort(eps_flat)
    
    # 顺磁态 (PM): 上下自旋各填 N_half
    f_PM = np.zeros(Omega)
    f_PM[sort_idx[:N_half]] = 1.0
    mu_PM = eps_flat[sort_idx[N_half-1]] # 占据的最高能级作为近似化学势
    eps_tilde = eps_flat - mu_PM

    # 铁磁态 (FM): 完全极化，单自旋填满 N_tot
    f_FM = np.zeros(Omega)
    f_FM[sort_idx[:N_tot]] = 1.0
    
    # 无相互作用的动能 (每格点平均)
    E_kin_PM = 2.0 * np.sum(f_PM * eps_flat) / Omega
    E_FM = np.sum(f_FM * eps_flat) / Omega

    # 4. 构建虚频 (Matsubara) 的连续积分映射网格
    N_omega = 60
    Gamma = 4.0 * t
    x_grid = np.linspace(0, np.pi/2 - 1e-6, N_omega)
    omega = Gamma * np.tan(x_grid)
    d_omega_dx = Gamma / (np.cos(x_grid)**2)
    omega_2d = omega[np.newaxis, :]  # 形状: (1, N_omega)

    # 预先生成用于动量守恒取模的整数索引
    px = np.arange(L)
    py = np.arange(L)
    PX, PY = np.meshgrid(px, py, indexing='ij')
    PX_flat = PX.flatten()
    PY_flat = PY.flatten()

    integral_q_sum = 0.0

    # 5. 执行公式 (4) 核心计算: 遍历所有外场动量 q
    for q_flat in range(Omega):
        qx = q_flat // L
        qy = q_flat % L

        # 动量守恒: 严格的整数索引取模，寻找 -p+q
        minus_p_q_x = (qx - PX_flat) % L
        minus_p_q_y = (qy - PY_flat) % L
        minus_p_q_idx = minus_p_q_x * L + minus_p_q_y

        f_minus_p_q = f_PM[minus_p_q_idx]
        eps_tilde_minus_p_q = eps_tilde[minus_p_q_idx]

        # 提取化简后的有效能量项 E_{p,q} 和分子项 N_{p,q}
        E_pq = eps_tilde + eps_tilde_minus_p_q
        N_pq = 1.0 - f_PM - f_minus_p_q

        # 升维以便利用 Numpy 广播机制处理所有 omega
        E_pq_2d = E_pq[:, np.newaxis]
        N_pq_2d = N_pq[:, np.newaxis]

        # 计算复数分母的绝对值平方 (添加 1e-12 避免物理极点完全重合的除零风险)
        denom = E_pq_2d**2 + omega_2d**2
        denom = np.where(denom < 1e-12, 1e-12, denom)

        # 沿着 p 求和，得到对于每一个 omega 的 chi1 和 chi2
        chi1_integrand = N_pq_2d * E_pq_2d / denom
        chi2_integrand = N_pq_2d * omega_2d / denom

        chi1 = np.sum(chi1_integrand, axis=0) / Omega
        chi2 = np.sum(chi2_integrand, axis=0) / Omega

        # 计算公式 (4) 积分核: ln[(1+U*chi1)^2 + (U*chi2)^2]
        log_term = np.log((1.0 + U * chi1)**2 + (U * chi2)**2)
        
        # 沿着虚频轴积分
        integral_omega = simpson(log_term * d_omega_dx, x=x_grid) / (2 * np.pi)
        
        integral_q_sum += integral_omega

    # 6. 整合结果
    # 注意公式(4)的偏置项 -U/2(1-n) 以及 q求和除以Omega归一化
    E_PM_int = - (U / 2.0) * (1.0 - actual_density) + (integral_q_sum / Omega)
    E_PM = E_kin_PM + E_PM_int

    return E_PM - E_FM

def main():
    # 扩大扫描范围，覆盖铁磁性更容易出现的较高掺杂和特定 t'/t 区间
    tp_vals = np.linspace(-0.4, -0.7, 50)
    dens_vals = np.linspace(0.01, 0.8, 50)
    
    L_lattice = 32  # 32x32 = 1024 个离散格点

    tasks = [(tp, dens, L_lattice) for tp in tp_vals for dens in dens_vals]

    try:
        n_cores = int(os.environ.get('SLURM_CPUS_PER_TASK', mp.cpu_count()))
    except:
        n_cores = mp.cpu_count()

    print(f"=== Hubbard Phase Diagram (Exact Discrete Eq. 4) ===")
    print(f"Cores: {n_cores}")
    print(f"Grid: {len(tp_vals)}x{len(dens_vals)} = {len(tasks)} points")
    print(f"Lattice: L={L_lattice} (Omega={L_lattice**2}), N_w=60")
    print(f"Physics: U={U / t}t, Strictly T=0 discrete shell filling")

    start_time = time.time()

    with mp.Pool(n_cores) as pool:
        # 使用 chunksize=1 使进度条在计算耗时任务时能平滑更新
        results = list(tqdm(pool.imap(solve_phase_point_eq4_discrete, tasks, chunksize=1), total=len(tasks)))

    elapsed = (time.time() - start_time) / 60
    print(f"Calculation finished in {elapsed:.2f} minutes.")

    delta_E_grid = np.array(results).reshape(len(tp_vals), len(dens_vals))

    filename_base = f"DiscreteEq4_L{L_lattice}_U{U:.1f}_tp{tp_vals[0]:.1f}_{tp_vals[-1]:.1f}"
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
    plt.title(f"Phase Diagram (Eq 4: Exact Discrete Grid $L={L_lattice}$)\nRed=Ferro, Blue=Para")

    plt.savefig(f"PhaseDiagram_{filename_base}.png", dpi=300)
    print("Done.")

if __name__ == "__main__":
    main()
