import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches  # 引入用于画图例的库
import multiprocessing as mp
import time
import os
from scipy.integrate import simpson
from scipy.optimize import brentq
from tqdm import tqdm

t = 1.0
U = 4.0 * t
T_SMEARING = 0.01 * t  
def
    return -2 * t * (np.cos(kx) + np.cos(ky)) - 4 * tp * np.cos(kx) * np.cos(ky)

def fermi_smooth(energies, mu):

    val = (energies - mu) / T_SMEARING
    val = np.clip(val, -100, 100)
    return 1.0 / (np.exp(val) + 1.0)

def integrate_2d_simpson(f_val, x_grid, y_grid):
    
    res_y = simpson(f_val, x=y_grid, axis=1)
    
    res_x = simpson(res_y, x=x_grid, axis=0)
    
    return res_x

def compute_chemical_potential(target_density, tp, n_grid=2000):
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

def compute_free_energy(tp, mu, n_grid=2000):
    kx = np.linspace(-np.pi, np.pi, n_grid)
    ky = np.linspace(-np.pi, np.pi, n_grid)
    KX, KY = np.meshgrid(kx, ky)
    
    eps = epsilon_k_continuous(KX, KY, tp)
    f = fermi_smooth(eps, mu)
    
    energy = integrate_2d_simpson(eps * f, kx, ky) / (2 * np.pi) ** 2
    return energy

def solve_phase_point_deltaE(args):
    tp_over_t, density, N_grid = args
    tp = tp_over_t * t

    if density <= 1e-5:
        return 0.0

    try:
        # --- 1. 计算化学势与无相互作用的动能 ---
        mu_para = compute_chemical_potential(density / 2.0, tp)
        mu_ferro = compute_chemical_potential(density, tp)

        E_kin_para = 2.0 * compute_free_energy(tp, mu_para)
        E_total_ferro = compute_free_energy(tp, mu_ferro) # 铁磁态因同向自旋排斥相互作用为0

        # --- 2. 准备动量网格 ---
        # Simpson 积分更喜欢奇数个点 (偶数个区间)
        kx_1d = np.linspace(-np.pi, np.pi, N_grid)
        ky_1d = np.linspace(-np.pi, np.pi, N_grid)
        KX, KY = np.meshgrid(kx_1d, ky_1d, indexing='ij')

        eps = epsilon_k_continuous(KX, KY, tp)
        eps_tilde = eps - mu_para
        f_p = fermi_smooth(eps_tilde, 0.0)

        # --- 3. 准备虚频网格 (利用 x = atan(w/Gamma) 将 [0, inf) 映射到 [0, pi/2) ) ---
        N_omega = 60
        Gamma = 4.0 * t
        x_grid = np.linspace(0, np.pi/2 - 1e-6, N_omega)
        omega = Gamma * np.tan(x_grid)
        d_omega_dx = Gamma / (np.cos(x_grid)**2)
        omega_3d = omega[np.newaxis, np.newaxis, :]  # 形状: (1, 1, N_omega)

        integrand_q = np.zeros((N_grid, N_grid))

        # --- 4. 计算 Equation (4) 中的双重积分 ---
        for i in range(N_grid):
            for j in range(N_grid):
                qx = kx_1d[i]
                qy = ky_1d[j]

                # cos(q-p) 是周期函数，直接减即可，无需取模
                eps_minus_p_q = epsilon_k_continuous(qx - KX, qy - KY, tp) - mu_para
                f_minus_p_q = fermi_smooth(eps_minus_p_q, 0.0)

                # 提取化简后的有效能量项和分子项
                E_p = eps_tilde + eps_minus_p_q
                N_p = 1.0 - f_p - f_minus_p_q

                E_p_3d = E_p[:, :, np.newaxis]
                N_p_3d = N_p[:, :, np.newaxis]

                # 计算复数分母的绝对值平方 (添加 1e-12 防止在 E=0 且 w=0 处的 0/0 警告)
                denom = E_p_3d**2 + omega_3d**2
                denom = np.where(denom < 1e-12, 1e-12, denom)

                # 计算实部 chi_1 和虚部 chi_2 的被积函数
                chi1_integrand = (E_p_3d * N_p_3d) / denom
                chi2_integrand = (omega_3d * N_p_3d) / denom

                # 并行对 p 积分得到对于每个 omega 的 chi_1 和 chi_2 (形状: N_omega)
                chi1 = integrate_2d_simpson(chi1_integrand, kx_1d, ky_1d) / (2*np.pi)**2
                chi2 = integrate_2d_simpson(chi2_integrand, kx_1d, ky_1d) / (2*np.pi)**2

                # Equation 4 的 ln 积分核
                log_term = np.log((1.0 + U * chi1)**2 + (U * chi2)**2)

                # 沿着 x (即映射后的虚频 omega) 积分
                integral_omega = simpson(log_term * d_omega_dx, x=x_grid) / (2*np.pi)
                integrand_q[i, j] = integral_omega

        # 沿着 q 积分
        integral_q = integrate_2d_simpson(integrand_q, kx_1d, ky_1d) / (2*np.pi)**2

        # 组合 Eq (4) 最终的基态能量
        E_total_para = E_kin_para - (U / 2.0) * (1.0 - density) + integral_q

        return E_total_para - E_total_ferro

    except Exception as e:
        print(f"Error at tp={tp_over_t}, n={density}: {e}")
        return 0.0

def main():
    tp_vals = np.linspace(-0.4, -0.7, 50)
    dens_vals = np.linspace(0.01, 0.7, 50)
    
    # 极度平滑的方程，只需要 37x37 就能达到极高的 Simpson 精度！
    N_grid =61

    tasks = [(tp, dens, N_grid) for tp in tp_vals for dens in dens_vals]

    try:
        n_cores = int(os.environ['SLURM_CPUS_PER_TASK'])
    except:
        n_cores = mp.cpu_count()

    print(f"=== Hubbard Phase Diagram (Eq. 4 Matsubara Integration) ===")
    print(f"Cores: {n_cores}")
    print(f"Grid: {len(tp_vals)}x{len(dens_vals)} = {len(tasks)} points")
    print(f"Integration Mesh: N_q={N_grid}, N_k={N_grid}, N_w=60")
    print(f"Physics: U={U / t}t, T_smear={T_SMEARING}t (Strictly No Artificial Broadening)")

    start_time = time.time()

    with mp.Pool(n_cores) as pool:
        results = list(tqdm(pool.imap(solve_phase_point_deltaE, tasks, chunksize=1), total=len(tasks)))

    elapsed = (time.time() - start_time) / 60
    print(f"Calculation finished in {elapsed:.2f} minutes.")

    delta_E_grid = np.array(results).reshape(len(tp_vals), len(dens_vals))

    filename_base = f"two_Eq4_U{U:.1f}_tp{tp_vals[0]:.1f}_{tp_vals[-1]:.1f}"
    np.savez(f"results_{filename_base}.npz", tp=tp_vals, dens=dens_vals, delta_E=delta_E_grid)

    # ================= 修改后的绘图部分 =================
    plt.figure(figsize=(10, 8))
    X, Y = np.meshgrid(dens_vals, tp_vals)

    # 使用边界划分：负值（顺磁）填白色，正值（铁磁）填浅灰色
    bounds = [-np.inf, 0, np.inf]
    plt.contourf(X, Y, delta_E_grid, levels=bounds, colors=['white', 'lightgray'])

    # 画出黑色的零等能线相界
    plt.contour(X, Y, delta_E_grid, levels=[0], colors='black', linewidths=2)

    # 绘制自定义图例
    fm_patch = mpatches.Patch(color='lightgray', label='Ferromagnetic (FM)')
    pm_patch = mpatches.Patch(color='white', label='Paramagnetic (PM)')
    plt.legend(handles=[fm_patch, pm_patch], loc='upper right', framealpha=0.9)

    plt.xlabel("Density $n$", fontsize=14)
    plt.ylabel("$t^{\\prime}/t$", fontsize=14)
    plt.title(f"Phase Diagram (Eq 4: Virtual Hole Excitation Included)\n$U={U/t}t$, $T={T_SMEARING/t}t$", fontsize=15)
    
    # 增加网格线更易读
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.savefig(f"PhaseDiagram_{filename_base}.png", dpi=300, bbox_inches='tight')
    print("Done.")

if __name__ == "__main__":
    main()
