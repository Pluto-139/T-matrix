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

# 将这段代码追加到你上面的代码之后

def run_convergence_test(target_density=0.01, tp=0.2, grid_sizes=None):
    if grid_sizes is None:
        # 从粗网格到极细网格
        grid_sizes = [100, 200, 400, 800, 1200, 1600]
        
    print(f"--- 开始收敛性测试 ---")
    print(f"目标密度: {target_density}, tp: {tp}, 温度平滑: {T_SMEARING}")
    print(f"{'N_grid':>6} | {'mu':>10} | {'Energy':>12} | {'DOS Error':>10} | {'Time (s)':>8}")
    print("-" * 60)

    mu_list = []
    energy_list = []
    dos_error_list = []
    times = []

    # 解析极限 (仅在低密度、且 tp 使得带底仍在 (0,0) 点时准确)
    min_eps = -4 * t - 4 * tp
    theoretical_dos = 1.0 / (4 * np.pi * (t + 2 * tp))

    for n in grid_sizes:
        start_time = time.time()
        
        # 1. 计算化学势
        mu = compute_chemical_potential(target_density, tp, n_grid=n)
        
        # 2. 计算能量
        energy = compute_free_energy(tp, mu, n_grid=n)
        
        # 3. 对比解析态密度
        calculated_dos = target_density / (mu - min_eps)
        dos_error = abs(theoretical_dos - calculated_dos) / theoretical_dos
        
        end_time = time.time()
        
        mu_list.append(mu)
        energy_list.append(energy)
        dos_error_list.append(dos_error)
        times.append(end_time - start_time)
        
        print(f"{n:6d} | {mu:10.6f} | {energy:12.8f} | {dos_error:9.2%} | {times[-1]:8.2f}")

    # ================= 绘图部分 =================
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 图1: 化学势的收敛
    axes[0].plot(grid_sizes, mu_list, marker='o', color='b')
    axes[0].set_title("Convergence of $\mu$")
    axes[0].set_xlabel("$N_{grid}$")
    axes[0].set_ylabel("Chemical Potential $\mu$")
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # 图2: 能量的收敛
    axes[1].plot(grid_sizes, energy_list, marker='s', color='r')
    axes[1].set_title("Convergence of Free Energy")
    axes[1].set_xlabel("$N_{grid}$")
    axes[1].set_ylabel("Free Energy")
    axes[1].grid(True, linestyle='--', alpha=0.6)

    # 图3: DOS误差的收敛 (对数坐标)
    axes[2].plot(grid_sizes, dos_error_list, marker='^', color='g')
    axes[2].set_yscale('log')
    axes[2].set_title("Error vs Analytical DOS")
    axes[2].set_xlabel("$N_{grid}$")
    axes[2].set_ylabel("Relative Error (Log Scale)")
    axes[2].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('convergence_test_results.png', dpi=300)
    print(f"\n测试完成！收敛趋势图已保存为 'convergence_test_results.png'")

# 运行测试 (假设目标密度非常低，比如 0.005)
if __name__ == '__main__':
    run_convergence_test(target_density=0.005, tp=1)
