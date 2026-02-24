import numpy as np
from multiprocessing import Pool
import time

# ==========================================
# 物理参数设置 (对齐文献 Fig 1)
# ==========================================
t = 1.0           # 最近邻跃迁能量
U = 4.0 * t       # 库仑排斥能 
L = 32            # 晶格尺寸 [cite: 86]
Omega = L * L     # 总格点数

def compute_phase_point(args):
    """
    计算相图上单个点 (rho, t_prime) 的基态能量，并返回是否为铁磁态
    """
    rho, t_prime = args
    N_total = int(rho * Omega)
    
    # 保证电子数为偶数，以便在顺磁态中均分自旋
    if N_total % 2 != 0:
        N_total -= 1
        
    # 极低密度或满带的边界情况直接忽略
    if N_total == 0 or N_total >= Omega:
        return rho, t_prime, 0.0, 0.0, 0 

    # 1. 生成离散动量网格和能量色散
    # 采用整数索引进行计算，方便后续利用周期性边界条件进行动量守恒的取模运算
    kx_idx, ky_idx = np.meshgrid(np.arange(L), np.arange(L))
    kx_idx = kx_idx.flatten()
    ky_idx = ky_idx.flatten()
    
    # 映射到 [-pi, pi) 计算实际能量 [cite: 28]
    kx_val = kx_idx * 2 * np.pi / L - np.pi
    ky_val = ky_idx * 2 * np.pi / L - np.pi
    
    # 公式 (2): 紧束缚模型色散 [cite: 28]
    eps = -2 * t * (np.cos(kx_val) + np.cos(ky_val)) + 4 * t_prime * np.cos(kx_val) * np.cos(ky_val)
    
    # 对能量排序，确定费米海
    sorted_order = np.argsort(eps)
    eps_sorted = eps[sorted_order]
    
    # ==========================================
    # 计算完全极化态 (Nagaoka 铁磁态) 的能量 E_FM
    # ==========================================
    # 公式(3)的 U 项在完全极化时为0，因此仅剩动能
    E_fm = np.sum(eps_sorted[:N_total]) / Omega
    
    # ==========================================
    # 计算顺磁态的能量 E_PM
    # ==========================================
    N_half = N_total // 2
    occ_pm = sorted_order[:N_half]      # 占据态索引
    unocc_pm = sorted_order[N_half:]    # 未占据态索引
    
    # 化学势 (最高占据态)
    mu = eps[occ_pm[-1]]
    eps_tilde = eps - mu
    
    # 顺磁态动能部分
    E_pm_kin = 2 * np.sum(eps[occ_pm]) / Omega
    
    # 预处理未占据态的动量索引，用于快速求和
    px_idx = unocc_pm % L
    py_idx = unocc_pm // L
    eps_tilde_p = eps_tilde[unocc_pm]
    
    # 创建一个 L x L 的布尔掩码，标记未占据态
    is_unocc = np.zeros((L, L), dtype=bool)
    is_unocc[py_idx, px_idx] = True
    
    interaction_sum = 0.0
    
    # 提取占据态的 x, y 动量索引
    occ_kx = kx_idx[occ_pm]
    occ_ky = ky_idx[occ_pm]
    eps_tilde_occ = eps_tilde[occ_pm]
    
    # 计算公式(3)和(60) [cite: 56, 57, 60, 61]
    for i in range(N_half):
        kx_i = occ_kx[i]
        ky_i = occ_ky[i]
        eps_k = eps_tilde_occ[i]
        
        for j in range(N_half):
            kx_j = occ_kx[j]
            ky_j = occ_ky[j]
            eps_kp = eps_tilde_occ[j]
            
            # 计算能量和 \omega [cite: 59, 60]
            omega = eps_k + eps_kp
            
            # 总动量 q = k + k'，利用周期性边界取模 
            qx = (kx_i + kx_j) % L
            qy = (ky_i + ky_j) % L
            
            # 动量守恒: 计算剩余动量 (-p + q)
            minus_p_q_x = (qx - px_idx) % L
            minus_p_q_y = (qy - py_idx) % L
            
            # 过滤掉 (-p + q) 落在占据态的项，对应公式(60)分子中的 (1-f) [cite: 60]
            valid_mask = is_unocc[minus_p_q_y, minus_p_q_x]
            
            if not np.any(valid_mask):
                # 如果没有满足条件的 p，chi_pp = 0
                interaction_sum += 1.0
                continue
            
            # 提取有效的未占据态能量
            p_valid_idx = np.where(valid_mask)[0]
            eps_p_valid = eps_tilde_p[p_valid_idx]
            
            # 将 (-p + q) 的二维索引转回一维，以提取能量
            minus_p_q_flat = minus_p_q_y[p_valid_idx] * L + minus_p_q_x[p_valid_idx]
            eps_minus_p_q_valid = eps_tilde[minus_p_q_flat]
            
            # 计算公式(60)分母: \tilde{\epsilon}_p + \tilde{\epsilon}_{-p+q} - \omega
            denom = eps_p_valid + eps_minus_p_q_valid - omega
            # 施加极小的截断值防止零点奇异性造成的数值崩溃
            denom[denom < 1e-12] = 1e-12 
            
            # 公式(60): 累加得到 \chi_pp
            chi_pp = np.sum(1.0 / denom) / Omega
            
            # 累加公式(3)中的排斥势分母项 [cite: 56]
            interaction_sum += 1.0 / (1.0 + U * chi_pp)

    # 乘以常数系数，转化为每个格点的能量
    E_pm_int = (U / (Omega * Omega)) * interaction_sum
    E_pm = E_pm_kin + E_pm_int
    
    # 比较能量：如果铁磁态能量更低，标为 1，否则为 0
    is_fm = 1 if E_fm < E_pm else 0
    
    return rho, t_prime, E_fm, E_pm, is_fm
    
def plot_phase_diagram(data, rho_range, tp_range, filename_prefix):
    """
    绘制相图
    """
    rho = data[:, 0]
    t_prime = data[:, 1]
    is_fm = data[:, 4]
    
    # 创建网格用于插值和绘图
    rho_grid, tp_grid = np.meshgrid(
        np.linspace(rho.min(), rho.max(), 200),
        np.linspace(t_prime.min(), t_prime.max(), 200)
    )
    
    # 将离散数据插值到网格上
    points = np.column_stack((rho, t_prime))
    phase_grid = griddata(points, is_fm, (rho_grid, tp_grid), method='linear')
    
    # 平滑处理以获得清晰的边界
    phase_grid_smooth = gaussian_filter(phase_grid, sigma=1.0)
    phase_grid_binary = (phase_grid_smooth > 0.5).astype(float)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 设置颜色映射：顺磁态为白色，铁磁态为灰色
    cmap = ListedColormap(['white', 'lightgray'])
    
    # 绘制相图
    im = ax.pcolormesh(rho_grid, tp_grid, phase_grid_binary, 
                       cmap=cmap, shading='auto', alpha=0.8)
    
    # 使用contour找到边界线
    contours = ax.contour(rho_grid, tp_grid, phase_grid_smooth, 
                          levels=[0.5], colors='black', linewidths=2, alpha=0.8)
    
    # 设置坐标轴标签
    ax.set_xlabel(r'Electron density $n$', fontsize=14)
    ax.set_ylabel(r'$t^{\prime}/t$', fontsize=14)
    
    # 设置坐标轴范围
    ax.set_xlim(rho.min(), rho.max())
    ax.set_ylim(t_prime.min(), t_prime.max())
    
    # 添加图例
    fm_patch = mpatches.Patch(color='lightgray', label='Ferromagnetic (FM)')
    pm_patch = mpatches.Patch(color='white', label='Paramagnetic (PM)')
    ax.legend(handles=[fm_patch, pm_patch], loc='upper right', fontsize=12)
    
    # 添加标题
    ax.set_title(f'Phase diagram for U = 4t\n'
                 f'$t^{{\prime}}/t$: [{tp_range[0]:.2f}, {tp_range[1]:.2f}], '
                 f'n: [{rho_range[0]:.2f}, {rho_range[1]:.2f}]', 
                 fontsize=14)
    
    # 添加网格线
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plot_filename = f"{filename_prefix}_phase.png"
    pdf_filename = f"{filename_prefix}_phase.pdf"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_filename, bbox_inches='tight')
    plt.close()
    
    print(f"Phase diagram saved as '{plot_filename}' and '{pdf_filename}'")
    
    return plot_filename

if __name__ == '__main__':
    rho_min, rho_max = 0.01, 0.75
    tp_min, tp_max = 0.40, 0.50
    n_rho = 40
    n_tp = 40
    
    # 生成所有需要计算的参数对
    tasks = [(r, tp) for tp in t_prime_vals for r in rho_vals]
    
    print(f"Total points to compute: {len(tasks)}")
    print(f"Parameter range: n ∈ [{rho_min:.2f}, {rho_max:.2f}], "
          f"t'/t ∈ [{tp_min:.2f}, {tp_max:.2f}]")
    start_time = time.time()
    
    # 定义进程池，使用集群节点上几乎所有的核心
    num_cores = 56  
    
    with Pool(processes=num_cores) as pool:
        results = pool.map(compute_phase_point, tasks)
    
    end_time = time.time()
    computation_time = (end_time - start_time) / 60
    print(f"Computation completed in {computation_time:.2f} minutes.")
    
    # 将结果转换为numpy数组
    results_array = np.array(results)
    
    # 生成带参数区间的文件名
    filename_prefix = f"phase_U4_tp{tp_min:.2f}-{tp_max:.2f}_n{rho_min:.2f}-{rho_max:.2f}"
    data_filename = f"{filename_prefix}_data.dat"
    
    # 保存数据为文本文件
    np.savetxt(data_filename, results_array, fmt="%.6f", 
               header="rho t_prime E_fm E_pm is_fm")
    print(f"Data saved to {data_filename}")
    
    # 绘制相图
    plot_filename = plot_phase_diagram(results_array, 
                                      (rho_min, rho_max), 
                                      (tp_min, tp_max), 
                                      filename_prefix)
