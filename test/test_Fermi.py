# step1_mu_convergence.py
import numpy as np
from hubbard_converged import compute_mu, get_adaptive_params, t

tp_val = -0.50   # 选一个代表性点
density = 0.03

nk_list = [100, 200, 500, 1000, 2000]

print(f"tp/t={tp_val}, n={density}")
print(f"{'n_grid':>8} {'eta/t':>8} {'T_smear/t':>10} {'mu_para/t':>12} {'mu_ferro/t':>12} {'变化(para)':>12}")
print("-" * 70)

prev_mu_para = None
for nk in nk_list:
    eta, T_smear = get_adaptive_params(nk)
    mu_para  = compute_mu(density / 2.0, tp_val * t, nk, T_smear)
    mu_ferro = compute_mu(density,        tp_val * t, nk, T_smear)
    change = abs(mu_para - prev_mu_para) if prev_mu_para is not None else float('nan')
    print(f"{nk:8d} {eta/t:8.4f} {T_smear/t:10.4f} {mu_para/t:12.8f} {mu_ferro/t:12.8f} {change:12.2e}")
    prev_mu_para = mu_para
