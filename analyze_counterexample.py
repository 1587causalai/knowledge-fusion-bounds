#!/usr/bin/env python3
import numpy as np

def kl_divergence_cauchy(mu_0, gamma_0, mu, gamma):
    return np.log((gamma_0 + gamma)**2 + (mu_0 - mu)**2) - np.log(4 * gamma_0 * gamma)

print("分析反例：为什么 delta_mu=0.5, delta_s=-0.5 会导致不等式不成立")
print("="*60)

# 参数设置
N = 2
delta_mu = 0.5
delta_s = -0.5
gamma_0 = 1.0

# 同向扰动
delta_mu_k = np.ones(N) * delta_mu  # [0.5, 0.5]
delta_s_k = np.ones(N) * delta_s    # [-0.5, -0.5]

mu_fused = np.sum(delta_mu_k)  # 1.0
gamma_fused = gamma_0 * np.prod(np.exp(delta_s_k))  # exp(-1) ≈ 0.368

print(f"基准分布 P_0: mu_0=0, gamma_0={gamma_0}")
print(f"扰动分布 P_k: mu_k={delta_mu}, gamma_k={gamma_0 * np.exp(delta_s)} ≈ {gamma_0 * np.exp(delta_s):.3f}")
print(f"融合分布 P_fused: mu_fused={mu_fused}, gamma_fused={gamma_fused:.3f}")
print()

# 分析关键点：尺度参数的变化
print("关键观察：")
print(f"1. 每个扰动分布的尺度参数 gamma_k = {gamma_0 * np.exp(delta_s):.3f} < gamma_0 = {gamma_0}")
print(f"2. 融合分布的尺度参数 gamma_fused = {gamma_fused:.3f} << gamma_0 = {gamma_0}")
print(f"3. 位置参数累积：mu_fused = {N} × {delta_mu} = {mu_fused}")
print()

# 计算各项KL散度
kl_individual = kl_divergence_cauchy(0, gamma_0, delta_mu, gamma_0 * np.exp(delta_s))
kl_fused = kl_divergence_cauchy(0, gamma_0, mu_fused, gamma_fused)

print("KL散度分析：")
print(f"单个扰动分布的KL散度: {kl_individual:.6f}")
print(f"融合分布的KL散度: {kl_fused:.6f}")
print(f"不等式右侧: N × KL_individual = {N} × {kl_individual:.6f} = {N * kl_individual:.6f}")
print(f"比率: {kl_fused / (N * kl_individual):.6f}")
print()

# 分析为什么会出现反例
print("反例分析：")
print("当尺度参数减小(delta_s < 0)且位置参数增大(delta_mu > 0)时：")
print("- 每个扰动分布相对于基准分布的KL散度较小（因为尺度变小）")
print("- 但融合分布的位置参数累积效应很强，且尺度参数变得很小")
print("- 小的尺度参数使得融合分布与基准分布差异很大")
print("- 这种非线性效应导致融合分布的KL散度超过了不等式右侧")
print()

# 验证这是否是系统性问题
print("验证其他相似参数组合：")
test_cases = [
    (0.4, -0.5),
    (0.6, -0.5),
    (0.5, -0.4),
    (0.5, -0.6),
    (0.3, -0.7),
    (0.7, -0.3)
]

for test_mu, test_s in test_cases:
    mu_f = N * test_mu
    gamma_f = gamma_0 * (np.exp(test_s))**N
    kl_f = kl_divergence_cauchy(0, gamma_0, mu_f, gamma_f)
    kl_i = kl_divergence_cauchy(0, gamma_0, test_mu, gamma_0 * np.exp(test_s))
    ratio = kl_f / (N * kl_i)
    holds = kl_f <= N * kl_i
    print(f"delta_mu={test_mu}, delta_s={test_s}: ratio={ratio:.4f}, holds={holds}") 