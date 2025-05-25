#!/usr/bin/env python3
import numpy as np

def kl_divergence_cauchy(mu_0, gamma_0, mu, gamma):
    return np.log((gamma_0 + gamma)**2 + (mu_0 - mu)**2) - np.log(4 * gamma_0 * gamma)

# 检查 N=2, delta_mu=0.5, delta_s=-0.5 的情况
N = 2
delta_mu = 0.5
delta_s = -0.5
gamma_0 = 1.0

print(f"Analyzing case: N={N}, delta_mu={delta_mu}, delta_s={delta_s}")

# 同向扰动
delta_mu_k = np.ones(N) * delta_mu  # [0.5, 0.5]
delta_s_k = np.ones(N) * delta_s    # [-0.5, -0.5]

mu_fused = np.sum(delta_mu_k)  # 1.0
gamma_fused = gamma_0 * np.prod(np.exp(delta_s_k))  # 1 * exp(-0.5)^2 = exp(-1)

print(f'Fused distribution: mu={mu_fused}, gamma={gamma_fused}')

# 计算融合分布的KL散度
kl_fused = kl_divergence_cauchy(0, gamma_0, mu_fused, gamma_fused)
print(f'KL_fused = {kl_fused}')

# 计算每个扰动分布的KL散度
kl_sum = 0
for k in range(N):
    gamma_k = gamma_0 * np.exp(delta_s_k[k])  # 1 * exp(-0.5)
    kl_k = kl_divergence_cauchy(0, gamma_0, delta_mu_k[k], gamma_k)
    print(f'Distribution {k+1}: mu={delta_mu_k[k]}, gamma={gamma_k}, KL={kl_k}')
    kl_sum += kl_k

print(f'Sum of individual KLs = {kl_sum}')
print(f'Right side = N * sum = {N * kl_sum}')
print(f'Ratio = {kl_fused / (N * kl_sum)}')
print(f'Inequality holds: {kl_fused <= N * kl_sum}')

print("\n" + "="*50)

# 检查 N=3 的情况
N = 3
print(f"Analyzing case: N={N}, delta_mu={delta_mu}, delta_s={delta_s}")

delta_mu_k = np.ones(N) * delta_mu  # [0.5, 0.5, 0.5]
delta_s_k = np.ones(N) * delta_s    # [-0.5, -0.5, -0.5]

mu_fused = np.sum(delta_mu_k)  # 1.5
gamma_fused = gamma_0 * np.prod(np.exp(delta_s_k))  # 1 * exp(-0.5)^3 = exp(-1.5)

print(f'Fused distribution: mu={mu_fused}, gamma={gamma_fused}')

# 计算融合分布的KL散度
kl_fused = kl_divergence_cauchy(0, gamma_0, mu_fused, gamma_fused)
print(f'KL_fused = {kl_fused}')

# 计算每个扰动分布的KL散度
kl_sum = 0
for k in range(N):
    gamma_k = gamma_0 * np.exp(delta_s_k[k])  # 1 * exp(-0.5)
    kl_k = kl_divergence_cauchy(0, gamma_0, delta_mu_k[k], gamma_k)
    print(f'Distribution {k+1}: mu={delta_mu_k[k]}, gamma={gamma_k}, KL={kl_k}')
    kl_sum += kl_k

print(f'Sum of individual KLs = {kl_sum}')
print(f'Right side = N * sum = {N * kl_sum}')
print(f'Ratio = {kl_fused / (N * kl_sum)}')
print(f'Inequality holds: {kl_fused <= N * kl_sum}') 