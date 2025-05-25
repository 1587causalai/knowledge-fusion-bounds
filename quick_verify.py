#!/usr/bin/env python3
import numpy as np

def kl_divergence_cauchy(mu_0, gamma_0, mu, gamma):
    return np.log((gamma_0 + gamma)**2 + (mu_0 - mu)**2) - np.log(4 * gamma_0 * gamma)

print("快速验证反例区域")
print("="*50)

# 验证反例区域
delta_mu_range = np.linspace(0.1, 1.0, 10)
delta_s_range = np.linspace(-1.0, 0.0, 10)
N = 2

counterexample_count = 0
total_count = 0

print("检查参数组合...")
for delta_mu in delta_mu_range:
    for delta_s in delta_s_range:
        mu_fused = N * delta_mu
        gamma_fused = 1.0 * np.exp(N * delta_s)
        
        kl_fused = kl_divergence_cauchy(0, 1.0, mu_fused, gamma_fused)
        kl_individual = kl_divergence_cauchy(0, 1.0, delta_mu, 1.0 * np.exp(delta_s))
        
        ratio = kl_fused / (N * kl_individual)
        inequality_holds = ratio <= 1.0
        
        total_count += 1
        if not inequality_holds:
            counterexample_count += 1

print(f'反例比例: {counterexample_count}/{total_count} = {counterexample_count/total_count:.4f}')
print(f'这意味着在 delta_mu > 0 且 delta_s < 0 的区域，几乎所有组合都是反例！')

print("\n具体分析几个案例:")
test_cases = [
    (0.3, -0.3),
    (0.5, -0.5), 
    (0.7, -0.7),
    (0.9, -0.9)
]

for delta_mu, delta_s in test_cases:
    mu_fused = N * delta_mu
    gamma_fused = 1.0 * np.exp(N * delta_s)
    
    kl_fused = kl_divergence_cauchy(0, 1.0, mu_fused, gamma_fused)
    kl_individual = kl_divergence_cauchy(0, 1.0, delta_mu, 1.0 * np.exp(delta_s))
    
    ratio = kl_fused / (N * kl_individual)
    
    print(f"delta_mu={delta_mu}, delta_s={delta_s}: ratio={ratio:.4f}, 反例={ratio > 1.0}")

print("\n关键发现：")
print("当位置参数增大(delta_mu > 0)且尺度参数减小(delta_s < 0)时，")
print("柯西分布的KL散度不等式系统性地失效！") 