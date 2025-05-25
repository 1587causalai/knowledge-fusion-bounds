#!/usr/bin/env python3
import numpy as np

print("澄清尺度参数设置")
print("="*50)

# 基准分布的尺度参数
gamma_0 = 1.0
print(f"基准分布尺度参数: gamma_0 = {gamma_0}")

# 对数尺度参数扰动（可以是负数）
delta_s = -0.5
print(f"对数尺度参数扰动: delta_s = {delta_s}")

# 实际的尺度参数计算
gamma_k = gamma_0 * np.exp(delta_s)
print(f"扰动分布尺度参数: gamma_k = gamma_0 * exp(delta_s) = {gamma_0} * exp({delta_s}) = {gamma_k:.6f}")

print(f"\n关键点：")
print(f"1. delta_s 是对数尺度参数的扰动，可以是负数")
print(f"2. 实际尺度参数 gamma_k = gamma_0 * exp(delta_s) 始终 > 0")
print(f"3. 当 delta_s < 0 时，gamma_k < gamma_0（尺度变小）")
print(f"4. 当 delta_s > 0 时，gamma_k > gamma_0（尺度变大）")

print(f"\n验证不同 delta_s 值：")
delta_s_values = [-2, -1, -0.5, 0, 0.5, 1, 2]
for ds in delta_s_values:
    gamma = gamma_0 * np.exp(ds)
    print(f"delta_s = {ds:4.1f} => gamma = {gamma:8.6f} (始终 > 0)")

print(f"\n反例情况分析：")
N = 2
delta_s = -0.5
print(f"N = {N}, delta_s = {delta_s}")

# 每个扰动分布的尺度参数
gamma_individual = gamma_0 * np.exp(delta_s)
print(f"每个扰动分布: gamma_k = {gamma_individual:.6f} > 0")

# 融合分布的尺度参数
gamma_fused = gamma_0 * np.prod([np.exp(delta_s)] * N)
gamma_fused_alt = gamma_0 * np.exp(N * delta_s)
print(f"融合分布: gamma_fused = gamma_0 * exp(N * delta_s) = {gamma_0} * exp({N} * {delta_s}) = {gamma_fused:.6f} > 0")

print(f"\n所有尺度参数都是正数，符合柯西分布的要求！") 