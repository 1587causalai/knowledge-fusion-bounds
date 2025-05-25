#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from itertools import product

def kl_divergence_cauchy(mu_0, gamma_0, mu, gamma):
    """计算柯西分布之间的KL散度"""
    return np.log((gamma_0 + gamma)**2 + (mu_0 - mu)**2) - np.log(4 * gamma_0 * gamma)

def analyze_counterexample_region():
    """分析反例出现的参数区域"""
    print("=" * 80)
    print("深度分析：柯西分布KL散度不等式的反例现象")
    print("=" * 80)
    
    # 参数网格
    delta_mu_range = np.linspace(0.1, 1.0, 20)
    delta_s_range = np.linspace(-1.0, 0.0, 20)
    N_values = [2, 3, 4, 5]
    
    results = []
    
    for N in N_values:
        print(f"\n分析 N = {N} 的情况...")
        counterexample_count = 0
        total_count = 0
        
        for delta_mu, delta_s in product(delta_mu_range, delta_s_range):
            # 计算融合分布参数
            mu_fused = N * delta_mu
            gamma_fused = 1.0 * np.exp(N * delta_s)
            
            # 计算融合分布KL散度
            kl_fused = kl_divergence_cauchy(0, 1.0, mu_fused, gamma_fused)
            
            # 计算单个扰动分布KL散度
            kl_individual = kl_divergence_cauchy(0, 1.0, delta_mu, 1.0 * np.exp(delta_s))
            
            # 检查不等式
            right_side = N * kl_individual
            ratio = kl_fused / right_side if right_side > 0 else float('inf')
            inequality_holds = kl_fused <= right_side
            
            results.append({
                'N': N,
                'delta_mu': delta_mu,
                'delta_s': delta_s,
                'mu_fused': mu_fused,
                'gamma_fused': gamma_fused,
                'kl_fused': kl_fused,
                'kl_individual': kl_individual,
                'right_side': right_side,
                'ratio': ratio,
                'inequality_holds': inequality_holds
            })
            
            total_count += 1
            if not inequality_holds:
                counterexample_count += 1
        
        print(f"N = {N}: 反例比例 = {counterexample_count}/{total_count} = {counterexample_count/total_count:.4f}")
    
    return pd.DataFrame(results)

def find_critical_boundary():
    """寻找反例出现的临界边界"""
    print("\n" + "=" * 60)
    print("寻找反例出现的临界边界")
    print("=" * 60)
    
    N = 2
    delta_mu = 0.5
    
    # 在delta_s上进行精细搜索
    delta_s_values = np.linspace(-1.0, 0.0, 1000)
    critical_points = []
    
    for delta_s in delta_s_values:
        mu_fused = N * delta_mu
        gamma_fused = 1.0 * np.exp(N * delta_s)
        
        kl_fused = kl_divergence_cauchy(0, 1.0, mu_fused, gamma_fused)
        kl_individual = kl_divergence_cauchy(0, 1.0, delta_mu, 1.0 * np.exp(delta_s))
        
        ratio = kl_fused / (N * kl_individual)
        
        if abs(ratio - 1.0) < 0.001:  # 接近临界点
            critical_points.append((delta_s, ratio))
    
    if critical_points:
        print(f"发现临界点（delta_mu = {delta_mu}, N = {N}）:")
        for delta_s, ratio in critical_points[:5]:  # 显示前5个
            print(f"  delta_s ≈ {delta_s:.6f}, ratio ≈ {ratio:.6f}")
    
    return critical_points

def analyze_scaling_behavior():
    """分析反例随N的缩放行为"""
    print("\n" + "=" * 60)
    print("分析反例随N的缩放行为")
    print("=" * 60)
    
    delta_mu = 0.5
    delta_s = -0.5
    N_values = range(2, 11)
    
    scaling_data = []
    
    for N in N_values:
        mu_fused = N * delta_mu
        gamma_fused = 1.0 * np.exp(N * delta_s)
        
        kl_fused = kl_divergence_cauchy(0, 1.0, mu_fused, gamma_fused)
        kl_individual = kl_divergence_cauchy(0, 1.0, delta_mu, 1.0 * np.exp(delta_s))
        
        ratio = kl_fused / (N * kl_individual)
        inequality_holds = ratio <= 1.0
        
        scaling_data.append({
            'N': N,
            'mu_fused': mu_fused,
            'gamma_fused': gamma_fused,
            'kl_fused': kl_fused,
            'kl_individual': kl_individual,
            'ratio': ratio,
            'inequality_holds': inequality_holds
        })
        
        print(f"N = {N:2d}: mu_fused = {mu_fused:.1f}, gamma_fused = {gamma_fused:.6f}, "
              f"ratio = {ratio:.6f}, holds = {inequality_holds}")
    
    return scaling_data

def theoretical_analysis():
    """理论分析：为什么会出现反例"""
    print("\n" + "=" * 60)
    print("理论分析：反例出现的数学机制")
    print("=" * 60)
    
    print("KL散度公式：D_KL(P_0 || P) = log((γ_0 + γ)² + (μ_0 - μ)²) - log(4γ_0γ)")
    print("\n对于我们的设置（μ_0 = 0, γ_0 = 1）：")
    print("D_KL(P_0 || P) = log((1 + γ)² + μ²) - log(4γ)")
    
    print("\n单个扰动分布：")
    print("μ_k = δμ, γ_k = exp(δs)")
    print("D_KL(P_0 || P_k) = log((1 + exp(δs))² + δμ²) - log(4exp(δs))")
    
    print("\n融合分布：")
    print("μ_fused = N·δμ, γ_fused = exp(N·δs)")
    print("D_KL(P_0 || P_fused) = log((1 + exp(N·δs))² + (N·δμ)²) - log(4exp(N·δs))")
    
    print("\n关键观察：")
    print("1. 当δs < 0时，exp(N·δs) << exp(δs)，融合分布尺度参数急剧减小")
    print("2. 位置参数线性累积：N·δμ")
    print("3. KL散度对小尺度参数非常敏感")
    print("4. 这种非线性效应可能导致融合分布KL散度超过线性组合")

def visualize_counterexample():
    """可视化反例现象"""
    print("\n" + "=" * 60)
    print("生成反例可视化图表")
    print("=" * 60)
    
    # 创建参数网格
    delta_mu_range = np.linspace(0.1, 1.0, 50)
    delta_s_range = np.linspace(-1.0, 0.0, 50)
    
    N = 2
    ratio_grid = np.zeros((len(delta_s_range), len(delta_mu_range)))
    
    for i, delta_s in enumerate(delta_s_range):
        for j, delta_mu in enumerate(delta_mu_range):
            mu_fused = N * delta_mu
            gamma_fused = 1.0 * np.exp(N * delta_s)
            
            kl_fused = kl_divergence_cauchy(0, 1.0, mu_fused, gamma_fused)
            kl_individual = kl_divergence_cauchy(0, 1.0, delta_mu, 1.0 * np.exp(delta_s))
            
            ratio = kl_fused / (N * kl_individual) if kl_individual > 0 else 0
            ratio_grid[i, j] = ratio
    
    # 创建图表
    plt.figure(figsize=(12, 10))
    
    # 子图1：比率热图
    im = plt.imshow(ratio_grid, extent=[delta_mu_range[0], delta_mu_range[-1], 
                                       delta_s_range[0], delta_s_range[-1]], 
                   aspect='auto', origin='lower', cmap='RdYlBu_r', vmin=0, vmax=max(2.5, np.max(ratio_grid)))
    plt.colorbar(im, label='KL Ratio (KL_fused / (N * KL_individual))')
    plt.contour(delta_mu_range, delta_s_range, ratio_grid, levels=[1.0], colors='black', linewidths=2)
    plt.xlabel('δμ (Location Perturbation)')
    plt.ylabel('δs (Log-Scale Perturbation)')
    plt.title(f'KL Divergence Ratio Heatmap (N={N})')
    
    # 子图2：反例区域
    plt.subplot(2, 2, 2)
    counterexample_grid = (ratio_grid > 1.0).astype(int)
    plt.imshow(counterexample_grid, extent=[delta_mu_range[0], delta_mu_range[-1], 
                                           delta_s_range[0], delta_s_range[-1]], 
               aspect='auto', origin='lower', cmap='coolwarm', vmin=0, vmax=1)
    cbar = plt.colorbar(ticks=[0, 1])
    cbar.ax.set_yticklabels(['Inequality Holds', 'Counterexample'])
    plt.xlabel('δμ (Location Perturbation)')
    plt.ylabel('δs (Log-Scale Perturbation)')
    plt.title(f'Counterexample Region (N={N})')
    
    # 子图3：特定切片分析
    plt.subplot(2, 2, 3)
    delta_mu_fixed = 0.5
    idx = np.argmin(np.abs(delta_mu_range - delta_mu_fixed))
    ratio_slice = ratio_grid[:, idx]
    plt.plot(delta_s_range, ratio_slice, 'b-', linewidth=2)
    plt.axhline(y=1.0, color='r', linestyle='--', label='Threshold (Ratio=1)')
    plt.xlabel('δs (Log-Scale Perturbation)')
    plt.ylabel('KL Divergence Ratio')
    plt.title(f'Slice at δμ={delta_mu_fixed:.2f} (N={N})')
    plt.legend()
    plt.grid(True)
    
    # 子图4：不同N值的比较
    plt.subplot(2, 2, 4)
    delta_mu_fixed = 0.5
    delta_s_fixed = -0.5
    N_range = range(2, 8)
    ratios_vs_N = []
    
    for N in N_range:
        mu_fused = N * delta_mu_fixed
        gamma_fused = 1.0 * np.exp(N * delta_s_fixed)
        kl_fused = kl_divergence_cauchy(0, 1.0, mu_fused, gamma_fused)
        kl_individual = kl_divergence_cauchy(0, 1.0, delta_mu_fixed, 1.0 * np.exp(delta_s_fixed))
        ratio = kl_fused / (N * kl_individual)
        ratios_vs_N.append(ratio)
    
    plt.plot(N_range, ratios_vs_N, 'ro-', linewidth=2, markersize=6)
    plt.axhline(y=1.0, color='b', linestyle='--', label='Threshold (Ratio=1)')
    plt.xlabel('N (Number of Distributions)')
    plt.ylabel('KL Divergence Ratio')
    plt.title(f'Ratio vs. N (δμ={delta_mu_fixed:.2f}, δs={delta_s_fixed:.2f})')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.suptitle(f'Counterexample Analysis for KL Divergence Inequality (N={N_range[0]} to {N_range[-1]})', fontsize=16)
    
    # 确保results目录存在
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/counterexample_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形而不显示
    
    print("Chart saved as 'results/counterexample_analysis.png'")

if __name__ == "__main__":
    # 执行所有分析
    df_results = analyze_counterexample_region()
    critical_points = find_critical_boundary()
    scaling_data = analyze_scaling_behavior()
    theoretical_analysis()
    visualize_counterexample()
    
    # 保存详细结果
    os.makedirs('results', exist_ok=True)
    df_results.to_csv('results/counterexample_detailed_analysis.csv', index=False)
    print(f"\n详细结果已保存到 'results/counterexample_detailed_analysis.csv'")
    
    # 统计总结
    print("\n" + "=" * 60)
    print("总结统计")
    print("=" * 60)
    
    for N in [2, 3, 4, 5]:
        subset = df_results[df_results['N'] == N]
        counterexample_rate = (subset['inequality_holds'] == False).mean()
        print(f"N = {N}: 反例比例 = {counterexample_rate:.4f}")
    
    print(f"\n总体反例比例 = {(df_results['inequality_holds'] == False).mean():.4f}") 