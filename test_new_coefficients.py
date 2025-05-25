#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os

def kl_divergence_cauchy(mu_0, gamma_0, mu, gamma):
    """计算两个柯西分布之间的KL散度"""
    return np.log((gamma_0 + gamma)**2 + (mu_0 - mu)**2) - np.log(4 * gamma_0 * gamma)

def analyze_coefficient_impact():
    """分析不同系数(N, N^2, e^N)对KL散度不等式成立情况的影响"""
    print("分析不同系数对KL散度不等式的影响...")
    print("="*60)
    
    # 测试参数范围，聚焦于已知反例区域
    N_values = [2, 3, 4, 5]
    delta_mu_values = np.linspace(0.1, 10.0, 30)  # 位置参数扰动 (扩大范围)
    delta_s_values = np.linspace(-5.0, -0.01, 30)  # 对数尺度参数扰动(负值) (扩大范围)
    
    # 初始化结果存储
    results = []
    
    # 基准分布参数
    mu_0 = 0.0
    gamma_0 = 1.0
    
    for N in N_values:
        for delta_mu in delta_mu_values:
            for delta_s in delta_s_values:
                # 扰动分布参数 (同向相同扰动)
                mu_k = mu_0 + delta_mu
                gamma_k = gamma_0 * np.exp(delta_s)
                
                # 融合分布参数
                mu_fused = mu_0 + N * delta_mu
                gamma_fused = gamma_0 * np.exp(N * delta_s)
                
                # 计算KL散度
                D_indiv = kl_divergence_cauchy(mu_0, gamma_0, mu_k, gamma_k)
                D_fused = kl_divergence_cauchy(mu_0, gamma_0, mu_fused, gamma_fused)
                
                # 检查不同系数下的不等式成立情况
                ratio_N = D_fused / (N * N * D_indiv)      # 原始系数 N
                ratio_N2 = D_fused / (N**2 * N * D_indiv)  # 系数 N^2
                ratio_eN = D_fused / (np.exp(N) * N * D_indiv)  # 系数 e^N
                
                # 不等式成立判断
                inequality_N = ratio_N <= 1.0   # D_fused <= N * N * D_indiv
                inequality_N2 = ratio_N2 <= 1.0  # D_fused <= N^2 * N * D_indiv
                inequality_eN = ratio_eN <= 1.0  # D_fused <= e^N * N * D_indiv
                
                # 存储结果
                results.append({
                    'N': N,
                    'delta_mu': delta_mu,
                    'delta_s': delta_s,
                    'D_indiv': D_indiv,
                    'D_fused': D_fused,
                    'ratio_N': ratio_N,
                    'ratio_N2': ratio_N2, 
                    'ratio_eN': ratio_eN,
                    'inequality_N': inequality_N,
                    'inequality_N2': inequality_N2,
                    'inequality_eN': inequality_eN
                })
    
    # 转换为DataFrame
    df = pd.DataFrame(results)
    
    # 统计各系数下不等式成立的比例
    for N_val in N_values:
        df_n = df[df['N'] == N_val]
        count_N = df_n['inequality_N'].sum()
        count_N2 = df_n['inequality_N2'].sum()
        count_eN = df_n['inequality_eN'].sum()
        total = len(df_n)
        
        print(f"N = {N_val}:")
        print(f"  系数为N时不等式成立比例: {count_N/total:.4f} ({count_N}/{total})")
        print(f"  系数为N^2时不等式成立比例: {count_N2/total:.4f} ({count_N2}/{total})")
        print(f"  系数为e^N时不等式成立比例: {count_eN/total:.4f} ({count_eN}/{total})")
        
        # 如果存在反例，找出最大的比率
        if count_N < total:
            max_ratio_N = df_n[~df_n['inequality_N']]['ratio_N'].max()
            worst_case_N = df_n[df_n['ratio_N'] == max_ratio_N].iloc[0]
            print(f"  系数为N时最大比率: {max_ratio_N:.4f} (delta_mu={worst_case_N['delta_mu']:.2f}, delta_s={worst_case_N['delta_s']:.2f})")
        
        if count_N2 < total:
            max_ratio_N2 = df_n[~df_n['inequality_N2']]['ratio_N2'].max()
            worst_case_N2 = df_n[df_n['ratio_N2'] == max_ratio_N2].iloc[0]
            print(f"  系数为N^2时最大比率: {max_ratio_N2:.4f} (delta_mu={worst_case_N2['delta_mu']:.2f}, delta_s={worst_case_N2['delta_s']:.2f})")
        
        if count_eN < total:
            max_ratio_eN = df_n[~df_n['inequality_eN']]['ratio_eN'].max()
            worst_case_eN = df_n[df_n['ratio_eN'] == max_ratio_eN].iloc[0]
            print(f"  系数为e^N时最大比率: {max_ratio_eN:.4f} (delta_mu={worst_case_eN['delta_mu']:.2f}, delta_s={worst_case_eN['delta_s']:.2f})")
    
    # 保存结果到CSV文件
    os.makedirs('results', exist_ok=True)
    df.to_csv('results/coefficient_comparison_results.csv', index=False)
    print(f"结果已保存到 results/coefficient_comparison_results.csv")
    
    # 可视化
    plot_coefficient_comparison(df)
    
    return df

def plot_coefficient_comparison(df):
    """可视化不同系数下不等式成立的情况"""
    # 设置图表
    fig = plt.figure(figsize=(18, 12))
    plt.suptitle('KL Divergence Inequality Performance under Different Coefficients (N, N^2, e^N)', fontsize=16)
    
    # 每个N值创建一行子图
    N_values = sorted(df['N'].unique())
    
    for i, N_val in enumerate(N_values):
        df_n = df[df['N'] == N_val]
        
        # 系数为N时的反例区域
        ax1 = fig.add_subplot(len(N_values), 3, i*3+1)
        scatter = ax1.scatter(df_n['delta_mu'], df_n['delta_s'], 
                             c=df_n['ratio_N'], cmap='coolwarm', 
                             vmin=0.0, vmax=min(2.0, df_n['ratio_N'].max()*1.1))
        plt.colorbar(scatter, ax=ax1, label='ratio_N')
        ax1.set_xlabel('delta_mu')
        ax1.set_ylabel('delta_s')
        ax1.set_title(f'N={N_val}, Coefficient=N')
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax1.axvline(x=0, color='black', linestyle='--', alpha=0.3)
        # 绘制比率=1的等高线
        cs = ax1.contour(sorted(df_n['delta_mu'].unique()), 
                         sorted(df_n['delta_s'].unique()), 
                         df_n.pivot(index='delta_s', columns='delta_mu', values='ratio_N').values,
                         levels=[1.0], colors='black')
        ax1.clabel(cs, inline=1, fontsize=10)
        
        # 系数为N^2时的反例区域
        ax2 = fig.add_subplot(len(N_values), 3, i*3+2)
        scatter = ax2.scatter(df_n['delta_mu'], df_n['delta_s'], 
                             c=df_n['ratio_N2'], cmap='coolwarm', 
                             vmin=0.0, vmax=min(2.0, df_n['ratio_N2'].max()*1.1))
        plt.colorbar(scatter, ax=ax2, label='ratio_N2')
        ax2.set_xlabel('delta_mu')
        ax2.set_ylabel('delta_s')
        ax2.set_title(f'N={N_val}, Coefficient=N^2')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax2.axvline(x=0, color='black', linestyle='--', alpha=0.3)
        # 绘制比率=1的等高线
        cs = ax2.contour(sorted(df_n['delta_mu'].unique()), 
                         sorted(df_n['delta_s'].unique()), 
                         df_n.pivot(index='delta_s', columns='delta_mu', values='ratio_N2').values,
                         levels=[1.0], colors='black')
        ax2.clabel(cs, inline=1, fontsize=10)
        
        # 系数为e^N时的反例区域
        ax3 = fig.add_subplot(len(N_values), 3, i*3+3)
        scatter = ax3.scatter(df_n['delta_mu'], df_n['delta_s'], 
                             c=df_n['ratio_eN'], cmap='coolwarm', 
                             vmin=0.0, vmax=min(2.0, df_n['ratio_eN'].max()*1.1))
        plt.colorbar(scatter, ax=ax3, label='ratio_eN')
        ax3.set_xlabel('delta_mu')
        ax3.set_ylabel('delta_s')
        ax3.set_title(f'N={N_val}, Coefficient=e^N')
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax3.axvline(x=0, color='black', linestyle='--', alpha=0.3)
        # 绘制比率=1的等高线
        cs = ax3.contour(sorted(df_n['delta_mu'].unique()), 
                         sorted(df_n['delta_s'].unique()), 
                         df_n.pivot(index='delta_s', columns='delta_mu', values='ratio_eN').values,
                         levels=[1.0], colors='black')
        ax3.clabel(cs, inline=1, fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('results/coefficient_comparison.png', dpi=300, bbox_inches='tight')
    print(f"图表已保存到 results/coefficient_comparison.png")

def find_threshold_coefficient():
    """寻找在给定参数空间内使不等式完全成立的临界系数C"""
    print("\n寻找临界系数C...")
    print("="*60)
    
    # 测试参数范围，聚焦于已知反例区域
    N_values = [2, 3, 4, 5]
    delta_mu_values = np.linspace(0.1, 10.0, 30)  # 位置参数扰动 (扩大范围)
    delta_s_values = np.linspace(-5.0, -0.01, 30)  # 对数尺度参数扰动(负值) (扩大范围)
    
    # 为每个N值找到临界系数
    for N in N_values:
        max_ratio = 0
        worst_case = None
        
        for delta_mu in delta_mu_values:
            for delta_s in delta_s_values:
                # 扰动分布参数 (同向相同扰动)
                mu_k = 0.0 + delta_mu
                gamma_k = 1.0 * np.exp(delta_s)
                
                # 融合分布参数
                mu_fused = 0.0 + N * delta_mu
                gamma_fused = 1.0 * np.exp(N * delta_s)
                
                # 计算KL散度
                D_indiv = kl_divergence_cauchy(0.0, 1.0, mu_k, gamma_k)
                D_fused = kl_divergence_cauchy(0.0, 1.0, mu_fused, gamma_fused)
                
                # 计算比率：D_fused / (N * D_indiv)
                # 我们需要找到C使得 D_fused <= C * N * D_indiv
                ratio = D_fused / (N * D_indiv)
                
                if ratio > max_ratio:
                    max_ratio = ratio
                    worst_case = (delta_mu, delta_s)
        
        # 临界系数C应该至少是max_ratio
        print(f"N = {N}:")
        print(f"  最大比率 D_fused/(N*D_indiv) = {max_ratio:.4f}")
        print(f"  出现在 delta_mu = {worst_case[0]:.2f}, delta_s = {worst_case[1]:.2f}")
        print(f"  使不等式完全成立的临界系数 C >= {max_ratio:.4f}")
        print(f"  与N, N^2, e^N的比较:")
        print(f"    N = {N:.4f}")
        print(f"    N^2 = {N**2:.4f}")
        print(f"    e^N = {np.exp(N):.4f}")
        print(f"  结论: 最大比率 {max_ratio:.4f} {'<' if max_ratio < N else '>'} N")
        print(f"         最大比率 {max_ratio:.4f} {'<' if max_ratio < N**2 else '>'} N^2")
        print(f"         最大比率 {max_ratio:.4f} {'<' if max_ratio < np.exp(N) else '>'} e^N")

if __name__ == "__main__":
    df = analyze_coefficient_impact()
    find_threshold_coefficient() 