#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KL散度不等式实验结果可视化

本脚本用于生成KL散度不等式实验结果的可视化图表和数据表。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from matplotlib.colors import LinearSegmentedColormap

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 创建结果目录
os.makedirs('../visualizations', exist_ok=True)

def load_experiment_results():
    """
    加载实验结果数据
    
    返回:
    包含各实验结果的字典
    """
    results = {}
    
    # 加载实验1：仅位置参数扰动
    try:
        results['position_only'] = pd.read_csv('../experiments/results/experiment_position_only.csv')
    except FileNotFoundError:
        print("警告：找不到位置参数扰动实验结果文件")
    
    # 加载实验2：仅尺度参数扰动
    try:
        results['scale_only'] = pd.read_csv('../experiments/results/experiment_scale_only.csv')
    except FileNotFoundError:
        print("警告：找不到尺度参数扰动实验结果文件")
    
    # 加载实验3：同时考虑位置和尺度参数扰动
    try:
        results['both_params'] = pd.read_csv('../experiments/results/experiment_both_params.csv')
    except FileNotFoundError:
        print("警告：找不到双参数扰动实验结果文件")
    
    # 加载实验4：多维情况
    try:
        results['multivariate'] = pd.read_csv('../experiments/results/experiment_multivariate.csv')
    except FileNotFoundError:
        print("警告：找不到多维实验结果文件")
    
    # 加载实验5：极端情况
    try:
        results['extreme_cases'] = pd.read_csv('../experiments/results/experiment_extreme_cases.csv')
    except FileNotFoundError:
        print("警告：找不到极端情况实验结果文件")
    
    # 加载实验6：最优常数
    try:
        results['optimal_constant'] = pd.read_csv('../experiments/results/experiment_optimal_constant.csv')
    except FileNotFoundError:
        print("警告：找不到最优常数实验结果文件")
    
    # 加载汇总结果
    try:
        with open('../experiments/results/summary.json', 'r') as f:
            results['summary'] = json.load(f)
    except FileNotFoundError:
        print("警告：找不到汇总结果文件")
    
    return results

def visualize_position_only(df):
    """
    可视化仅位置参数扰动的实验结果
    
    参数:
    df: 实验结果数据框
    """
    if df is None or df.empty:
        print("没有位置参数扰动的实验数据可用于可视化")
        return
    
    # 图1：不同N值下的KL散度比较（同向扰动）
    plt.figure(figsize=(10, 6))
    same_dir = df[df['pattern'] == 'same_direction']
    for n in same_dir['N'].unique():
        subset = same_dir[same_dir['N'] == n]
        plt.plot(subset['delta_mu'], subset['kl_fused'], marker='o', label=f'KL_fused (N={n})')
        plt.plot(subset['delta_mu'], subset['right_side'], marker='x', linestyle='--', label=f'N*sum KL (N={n})')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('位置参数扰动大小 (delta_mu)')
    plt.ylabel('KL散度')
    plt.title('同向位置参数扰动下的KL散度比较')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../visualizations/position_only_same_direction.png', dpi=300)
    
    # 图2：不同N值下的KL散度比较（反向扰动）
    plt.figure(figsize=(10, 6))
    opposite_dir = df[df['pattern'] == 'opposite_direction']
    if not opposite_dir.empty:
        for n in opposite_dir['N'].unique():
            subset = opposite_dir[opposite_dir['N'] == n]
            plt.plot(subset['delta_mu'], subset['kl_fused'], marker='o', label=f'KL_fused (N={n})')
            plt.plot(subset['delta_mu'], subset['right_side'], marker='x', linestyle='--', label=f'N*sum KL (N={n})')
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('位置参数扰动大小 (delta_mu)')
        plt.ylabel('KL散度')
        plt.title('反向位置参数扰动下的KL散度比较')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('../visualizations/position_only_opposite_direction.png', dpi=300)
    
    # 图3：不等式成立比例
    plt.figure(figsize=(10, 6))
    result_by_n = df.groupby(['N', 'pattern'])['inequality_holds'].mean().reset_index()
    result_by_n_pivot = result_by_n.pivot(index='N', columns='pattern', values='inequality_holds')
    
    ax = result_by_n_pivot.plot(kind='bar', figsize=(10, 6))
    plt.xlabel('扰动分布数量 (N)')
    plt.ylabel('不等式成立比例')
    plt.title('不同N值和扰动模式下不等式成立的比例')
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    
    # 在柱状图上添加具体数值
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f')
    
    plt.tight_layout()
    plt.savefig('../visualizations/position_only_inequality_holds.png', dpi=300)
    
    # 图4：KL散度比率分布
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='N', y='ratio', hue='pattern', data=df)
    plt.axhline(y=1, color='r', linestyle='--')
    plt.xlabel('扰动分布数量 (N)')
    plt.ylabel('KL_fused / (N*sum KL) 比率')
    plt.title('不同N值和扰动模式下KL散度比率的分布')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../visualizations/position_only_ratio_boxplot.png', dpi=300)

def visualize_scale_only(df):
    """
    可视化仅尺度参数扰动的实验结果
    
    参数:
    df: 实验结果数据框
    """
    if df is None or df.empty:
        print("没有尺度参数扰动的实验数据可用于可视化")
        return
    
    # 图1：不同N值下的KL散度比较（同向扰动）
    plt.figure(figsize=(10, 6))
    same_dir = df[df['pattern'] == 'same_direction']
    for n in same_dir['N'].unique():
        subset = same_dir[same_dir['N'] == n]
        plt.plot(subset['delta_s'], subset['kl_fused'], marker='o', label=f'KL_fused (N={n})')
        plt.plot(subset['delta_s'], subset['right_side'], marker='x', linestyle='--', label=f'N*sum KL (N={n})')
    
    plt.xlabel('对数尺度参数扰动 (delta_s)')
    plt.ylabel('KL散度')
    plt.title('同向尺度参数扰动下的KL散度比较')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../visualizations/scale_only_same_direction.png', dpi=300)
    
    # 图2：不同N值下的KL散度比较（反向扰动）
    plt.figure(figsize=(10, 6))
    opposite_dir = df[df['pattern'] == 'opposite_direction']
    if not opposite_dir.empty:
        for n in opposite_dir['N'].unique():
            subset = opposite_dir[opposite_dir['N'] == n]
            plt.plot(subset['delta_s'], subset['kl_fused'], marker='o', label=f'KL_fused (N={n})')
            plt.plot(subset['delta_s'], subset['right_side'], marker='x', linestyle='--', label=f'N*sum KL (N={n})')
        
        plt.xlabel('对数尺度参数扰动 (delta_s)')
        plt.ylabel('KL散度')
        plt.title('反向尺度参数扰动下的KL散度比较')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('../visualizations/scale_only_opposite_direction.png', dpi=300)
    
    # 图3：不等式成立比例
    plt.figure(figsize=(10, 6))
    result_by_n = df.groupby(['N', 'pattern'])['inequality_holds'].mean().reset_index()
    result_by_n_pivot = result_by_n.pivot(index='N', columns='pattern', values='inequality_holds')
    
    ax = result_by_n_pivot.plot(kind='bar', figsize=(10, 6))
    plt.xlabel('扰动分布数量 (N)')
    plt.ylabel('不等式成立比例')
    plt.title('不同N值和扰动模式下不等式成立的比例')
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    
    # 在柱状图上添加具体数值
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f')
    
    plt.tight_layout()
    plt.savefig('../visualizations/scale_only_inequality_holds.png', dpi=300)
    
    # 图4：KL散度比率分布
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='N', y='ratio', hue='pattern', data=df)
    plt.axhline(y=1, color='r', linestyle='--')
    plt.xlabel('扰动分布数量 (N)')
    plt.ylabel('KL_fused / (N*sum KL) 比率')
    plt.title('不同N值和扰动模式下KL散度比率的分布')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../visualizations/scale_only_ratio_boxplot.png', dpi=300)

def visualize_both_params(df):
    """
    可视化同时考虑位置和尺度参数扰动的实验结果
    
    参数:
    df: 实验结果数据框
    """
    if df is None or df.empty:
        print("没有双参数扰动的实验数据可用于可视化")
        return
    
    # 图1：不同delta_mu和delta_s组合下的不等式成立情况热图（同向扰动）
    plt.figure(figsize=(12, 8))
    for n in df['N'].unique():
        plt.figure(figsize=(10, 8))
        same_dir = df[(df['pattern'] == 'same_direction') & (df['N'] == n)]
        
        if not same_dir.empty:
            # 创建透视表
            pivot = same_dir.pivot_table(
                index='delta_mu', 
                columns='delta_s', 
                values='inequality_holds',
                aggfunc='mean'
            )
            
            # 绘制热图
            sns.heatmap(pivot, annot=True, cmap='RdYlGn', vmin=0, vmax=1, fmt='.2f')
            plt.title(f'同向扰动下不等式成立情况 (N={n})')
            plt.xlabel('对数尺度参数扰动 (delta_s)')
            plt.ylabel('位置参数扰动 (delta_mu)')
            plt.tight_layout()
            plt.savefig(f'../visualizations/both_params_same_direction_N{n}.png', dpi=300)
    
    # 图2：不同delta_mu和delta_s组合下的不等式成立情况热图（反向扰动）
    for n in df['N'].unique():
        if n == 1:  # N=1时没有反向扰动
            continue
            
        plt.figure(figsize=(10, 8))
        opposite_dir = df[(df['pattern'] == 'opposite_direction') & (df['N'] == n)]
        
        if not opposite_dir.empty:
            # 创建透视表
            pivot = opposite_dir.pivot_table(
                index='delta_mu', 
                columns='delta_s', 
                values='inequality_holds',
                aggfunc='mean'
            )
            
            # 绘制热图
            sns.heatmap(pivot, annot=True, cmap='RdYlGn', vmin=0, vmax=1, fmt='.2f')
            plt.title(f'反向扰动下不等式成立情况 (N={n})')
            plt.xlabel('对数尺度参数扰动 (delta_s)')
            plt.ylabel('位置参数扰动 (delta_mu)')
            plt.tight_layout()
            plt.savefig(f'../visualizations/both_params_opposite_direction_N{n}.png', dpi=300)
    
    # 图3：不同N值下的KL散度比率分布
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='N', y='ratio', hue='pattern', data=df)
    plt.axhline(y=1, color='r', linestyle='--')
    plt.xlabel('扰动分布数量 (N)')
    plt.ylabel('KL_fused / (N*sum KL) 比率')
    plt.title('不同N值和扰动模式下KL散度比率的分布')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../visualizations/both_params_ratio_boxplot.png', dpi=300)
    
    # 图4：不同N值下不等式成立的比例
    plt.figure(figsize=(10, 6))
    result_by_n = df.groupby(['N', 'pattern'])['inequality_holds'].mean().reset_index()
    result_by_n_pivot = result_by_n.pivot(index='N', columns='pattern', values='inequality_holds')
    
    ax = result_by_n_pivot.plot(kind='bar', figsize=(10, 6))
    plt.xlabel('扰动分布数量 (N)')
    plt.ylabel('不等式成立比例')
    plt.title('不同N值和扰动模式下不等式成立的比例')
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    
    # 在柱状图上添加具体数值
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f')
    
    plt.tight_layout()
    plt.savefig('../visualizations/both_params_inequality_holds.png', dpi=300)

def visualize_multivariate(df):
    """
    可视化多维情况下的实验结果
    
    参数:
    df: 实验结果数据框
    """
    if df is None or df.empty:
        print("没有多维实验数据可用于可视化")
        return
    
    # 图1：不同维度d和扰动数量N下不等式成立的比例
    plt.figure(figsize=(12, 8))
    result_by_d_n = df.groupby(['d', 'N'])['inequality_holds'].mean().reset_index()
    pivot = result_by_d_n.pivot(index='d', columns='N', values='inequality_holds')
    
    sns.heatmap(pivot, annot=True, cmap='RdYlGn', vmin=0, vmax=1, fmt='.2f')
    plt.title('不同维度d和扰动数量N下不等式成立的比例')
    plt.xlabel('扰动分布数量 (N)')
    plt.ylabel('维度 (d)')
    plt.tight_layout()
    plt.savefig('../visualizations/multivariate_inequality_holds.png', dpi=300)
    
    # 图2：不同维度d和扰动数量N下KL散度比率的箱线图
    plt.figure(figsize=(14, 10))
    
    # 创建子图网格
    d_values = sorted(df['d'].unique())
    n_values = sorted(df['N'].unique())
    fig, axes = plt.subplots(len(d_values), 1, figsize=(12, 4*len(d_values)), sharex=True)
    
    for i, d in enumerate(d_values):
        subset = df[df['d'] == d]
        sns.boxplot(x='N', y='ratio', data=subset, ax=axes[i])
        axes[i].axhline(y=1, color='r', linestyle='--')
        axes[i].set_title(f'维度 d={d}')
        axes[i].set_ylabel('KL_fused / (N*sum KL) 比率')
        axes[i].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('扰动分布数量 (N)')
    plt.tight_layout()
    plt.savefig('../visualizations/multivariate_ratio_boxplot.png', dpi=300)
    
    # 图3：KL散度比率的散点图
    plt.figure(figsize=(12, 8))
    for d in df['d'].unique():
        for n in df['N'].unique():
            subset = df[(df['d'] == d) & (df['N'] == n)]
            plt.scatter(subset['kl_fused'], subset['right_side'], alpha=0.6, label=f'd={d}, N={n}')
    
    # 添加对角线
    max_val = max(df['kl_fused'].max(), df['right_side'].max())
    plt.plot([0, max_val], [0, max_val], 'r--')
    
    plt.xlabel('KL_fused')
    plt.ylabel('N*sum KL')
    plt.title('KL散度比较散点图')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../visualizations/multivariate_scatter.png', dpi=300)

def visualize_extreme_cases(df):
    """
    可视化极端情况下的实验结果
    
    参数:
    df: 实验结果数据框
    """
    if df is None or df.empty:
        print("没有极端情况实验数据可用于可视化")
        return
    
    # 图1：不同极端值下的KL散度比较（位置参数）
    plt.figure(figsize=(12, 8))
    position = df[df['param_type'] == 'position']
    
    for n in position['N'].unique():
        subset = position[position['N'] == n]
        plt.plot(subset['extreme_value'], subset['kl_fused'], marker='o', label=f'KL_fused (N={n})')
        plt.plot(subset['extreme_value'], subset['right_side'], marker='x', linestyle='--', label=f'N*sum KL (N={n})')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('极端位置参数值')
    plt.ylabel('KL散度')
    plt.title('极端位置参数值下的KL散度比较')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../visualizations/extreme_cases_position.png', dpi=300)
    
    # 图2：不同极端值下的KL散度比较（尺度参数）
    plt.figure(figsize=(12, 8))
    scale = df[df['param_type'] == 'scale']
    
    for n in scale['N'].unique():
        subset = scale[scale['N'] == n]
        plt.plot(subset['extreme_value'], subset['kl_fused'], marker='o', label=f'KL_fused (N={n})')
        plt.plot(subset['extreme_value'], subset['right_side'], marker='x', linestyle='--', label=f'N*sum KL (N={n})')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('极端尺度参数值')
    plt.ylabel('KL散度')
    plt.title('极端尺度参数值下的KL散度比较')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../visualizations/extreme_cases_scale.png', dpi=300)
    
    # 图3：不等式成立情况
    plt.figure(figsize=(12, 8))
    result_by_type = df.groupby(['N', 'param_type'])['inequality_holds'].mean().reset_index()
    result_pivot = result_by_type.pivot(index='N', columns='param_type', values='inequality_holds')
    
    ax = result_pivot.plot(kind='bar', figsize=(10, 6))
    plt.xlabel('扰动分布数量 (N)')
    plt.ylabel('不等式成立比例')
    plt.title('不同N值和参数类型下不等式成立的比例')
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    
    # 在柱状图上添加具体数值
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f')
    
    plt.tight_layout()
    plt.savefig('../visualizations/extreme_cases_inequality_holds.png', dpi=300)
    
    # 图4：KL散度比率
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='N', y='ratio', hue='param_type', data=df)
    plt.axhline(y=1, color='r', linestyle='--')
    plt.xlabel('扰动分布数量 (N)')
    plt.ylabel('KL_fused / (N*sum KL) 比率')
    plt.title('不同N值和参数类型下KL散度比率的分布')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../visualizations/extreme_cases_ratio_boxplot.png', dpi=300)

def visualize_optimal_constant(df):
    """
    可视化最优常数C的实验结果
    
    参数:
    df: 实验结果数据框
    """
    if df is None or df.empty:
        print("没有最优常数实验数据可用于可视化")
        return
    
    # 图1：不同维度d和扰动数量N下的最大比率
    plt.figure(figsize=(12, 8))
    pivot_max = df.pivot(index='d', columns='N', values='max_ratio')
    
    sns.heatmap(pivot_max, annot=True, cmap='YlOrRd', fmt='.2f')
    plt.title('不同维度d和扰动数量N下的最大比率')
    plt.xlabel('扰动分布数量 (N)')
    plt.ylabel('维度 (d)')
    plt.tight_layout()
    plt.savefig('../visualizations/optimal_constant_max_ratio.png', dpi=300)
    
    # 图2：不同维度d和扰动数量N下的平均比率
    plt.figure(figsize=(12, 8))
    pivot_mean = df.pivot(index='d', columns='N', values='mean_ratio')
    
    sns.heatmap(pivot_mean, annot=True, cmap='YlOrRd', fmt='.2f')
    plt.title('不同维度d和扰动数量N下的平均比率')
    plt.xlabel('扰动分布数量 (N)')
    plt.ylabel('维度 (d)')
    plt.tight_layout()
    plt.savefig('../visualizations/optimal_constant_mean_ratio.png', dpi=300)
    
    # 图3：不同维度d和扰动数量N下的95%分位数比率
    plt.figure(figsize=(12, 8))
    pivot_p95 = df.pivot(index='d', columns='N', values='p95_ratio')
    
    sns.heatmap(pivot_p95, annot=True, cmap='YlOrRd', fmt='.2f')
    plt.title('不同维度d和扰动数量N下的95%分位数比率')
    plt.xlabel('扰动分布数量 (N)')
    plt.ylabel('维度 (d)')
    plt.tight_layout()
    plt.savefig('../visualizations/optimal_constant_p95_ratio.png', dpi=300)
    
    # 图4：比率与N的关系
    plt.figure(figsize=(12, 8))
    
    # 对每个维度d，绘制不同统计量随N的变化
    for d in df['d'].unique():
        subset = df[df['d'] == d]
        plt.plot(subset['N'], subset['max_ratio'], marker='o', label=f'最大比率 (d={d})')
        plt.plot(subset['N'], subset['p95_ratio'], marker='s', linestyle='--', label=f'95%分位数 (d={d})')
    
    plt.xlabel('扰动分布数量 (N)')
    plt.ylabel('KL_fused / sum KL 比率')
    plt.title('不同维度d下比率与扰动数量N的关系')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../visualizations/optimal_constant_ratio_vs_N.png', dpi=300)
    
    # 图5：最优常数C的估计
    plt.figure(figsize=(12, 8))
    
    # 计算每个N值下的最大比率（跨所有维度）
    max_by_n = df.groupby('N')['max_ratio'].max().reset_index()
    p99_by_n = df.groupby('N')['p99_ratio'].max().reset_index()
    
    plt.plot(max_by_n['N'], max_by_n['max_ratio'], marker='o', label='最大比率')
    plt.plot(p99_by_n['N'], p99_by_n['p99_ratio'], marker='s', label='99%分位数')
    
    # 拟合曲线 C = N^a
    from scipy.optimize import curve_fit
    
    def power_law(x, a):
        return x ** a
    
    # 对最大比率拟合
    popt_max, _ = curve_fit(power_law, max_by_n['N'], max_by_n['max_ratio'])
    x_fit = np.linspace(min(max_by_n['N']), max(max_by_n['N']), 100)
    y_fit_max = power_law(x_fit, popt_max[0])
    
    # 对99%分位数拟合
    popt_p99, _ = curve_fit(power_law, p99_by_n['N'], p99_by_n['p99_ratio'])
    y_fit_p99 = power_law(x_fit, popt_p99[0])
    
    plt.plot(x_fit, y_fit_max, 'r-', label=f'拟合曲线: C = N^{popt_max[0]:.3f}')
    plt.plot(x_fit, y_fit_p99, 'g-', label=f'拟合曲线: C = N^{popt_p99[0]:.3f}')
    
    plt.xlabel('扰动分布数量 (N)')
    plt.ylabel('最优常数C')
    plt.title('最优常数C与扰动数量N的关系')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../visualizations/optimal_constant_estimation.png', dpi=300)
    
    # 保存拟合结果
    fit_results = {
        'max_ratio_exponent': float(popt_max[0]),
        'p99_ratio_exponent': float(popt_p99[0]),
        'suggested_formula': f"C ≈ N^{max(popt_max[0], popt_p99[0]):.3f}"
    }
    
    with open('../experiments/results/optimal_constant_fit.json', 'w') as f:
        json.dump(fit_results, f, indent=4)

def create_summary_tables():
    """
    创建实验结果的汇总表格
    """
    # 加载所有实验结果
    results = load_experiment_results()
    
    # 表1：不等式成立比例汇总
    summary_data = []
    
    # 位置参数扰动
    if 'position_only' in results:
        df = results['position_only']
        for n in sorted(df['N'].unique()):
            for pattern in sorted(df['pattern'].unique()):
                subset = df[(df['N'] == n) & (df['pattern'] == pattern)]
                summary_data.append({
                    '实验类型': '仅位置参数扰动',
                    '扰动数量 (N)': n,
                    '扰动模式': pattern,
                    '不等式成立比例': subset['inequality_holds'].mean(),
                    '样本数': len(subset)
                })
    
    # 尺度参数扰动
    if 'scale_only' in results:
        df = results['scale_only']
        for n in sorted(df['N'].unique()):
            for pattern in sorted(df['pattern'].unique()):
                subset = df[(df['N'] == n) & (df['pattern'] == pattern)]
                summary_data.append({
                    '实验类型': '仅尺度参数扰动',
                    '扰动数量 (N)': n,
                    '扰动模式': pattern,
                    '不等式成立比例': subset['inequality_holds'].mean(),
                    '样本数': len(subset)
                })
    
    # 双参数扰动
    if 'both_params' in results:
        df = results['both_params']
        for n in sorted(df['N'].unique()):
            for pattern in sorted(df['pattern'].unique()):
                subset = df[(df['N'] == n) & (df['pattern'] == pattern)]
                summary_data.append({
                    '实验类型': '双参数扰动',
                    '扰动数量 (N)': n,
                    '扰动模式': pattern,
                    '不等式成立比例': subset['inequality_holds'].mean(),
                    '样本数': len(subset)
                })
    
    # 多维情况
    if 'multivariate' in results:
        df = results['multivariate']
        for d in sorted(df['d'].unique()):
            for n in sorted(df['N'].unique()):
                subset = df[(df['d'] == d) & (df['N'] == n)]
                summary_data.append({
                    '实验类型': f'多维 (d={d})',
                    '扰动数量 (N)': n,
                    '扰动模式': 'random',
                    '不等式成立比例': subset['inequality_holds'].mean(),
                    '样本数': len(subset)
                })
    
    # 极端情况
    if 'extreme_cases' in results:
        df = results['extreme_cases']
        for n in sorted(df['N'].unique()):
            for param_type in sorted(df['param_type'].unique()):
                subset = df[(df['N'] == n) & (df['param_type'] == param_type)]
                summary_data.append({
                    '实验类型': f'极端情况 ({param_type})',
                    '扰动数量 (N)': n,
                    '扰动模式': 'extreme',
                    '不等式成立比例': subset['inequality_holds'].mean(),
                    '样本数': len(subset)
                })
    
    # 创建汇总表格
    summary_df = pd.DataFrame(summary_data)
    if not summary_df.empty:
        summary_df.to_csv('../experiments/results/inequality_holds_summary.csv', index=False)
        
        # 创建HTML表格
        html_table = summary_df.to_html(index=False)
        with open('../visualizations/inequality_holds_summary.html', 'w') as f:
            f.write(html_table)
    
    # 表2：最优常数C的估计
    if 'optimal_constant' in results:
        df = results['optimal_constant']
        
        # 按N分组，计算跨所有维度的最大比率
        optimal_by_n = df.groupby('N').agg({
            'max_ratio': 'max',
            'p95_ratio': 'max',
            'p99_ratio': 'max',
            'mean_ratio': 'mean'
        }).reset_index()
        
        optimal_by_n.to_csv('../experiments/results/optimal_constant_by_N.csv', index=False)
        
        # 创建HTML表格
        html_table = optimal_by_n.to_html(index=False)
        with open('../visualizations/optimal_constant_by_N.html', 'w') as f:
            f.write(html_table)
    
    # 表3：反例情况汇总
    counterexamples = []
    
    # 检查所有实验中的反例
    for exp_type, df in results.items():
        if exp_type not in ['summary', 'optimal_constant'] and isinstance(df, pd.DataFrame) and not df.empty:
            # 找出不等式不成立的情况
            counter = df[df['inequality_holds'] == False]
            if not counter.empty:
                # 选择比率最高的几个反例
                top_counters = counter.nlargest(min(5, len(counter)), 'ratio')
                
                for _, row in top_counters.iterrows():
                    example = {'实验类型': exp_type}
                    for col in row.index:
                        if col not in ['inequality_holds']:
                            example[col] = row[col]
                    counterexamples.append(example)
    
    # 创建反例表格
    if counterexamples:
        counter_df = pd.DataFrame(counterexamples)
        counter_df.to_csv('../experiments/results/counterexamples.csv', index=False)
        
        # 创建HTML表格
        html_table = counter_df.to_html(index=False)
        with open('../visualizations/counterexamples.html', 'w') as f:
            f.write(html_table)

def run_all_visualizations():
    """
    运行所有可视化函数
    """
    print("开始生成可视化图表和数据表...")
    
    # 加载实验结果
    results = load_experiment_results()
    
    # 运行各实验的可视化
    visualize_position_only(results.get('position_only'))
    visualize_scale_only(results.get('scale_only'))
    visualize_both_params(results.get('both_params'))
    visualize_multivariate(results.get('multivariate'))
    visualize_extreme_cases(results.get('extreme_cases'))
    visualize_optimal_constant(results.get('optimal_constant'))
    
    # 创建汇总表格
    create_summary_tables()
    
    print("所有可视化图表和数据表已生成完成")

if __name__ == "__main__":
    run_all_visualizations()
