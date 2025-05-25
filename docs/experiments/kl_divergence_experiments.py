#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KL散度不等式数值验证实验

本脚本实现了独立柯西分布KL散度不等式的数值验证实验，用于检验以下不等式是否成立：
D_{KL}(P_0 || P_{fused}) <= N * sum_{k=1}^N D_{KL}(P_0 || P_k)

实验覆盖不同维度d、扰动数量N和参数配置，重点关注可能的反例情况。
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product
import os
import json
from tqdm import tqdm

# 创建结果目录
os.makedirs('../visualizations', exist_ok=True)
os.makedirs('../experiments/results', exist_ok=True)

def kl_divergence_cauchy(mu_0, gamma_0, mu, gamma):
    """
    计算一维柯西分布之间的KL散度
    
    参数:
    mu_0, gamma_0: 分布P_0的位置和尺度参数
    mu, gamma: 分布P的位置和尺度参数
    
    返回:
    KL散度 D_KL(P_0 || P)
    """
    return np.log((gamma_0 + gamma)**2 + (mu_0 - mu)**2) - np.log(4 * gamma_0 * gamma)

def kl_divergence_multivariate_cauchy(mu_0, gamma_0, mu, gamma):
    """
    计算多维独立柯西分布之间的KL散度
    
    参数:
    mu_0, gamma_0: 分布P_0的位置和尺度参数向量
    mu, gamma: 分布P的位置和尺度参数向量
    
    返回:
    KL散度 D_KL(P_0 || P)
    """
    d = len(mu_0)
    kl_sum = 0
    for i in range(d):
        kl_sum += kl_divergence_cauchy(mu_0[i], gamma_0[i], mu[i], gamma[i])
    return kl_sum

def experiment_1d_position_only(N_values, delta_mu_values, gamma_0=1.0):
    """
    实验1：一维情况下，仅位置参数扰动的情况
    
    参数:
    N_values: 扰动分布数量列表
    delta_mu_values: 位置参数扰动值列表
    gamma_0: 基准分布的尺度参数
    
    返回:
    结果数据框
    """
    results = []
    
    for N, delta_mu in product(N_values, delta_mu_values):
        # 同向扰动
        delta_mu_k = np.ones(N) * delta_mu
        mu_fused = np.sum(delta_mu_k)
        
        # 计算融合分布的KL散度
        kl_fused = kl_divergence_cauchy(0, gamma_0, mu_fused, gamma_0)
        
        # 计算每个扰动分布的KL散度
        kl_sum = 0
        for k in range(N):
            kl_sum += kl_divergence_cauchy(0, gamma_0, delta_mu_k[k], gamma_0)
        
        # 计算不等式右侧
        right_side = N * kl_sum
        
        # 记录结果
        results.append({
            'N': N,
            'delta_mu': delta_mu,
            'pattern': 'same_direction',
            'kl_fused': kl_fused,
            'right_side': right_side,
            'ratio': kl_fused / right_side if right_side > 0 else float('inf'),
            'inequality_holds': kl_fused <= right_side
        })
        
        # 反向扰动
        if N > 1:
            delta_mu_k = np.zeros(N)
            delta_mu_k[0] = delta_mu
            delta_mu_k[1:] = -delta_mu / (N-1)
            mu_fused = np.sum(delta_mu_k)
            
            # 计算融合分布的KL散度
            kl_fused = kl_divergence_cauchy(0, gamma_0, mu_fused, gamma_0)
            
            # 计算每个扰动分布的KL散度
            kl_sum = 0
            for k in range(N):
                kl_sum += kl_divergence_cauchy(0, gamma_0, delta_mu_k[k], gamma_0)
            
            # 计算不等式右侧
            right_side = N * kl_sum
            
            # 记录结果
            results.append({
                'N': N,
                'delta_mu': delta_mu,
                'pattern': 'opposite_direction',
                'kl_fused': kl_fused,
                'right_side': right_side,
                'ratio': kl_fused / right_side if right_side > 0 else float('inf'),
                'inequality_holds': kl_fused <= right_side
            })
    
    return pd.DataFrame(results)

def experiment_1d_scale_only(N_values, delta_s_values, gamma_0=1.0):
    """
    实验2：一维情况下，仅尺度参数扰动的情况
    
    参数:
    N_values: 扰动分布数量列表
    delta_s_values: 对数尺度参数扰动值列表
    gamma_0: 基准分布的尺度参数
    
    返回:
    结果数据框
    """
    results = []
    
    for N, delta_s in product(N_values, delta_s_values):
        # 同向扰动
        delta_s_k = np.ones(N) * delta_s
        gamma_fused = gamma_0 * np.prod(np.exp(delta_s_k))
        
        # 计算融合分布的KL散度
        kl_fused = kl_divergence_cauchy(0, gamma_0, 0, gamma_fused)
        
        # 计算每个扰动分布的KL散度
        kl_sum = 0
        for k in range(N):
            gamma_k = gamma_0 * np.exp(delta_s_k[k])
            kl_sum += kl_divergence_cauchy(0, gamma_0, 0, gamma_k)
        
        # 计算不等式右侧
        right_side = N * kl_sum
        
        # 记录结果
        results.append({
            'N': N,
            'delta_s': delta_s,
            'pattern': 'same_direction',
            'kl_fused': kl_fused,
            'right_side': right_side,
            'ratio': kl_fused / right_side if right_side > 0 else float('inf'),
            'inequality_holds': kl_fused <= right_side
        })
        
        # 反向扰动
        if N > 1:
            delta_s_k = np.zeros(N)
            delta_s_k[0] = delta_s
            delta_s_k[1:] = -delta_s / (N-1)
            gamma_fused = gamma_0 * np.prod(np.exp(delta_s_k))
            
            # 计算融合分布的KL散度
            kl_fused = kl_divergence_cauchy(0, gamma_0, 0, gamma_fused)
            
            # 计算每个扰动分布的KL散度
            kl_sum = 0
            for k in range(N):
                gamma_k = gamma_0 * np.exp(delta_s_k[k])
                kl_sum += kl_divergence_cauchy(0, gamma_0, 0, gamma_k)
            
            # 计算不等式右侧
            right_side = N * kl_sum
            
            # 记录结果
            results.append({
                'N': N,
                'delta_s': delta_s,
                'pattern': 'opposite_direction',
                'kl_fused': kl_fused,
                'right_side': right_side,
                'ratio': kl_fused / right_side if right_side > 0 else float('inf'),
                'inequality_holds': kl_fused <= right_side
            })
    
    return pd.DataFrame(results)

def experiment_1d_both_params(N_values, delta_mu_values, delta_s_values, gamma_0=1.0):
    """
    实验3：一维情况下，同时考虑位置和尺度参数扰动
    
    参数:
    N_values: 扰动分布数量列表
    delta_mu_values: 位置参数扰动值列表
    delta_s_values: 对数尺度参数扰动值列表
    gamma_0: 基准分布的尺度参数
    
    返回:
    结果数据框
    """
    results = []
    
    for N, delta_mu, delta_s in product(N_values, delta_mu_values, delta_s_values):
        # 同向扰动
        delta_mu_k = np.ones(N) * delta_mu
        delta_s_k = np.ones(N) * delta_s
        
        mu_fused = np.sum(delta_mu_k)
        gamma_fused = gamma_0 * np.prod(np.exp(delta_s_k))
        
        # 计算融合分布的KL散度
        kl_fused = kl_divergence_cauchy(0, gamma_0, mu_fused, gamma_fused)
        
        # 计算每个扰动分布的KL散度
        kl_sum = 0
        for k in range(N):
            gamma_k = gamma_0 * np.exp(delta_s_k[k])
            kl_sum += kl_divergence_cauchy(0, gamma_0, delta_mu_k[k], gamma_k)
        
        # 计算不等式右侧
        right_side = N * kl_sum
        
        # 记录结果
        results.append({
            'N': N,
            'delta_mu': delta_mu,
            'delta_s': delta_s,
            'pattern': 'same_direction',
            'kl_fused': kl_fused,
            'right_side': right_side,
            'ratio': kl_fused / right_side if right_side > 0 else float('inf'),
            'inequality_holds': kl_fused <= right_side
        })
        
        # 反向扰动
        if N > 1:
            # 位置参数反向
            delta_mu_k = np.zeros(N)
            delta_mu_k[0] = delta_mu
            delta_mu_k[1:] = -delta_mu / (N-1)
            
            # 尺度参数反向
            delta_s_k = np.zeros(N)
            delta_s_k[0] = delta_s
            delta_s_k[1:] = -delta_s / (N-1)
            
            mu_fused = np.sum(delta_mu_k)
            gamma_fused = gamma_0 * np.prod(np.exp(delta_s_k))
            
            # 计算融合分布的KL散度
            kl_fused = kl_divergence_cauchy(0, gamma_0, mu_fused, gamma_fused)
            
            # 计算每个扰动分布的KL散度
            kl_sum = 0
            for k in range(N):
                gamma_k = gamma_0 * np.exp(delta_s_k[k])
                kl_sum += kl_divergence_cauchy(0, gamma_0, delta_mu_k[k], gamma_k)
            
            # 计算不等式右侧
            right_side = N * kl_sum
            
            # 记录结果
            results.append({
                'N': N,
                'delta_mu': delta_mu,
                'delta_s': delta_s,
                'pattern': 'opposite_direction',
                'kl_fused': kl_fused,
                'right_side': right_side,
                'ratio': kl_fused / right_side if right_side > 0 else float('inf'),
                'inequality_holds': kl_fused <= right_side
            })
    
    return pd.DataFrame(results)

def experiment_multivariate(d_values, N_values, delta_mu_range, delta_s_range, num_trials=100):
    """
    实验4：多维情况下的随机参数实验
    
    参数:
    d_values: 维度列表
    N_values: 扰动分布数量列表
    delta_mu_range: 位置参数扰动范围 (min, max)
    delta_s_range: 对数尺度参数扰动范围 (min, max)
    num_trials: 每种配置的随机试验次数
    
    返回:
    结果数据框
    """
    results = []
    
    for d, N in product(d_values, N_values):
        for trial in range(num_trials):
            # 基准分布参数
            mu_0 = np.zeros(d)
            gamma_0 = np.ones(d)
            
            # 随机生成扰动参数
            delta_mu_k = np.random.uniform(delta_mu_range[0], delta_mu_range[1], (N, d))
            delta_s_k = np.random.uniform(delta_s_range[0], delta_s_range[1], (N, d))
            
            # 计算融合分布参数
            mu_fused = mu_0 + np.sum(delta_mu_k, axis=0)
            gamma_fused = gamma_0 * np.prod(np.exp(delta_s_k), axis=0)
            
            # 计算融合分布的KL散度
            kl_fused = kl_divergence_multivariate_cauchy(mu_0, gamma_0, mu_fused, gamma_fused)
            
            # 计算每个扰动分布的KL散度
            kl_sum = 0
            for k in range(N):
                mu_k = mu_0 + delta_mu_k[k]
                gamma_k = gamma_0 * np.exp(delta_s_k[k])
                kl_sum += kl_divergence_multivariate_cauchy(mu_0, gamma_0, mu_k, gamma_k)
            
            # 计算不等式右侧
            right_side = N * kl_sum
            
            # 记录结果
            results.append({
                'd': d,
                'N': N,
                'trial': trial,
                'kl_fused': kl_fused,
                'right_side': right_side,
                'ratio': kl_fused / right_side if right_side > 0 else float('inf'),
                'inequality_holds': kl_fused <= right_side
            })
    
    return pd.DataFrame(results)

def experiment_extreme_cases(N_values, extreme_values):
    """
    实验5：极端情况测试
    
    参数:
    N_values: 扰动分布数量列表
    extreme_values: 极端参数值列表
    
    返回:
    结果数据框
    """
    results = []
    
    for N, value in product(N_values, extreme_values):
        # 极端位置参数，尺度参数正常
        delta_mu_k = np.ones(N) * value
        delta_s_k = np.zeros(N)
        
        mu_fused = np.sum(delta_mu_k)
        gamma_fused = np.exp(np.sum(delta_s_k))
        
        # 计算融合分布的KL散度
        kl_fused = kl_divergence_cauchy(0, 1, mu_fused, gamma_fused)
        
        # 计算每个扰动分布的KL散度
        kl_sum = 0
        for k in range(N):
            gamma_k = np.exp(delta_s_k[k])
            kl_sum += kl_divergence_cauchy(0, 1, delta_mu_k[k], gamma_k)
        
        # 计算不等式右侧
        right_side = N * kl_sum
        
        # 记录结果
        results.append({
            'N': N,
            'extreme_value': value,
            'param_type': 'position',
            'kl_fused': kl_fused,
            'right_side': right_side,
            'ratio': kl_fused / right_side if right_side > 0 else float('inf'),
            'inequality_holds': kl_fused <= right_side
        })
        
        # 极端尺度参数，位置参数正常
        delta_mu_k = np.zeros(N)
        delta_s_k = np.ones(N) * value
        
        mu_fused = np.sum(delta_mu_k)
        gamma_fused = np.exp(np.sum(delta_s_k))
        
        # 计算融合分布的KL散度
        kl_fused = kl_divergence_cauchy(0, 1, mu_fused, gamma_fused)
        
        # 计算每个扰动分布的KL散度
        kl_sum = 0
        for k in range(N):
            gamma_k = np.exp(delta_s_k[k])
            kl_sum += kl_divergence_cauchy(0, 1, delta_mu_k[k], gamma_k)
        
        # 计算不等式右侧
        right_side = N * kl_sum
        
        # 记录结果
        results.append({
            'N': N,
            'extreme_value': value,
            'param_type': 'scale',
            'kl_fused': kl_fused,
            'right_side': right_side,
            'ratio': kl_fused / right_side if right_side > 0 else float('inf'),
            'inequality_holds': kl_fused <= right_side
        })
    
    return pd.DataFrame(results)

def experiment_optimal_constant(d_values, N_values, delta_mu_range, delta_s_range, num_trials=100):
    """
    实验6：寻找最优常数C
    
    参数:
    d_values: 维度列表
    N_values: 扰动分布数量列表
    delta_mu_range: 位置参数扰动范围 (min, max)
    delta_s_range: 对数尺度参数扰动范围 (min, max)
    num_trials: 每种配置的随机试验次数
    
    返回:
    结果数据框
    """
    results = []
    
    for d, N in product(d_values, N_values):
        ratios = []
        
        for trial in range(num_trials):
            # 基准分布参数
            mu_0 = np.zeros(d)
            gamma_0 = np.ones(d)
            
            # 随机生成扰动参数
            delta_mu_k = np.random.uniform(delta_mu_range[0], delta_mu_range[1], (N, d))
            delta_s_k = np.random.uniform(delta_s_range[0], delta_s_range[1], (N, d))
            
            # 计算融合分布参数
            mu_fused = mu_0 + np.sum(delta_mu_k, axis=0)
            gamma_fused = gamma_0 * np.prod(np.exp(delta_s_k), axis=0)
            
            # 计算融合分布的KL散度
            kl_fused = kl_divergence_multivariate_cauchy(mu_0, gamma_0, mu_fused, gamma_fused)
            
            # 计算每个扰动分布的KL散度
            kl_sum = 0
            for k in range(N):
                mu_k = mu_0 + delta_mu_k[k]
                gamma_k = gamma_0 * np.exp(delta_s_k[k])
                kl_sum += kl_divergence_multivariate_cauchy(mu_0, gamma_0, mu_k, gamma_k)
            
            # 计算比率
            if kl_sum > 0:
                ratio = kl_fused / kl_sum
                ratios.append(ratio)
        
        if ratios:
            # 记录结果
            results.append({
                'd': d,
                'N': N,
                'min_ratio': min(ratios),
                'max_ratio': max(ratios),
                'mean_ratio': np.mean(ratios),
                'median_ratio': np.median(ratios),
                'p95_ratio': np.percentile(ratios, 95),
                'p99_ratio': np.percentile(ratios, 99),
                'num_trials': len(ratios)
            })
    
    return pd.DataFrame(results)

def run_all_experiments():
    """
    运行所有实验并保存结果
    """
    print("开始运行实验...")
    
    # 实验参数
    N_values = [2, 3, 5, 10]
    delta_mu_values = [0.1, 0.5, 1, 2, 5, 10, 50, 100]
    delta_s_values = [-2, -1, -0.5, 0.5, 1, 2]
    d_values = [1, 2, 5, 10]
    extreme_values = [1e2, 1e3, 1e4, 1e5]
    
    # 实验1：仅位置参数扰动
    print("运行实验1：仅位置参数扰动...")
    df_pos = experiment_1d_position_only(N_values, delta_mu_values)
    df_pos.to_csv('../experiments/results/experiment_position_only.csv', index=False)
    
    # 实验2：仅尺度参数扰动
    print("运行实验2：仅尺度参数扰动...")
    df_scale = experiment_1d_scale_only(N_values, delta_s_values)
    df_scale.to_csv('../experiments/results/experiment_scale_only.csv', index=False)
    
    # 实验3：同时考虑位置和尺度参数扰动
    print("运行实验3：同时考虑位置和尺度参数扰动...")
    df_both = experiment_1d_both_params(N_values, delta_mu_values[:4], delta_s_values)
    df_both.to_csv('../experiments/results/experiment_both_params.csv', index=False)
    
    # 实验4：多维情况
    print("运行实验4：多维情况下的随机参数实验...")
    df_multi = experiment_multivariate(d_values, N_values[:3], (-5, 5), (-1, 1), num_trials=50)
    df_multi.to_csv('../experiments/results/experiment_multivariate.csv', index=False)
    
    # 实验5：极端情况
    print("运行实验5：极端情况测试...")
    df_extreme = experiment_extreme_cases(N_values, extreme_values)
    df_extreme.to_csv('../experiments/results/experiment_extreme_cases.csv', index=False)
    
    # 实验6：寻找最优常数
    print("运行实验6：寻找最优常数C...")
    df_optimal = experiment_optimal_constant(d_values, N_values, (-5, 5), (-1, 1), num_trials=50)
    df_optimal.to_csv('../experiments/results/experiment_optimal_constant.csv', index=False)
    
    # 汇总结果
    summary = {
        'position_only': df_pos['inequality_holds'].mean(),
        'scale_only': df_scale['inequality_holds'].mean(),
        'both_params': df_both['inequality_holds'].mean(),
        'multivariate': df_multi['inequality_holds'].mean(),
        'extreme_cases': df_extreme['inequality_holds'].mean()
    }
    
    with open('../experiments/results/summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
    
    print("所有实验完成，结果已保存到 ../experiments/results/ 目录")
    
    return {
        'position_only': df_pos,
        'scale_only': df_scale,
        'both_params': df_both,
        'multivariate': df_multi,
        'extreme_cases': df_extreme,
        'optimal_constant': df_optimal
    }

if __name__ == "__main__":
    results = run_all_experiments()
