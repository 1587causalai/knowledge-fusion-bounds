#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KL Divergence Inequality Experimental Results Visualization

This script generates visualization charts and data tables for KL divergence inequality experimental results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from matplotlib.colors import LinearSegmentedColormap

# Set matplotlib style
plt.style.use('default')
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display

# Create results directory
os.makedirs('docs/assets', exist_ok=True)

def load_experiment_results():
    """
    Load experimental results data
    
    Returns:
    Dictionary containing various experimental results
    """
    results = {}
    
    # Load experiment 1: Position parameter perturbations only
    try:
        results['position_only'] = pd.read_csv('experiments/results/experiment_position_only.csv')
    except FileNotFoundError:
        print("Warning: Position parameter perturbation experimental results file not found")
    
    # Load experiment 2: Scale parameter perturbations only
    try:
        results['scale_only'] = pd.read_csv('experiments/results/experiment_scale_only.csv')
    except FileNotFoundError:
        print("Warning: Scale parameter perturbation experimental results file not found")
    
    # Load experiment 3: Both position and scale parameter perturbations
    try:
        results['both_params'] = pd.read_csv('experiments/results/experiment_both_params.csv')
    except FileNotFoundError:
        print("Warning: Dual parameter perturbation experimental results file not found")
    
    # Load experiment 4: Multivariate case
    try:
        results['multivariate'] = pd.read_csv('experiments/results/experiment_multivariate.csv')
    except FileNotFoundError:
        print("Warning: Multivariate experimental results file not found")
    
    # Load experiment 5: Extreme cases
    try:
        results['extreme_cases'] = pd.read_csv('experiments/results/experiment_extreme_cases.csv')
    except FileNotFoundError:
        print("Warning: Extreme cases experimental results file not found")
    
    # Load experiment 6: Optimal constant
    try:
        results['optimal_constant'] = pd.read_csv('experiments/results/experiment_optimal_constant.csv')
    except FileNotFoundError:
        print("Warning: Optimal constant experimental results file not found")
    
    # Load summary results
    try:
        with open('experiments/results/summary.json', 'r') as f:
            results['summary'] = json.load(f)
    except FileNotFoundError:
        print("Warning: Summary results file not found")
    
    return results

def visualize_position_only(df):
    """
    Visualize experimental results for position parameter perturbations only
    
    Parameters:
    df: Experimental results dataframe
    """
    if df is None or df.empty:
        print("No position parameter perturbation experimental data available for visualization")
        return
    
    # Figure 1: KL divergence comparison under same-direction perturbations
    plt.figure(figsize=(10, 6))
    same_dir = df[df['pattern'] == 'same_direction']
    for n in same_dir['N'].unique():
        subset = same_dir[same_dir['N'] == n]
        plt.plot(subset['delta_mu'], subset['kl_fused'], marker='o', label=f'KL_fused (N={n})')
        plt.plot(subset['delta_mu'], subset['right_side'], marker='x', linestyle='--', label=f'N*sum KL (N={n})')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Position Parameter Perturbation (delta_mu)')
    plt.ylabel('KL Divergence')
    plt.title('KL Divergence Comparison under Same-Direction Position Perturbations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('docs/assets/position_only_same_direction.png', dpi=300)
    plt.close()
    
    # Figure 2: KL divergence comparison under opposite-direction perturbations
    plt.figure(figsize=(10, 6))
    opposite_dir = df[df['pattern'] == 'opposite_direction']
    if not opposite_dir.empty:
        for n in opposite_dir['N'].unique():
            subset = opposite_dir[opposite_dir['N'] == n]
            plt.plot(subset['delta_mu'], subset['kl_fused'], marker='o', label=f'KL_fused (N={n})')
            plt.plot(subset['delta_mu'], subset['right_side'], marker='x', linestyle='--', label=f'N*sum KL (N={n})')
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Position Parameter Perturbation (delta_mu)')
        plt.ylabel('KL Divergence')
        plt.title('KL Divergence Comparison under Opposite-Direction Position Perturbations')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('docs/assets/position_only_opposite_direction.png', dpi=300)
        plt.close()
    
    # Figure 3: Inequality satisfaction rate
    plt.figure(figsize=(10, 6))
    result_by_n = df.groupby(['N', 'pattern'])['inequality_holds'].mean().reset_index()
    result_by_n_pivot = result_by_n.pivot(index='N', columns='pattern', values='inequality_holds')
    
    ax = result_by_n_pivot.plot(kind='bar', figsize=(10, 6))
    plt.xlabel('Number of Perturbation Distributions (N)')
    plt.ylabel('Inequality Satisfaction Rate')
    plt.title('Inequality Satisfaction Rate by N and Perturbation Pattern')
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f')
    
    plt.tight_layout()
    plt.savefig('docs/assets/position_only_inequality_holds.png', dpi=300)
    plt.close()
    
    # Figure 4: KL divergence ratio distribution
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='N', y='ratio', hue='pattern', data=df)
    plt.axhline(y=1, color='r', linestyle='--')
    plt.xlabel('Number of Perturbation Distributions (N)')
    plt.ylabel('KL_fused / (N*sum KL) Ratio')
    plt.title('KL Divergence Ratio Distribution by N and Perturbation Pattern')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('docs/assets/position_only_ratio_boxplot.png', dpi=300)
    plt.close()

def visualize_scale_only(df):
    """
    Visualize experimental results for scale parameter perturbations only
    
    Parameters:
    df: Experimental results dataframe
    """
    if df is None or df.empty:
        print("No scale parameter perturbation experimental data available for visualization")
        return
    
    # Figure 1: KL divergence comparison under same-direction perturbations
    plt.figure(figsize=(10, 6))
    same_dir = df[df['pattern'] == 'same_direction']
    for n in same_dir['N'].unique():
        subset = same_dir[same_dir['N'] == n]
        plt.plot(subset['delta_s'], subset['kl_fused'], marker='o', label=f'KL_fused (N={n})')
        plt.plot(subset['delta_s'], subset['right_side'], marker='x', linestyle='--', label=f'N*sum KL (N={n})')
    
    plt.xlabel('Log Scale Parameter Perturbation (delta_s)')
    plt.ylabel('KL Divergence')
    plt.title('KL Divergence Comparison under Same-Direction Scale Perturbations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('docs/assets/scale_only_same_direction.png', dpi=300)
    plt.close()
    
    # Figure 2: KL divergence comparison under opposite-direction perturbations
    plt.figure(figsize=(10, 6))
    opposite_dir = df[df['pattern'] == 'opposite_direction']
    if not opposite_dir.empty:
        for n in opposite_dir['N'].unique():
            subset = opposite_dir[opposite_dir['N'] == n]
            plt.plot(subset['delta_s'], subset['kl_fused'], marker='o', label=f'KL_fused (N={n})')
            plt.plot(subset['delta_s'], subset['right_side'], marker='x', linestyle='--', label=f'N*sum KL (N={n})')
        
        plt.xlabel('Log Scale Parameter Perturbation (delta_s)')
        plt.ylabel('KL Divergence')
        plt.title('KL Divergence Comparison under Opposite-Direction Scale Perturbations')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('docs/assets/scale_only_opposite_direction.png', dpi=300)
        plt.close()
    
    # Figure 3: Inequality satisfaction rate
    plt.figure(figsize=(10, 6))
    result_by_n = df.groupby(['N', 'pattern'])['inequality_holds'].mean().reset_index()
    result_by_n_pivot = result_by_n.pivot(index='N', columns='pattern', values='inequality_holds')
    
    ax = result_by_n_pivot.plot(kind='bar', figsize=(10, 6))
    plt.xlabel('Number of Perturbation Distributions (N)')
    plt.ylabel('Inequality Satisfaction Rate')
    plt.title('Inequality Satisfaction Rate by N and Perturbation Pattern')
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f')
    
    plt.tight_layout()
    plt.savefig('docs/assets/scale_only_inequality_holds.png', dpi=300)
    plt.close()
    
    # Figure 4: KL divergence ratio distribution
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='N', y='ratio', hue='pattern', data=df)
    plt.axhline(y=1, color='r', linestyle='--')
    plt.xlabel('Number of Perturbation Distributions (N)')
    plt.ylabel('KL_fused / (N*sum KL) Ratio')
    plt.title('KL Divergence Ratio Distribution by N and Perturbation Pattern')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('docs/assets/scale_only_ratio_boxplot.png', dpi=300)
    plt.close()

def visualize_both_params(df):
    """
    Visualize experimental results considering both position and scale parameter perturbations
    
    Parameters:
    df: Experimental results dataframe
    """
    if df is None or df.empty:
        print("No dual parameter perturbation experimental data available for visualization")
        return
    
    # Figure 1: Inequality satisfaction heatmap under same-direction perturbations for different delta_mu and delta_s combinations
    plt.figure(figsize=(12, 8))
    for n in df['N'].unique():
        plt.figure(figsize=(10, 8))
        same_dir = df[(df['pattern'] == 'same_direction') & (df['N'] == n)]
        
        if not same_dir.empty:
            # Create pivot table
            pivot = same_dir.pivot_table(
                index='delta_mu', 
                columns='delta_s', 
                values='inequality_holds',
                aggfunc='mean'
            )
            
            # Draw heatmap
            sns.heatmap(pivot, annot=True, cmap='RdYlGn', vmin=0, vmax=1, fmt='.2f')
            plt.title(f'Inequality Satisfaction under Same-Direction Perturbations (N={n})')
            plt.xlabel('Log Scale Parameter Perturbation (delta_s)')
            plt.ylabel('Position Parameter Perturbation (delta_mu)')
            plt.tight_layout()
            plt.savefig(f'docs/assets/both_params_same_direction_N{n}.png', dpi=300)
            plt.close()
    
    # Figure 2: Inequality satisfaction heatmap under opposite-direction perturbations for different delta_mu and delta_s combinations
    for n in df['N'].unique():
        if n == 1:  # No opposite-direction perturbations when N=1
            continue
            
        plt.figure(figsize=(10, 8))
        opposite_dir = df[(df['pattern'] == 'opposite_direction') & (df['N'] == n)]
        
        if not opposite_dir.empty:
            # Create pivot table
            pivot = opposite_dir.pivot_table(
                index='delta_mu', 
                columns='delta_s', 
                values='inequality_holds',
                aggfunc='mean'
            )
            
            # Draw heatmap
            sns.heatmap(pivot, annot=True, cmap='RdYlGn', vmin=0, vmax=1, fmt='.2f')
            plt.title(f'Inequality Satisfaction under Opposite-Direction Perturbations (N={n})')
            plt.xlabel('Log Scale Parameter Perturbation (delta_s)')
            plt.ylabel('Position Parameter Perturbation (delta_mu)')
            plt.tight_layout()
            plt.savefig(f'docs/assets/both_params_opposite_direction_N{n}.png', dpi=300)
            plt.close()
    
    # Figure 3: KL divergence ratio distribution for different N values
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='N', y='ratio', hue='pattern', data=df)
    plt.axhline(y=1, color='r', linestyle='--')
    plt.xlabel('Number of Perturbation Distributions (N)')
    plt.ylabel('KL_fused / (N*sum KL) Ratio')
    plt.title('KL Divergence Ratio Distribution by N and Perturbation Pattern')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('docs/assets/both_params_ratio_boxplot.png', dpi=300)
    plt.close()
    
    # Figure 4: Inequality satisfaction rate for different N values
    plt.figure(figsize=(10, 6))
    result_by_n = df.groupby(['N', 'pattern'])['inequality_holds'].mean().reset_index()
    result_by_n_pivot = result_by_n.pivot(index='N', columns='pattern', values='inequality_holds')
    
    ax = result_by_n_pivot.plot(kind='bar', figsize=(10, 6))
    plt.xlabel('Number of Perturbation Distributions (N)')
    plt.ylabel('Inequality Satisfaction Rate')
    plt.title('Inequality Satisfaction Rate by N and Perturbation Pattern')
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f')
    
    plt.tight_layout()
    plt.savefig('docs/assets/both_params_inequality_holds.png', dpi=300)
    plt.close()

def visualize_multivariate(df):
    """
    Visualize experimental results for multivariate cases
    
    Parameters:
    df: Experimental results dataframe
    """
    if df is None or df.empty:
        print("No multivariate experimental data available for visualization")
        return
    
    # Figure 1: Inequality satisfaction rate for different dimensions d and perturbation numbers N
    plt.figure(figsize=(12, 8))
    result_by_d_n = df.groupby(['d', 'N'])['inequality_holds'].mean().reset_index()
    pivot = result_by_d_n.pivot(index='d', columns='N', values='inequality_holds')
    
    sns.heatmap(pivot, annot=True, cmap='RdYlGn', vmin=0, vmax=1, fmt='.2f')
    plt.title('Inequality Satisfaction Rate by Dimension d and Number of Perturbations N')
    plt.xlabel('Number of Perturbation Distributions (N)')
    plt.ylabel('Dimension (d)')
    plt.tight_layout()
    plt.savefig('docs/assets/multivariate_inequality_holds.png', dpi=300)
    plt.close()
    
    # Figure 2: KL divergence ratio boxplot for different dimensions d and perturbation numbers N
    plt.figure(figsize=(14, 10))
    
    # Create subplot grid
    d_values = sorted(df['d'].unique())
    n_values = sorted(df['N'].unique())
    fig, axes = plt.subplots(len(d_values), 1, figsize=(12, 4*len(d_values)), sharex=True)
    
    for i, d in enumerate(d_values):
        subset = df[df['d'] == d]
        sns.boxplot(x='N', y='ratio', data=subset, ax=axes[i])
        axes[i].axhline(y=1, color='r', linestyle='--')
        axes[i].set_title(f'Dimension d={d}')
        axes[i].set_ylabel('KL_fused / (N*sum KL) Ratio')
        axes[i].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Number of Perturbation Distributions (N)')
    plt.tight_layout()
    plt.savefig('docs/assets/multivariate_ratio_boxplot.png', dpi=300)
    plt.close()
    
    # Figure 3: KL divergence ratio scatter plot
    plt.figure(figsize=(12, 8))
    for d in df['d'].unique():
        for n in df['N'].unique():
            subset = df[(df['d'] == d) & (df['N'] == n)]
            plt.scatter(subset['kl_fused'], subset['right_side'], alpha=0.6, label=f'd={d}, N={n}')
    
    # Add diagonal line
    max_val = max(df['kl_fused'].max(), df['right_side'].max())
    plt.plot([0, max_val], [0, max_val], 'r--')
    
    plt.xlabel('KL_fused')
    plt.ylabel('N*sum KL')
    plt.title('KL Divergence Comparison Scatter Plot')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('docs/assets/multivariate_scatter.png', dpi=300)
    plt.close()

def visualize_extreme_cases(df):
    """
    Visualize experimental results for extreme cases
    
    Parameters:
    df: Experimental results dataframe
    """
    if df is None or df.empty:
        print("No extreme cases experimental data available for visualization")
        return
    
    # Figure 1: KL divergence comparison under extreme values (position parameters)
    plt.figure(figsize=(12, 8))
    position = df[df['param_type'] == 'position']
    
    for n in position['N'].unique():
        subset = position[position['N'] == n]
        plt.plot(subset['extreme_value'], subset['kl_fused'], marker='o', label=f'KL_fused (N={n})')
        plt.plot(subset['extreme_value'], subset['right_side'], marker='x', linestyle='--', label=f'N*sum KL (N={n})')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Extreme Position Parameter Value')
    plt.ylabel('KL Divergence')
    plt.title('KL Divergence Comparison under Extreme Position Parameter Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('docs/assets/extreme_cases_position.png', dpi=300)
    plt.close()
    
    # Figure 2: KL divergence comparison under extreme values (scale parameters)
    plt.figure(figsize=(12, 8))
    scale = df[df['param_type'] == 'scale']
    
    for n in scale['N'].unique():
        subset = scale[scale['N'] == n]
        plt.plot(subset['extreme_value'], subset['kl_fused'], marker='o', label=f'KL_fused (N={n})')
        plt.plot(subset['extreme_value'], subset['right_side'], marker='x', linestyle='--', label=f'N*sum KL (N={n})')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Extreme Scale Parameter Value')
    plt.ylabel('KL Divergence')
    plt.title('KL Divergence Comparison under Extreme Scale Parameter Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('docs/assets/extreme_cases_scale.png', dpi=300)
    plt.close()
    
    # Figure 3: Inequality satisfaction status
    plt.figure(figsize=(12, 8))
    result_by_type = df.groupby(['N', 'param_type'])['inequality_holds'].mean().reset_index()
    result_pivot = result_by_type.pivot(index='N', columns='param_type', values='inequality_holds')
    
    ax = result_pivot.plot(kind='bar', figsize=(10, 6))
    plt.xlabel('Number of Perturbation Distributions (N)')
    plt.ylabel('Inequality Satisfaction Rate')
    plt.title('Inequality Satisfaction Rate by N and Parameter Type')
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f')
    
    plt.tight_layout()
    plt.savefig('docs/assets/extreme_cases_inequality_holds.png', dpi=300)
    plt.close()
    
    # Figure 4: KL divergence ratio
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='N', y='ratio', hue='param_type', data=df)
    plt.axhline(y=1, color='r', linestyle='--')
    plt.xlabel('Number of Perturbation Distributions (N)')
    plt.ylabel('KL_fused / (N*sum KL) Ratio')
    plt.title('KL Divergence Ratio Distribution by N and Parameter Type')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('docs/assets/extreme_cases_ratio_boxplot.png', dpi=300)
    plt.close()

def visualize_optimal_constant(df):
    """
    Visualize experimental results for optimal constant C
    
    Parameters:
    df: Experimental results dataframe
    """
    if df is None or df.empty:
        print("No optimal constant experimental data available for visualization")
        return
    
    # Figure 1: Maximum ratio for different dimensions d and perturbation numbers N
    plt.figure(figsize=(12, 8))
    pivot_max = df.pivot(index='d', columns='N', values='max_ratio')
    
    sns.heatmap(pivot_max, annot=True, cmap='YlOrRd', fmt='.2f')
    plt.title('Maximum Ratio by Dimension d and Number of Perturbations N')
    plt.xlabel('Number of Perturbation Distributions (N)')
    plt.ylabel('Dimension (d)')
    plt.tight_layout()
    plt.savefig('docs/assets/optimal_constant_max_ratio.png', dpi=300)
    plt.close()
    
    # Figure 2: Average ratio for different dimensions d and perturbation numbers N
    plt.figure(figsize=(12, 8))
    pivot_mean = df.pivot(index='d', columns='N', values='mean_ratio')
    
    sns.heatmap(pivot_mean, annot=True, cmap='YlOrRd', fmt='.2f')
    plt.title('Average Ratio by Dimension d and Number of Perturbations N')
    plt.xlabel('Number of Perturbation Distributions (N)')
    plt.ylabel('Dimension (d)')
    plt.tight_layout()
    plt.savefig('docs/assets/optimal_constant_mean_ratio.png', dpi=300)
    plt.close()
    
    # Figure 3: 95th percentile ratio for different dimensions d and perturbation numbers N
    plt.figure(figsize=(12, 8))
    pivot_p95 = df.pivot(index='d', columns='N', values='p95_ratio')
    
    sns.heatmap(pivot_p95, annot=True, cmap='YlOrRd', fmt='.2f')
    plt.title('95th Percentile Ratio by Dimension d and Number of Perturbations N')
    plt.xlabel('Number of Perturbation Distributions (N)')
    plt.ylabel('Dimension (d)')
    plt.tight_layout()
    plt.savefig('docs/assets/optimal_constant_p95_ratio.png', dpi=300)
    plt.close()
    
    # Figure 4: Relationship between ratio and N
    plt.figure(figsize=(12, 8))
    
    # For each dimension d, plot different statistics vs N
    for d in df['d'].unique():
        subset = df[df['d'] == d]
        plt.plot(subset['N'], subset['max_ratio'], marker='o', label=f'Max ratio (d={d})')
        plt.plot(subset['N'], subset['p95_ratio'], marker='s', linestyle='--', label=f'95th percentile (d={d})')
    
    plt.xlabel('Number of Perturbation Distributions (N)')
    plt.ylabel('KL_fused / sum KL Ratio')
    plt.title('Relationship between Ratio and Number of Perturbations N for Different Dimensions d')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('docs/assets/optimal_constant_ratio_vs_N.png', dpi=300)
    plt.close()
    
    # Figure 5: Optimal constant C estimation
    plt.figure(figsize=(12, 8))
    
    # Calculate maximum ratio for each N value (across all dimensions)
    max_by_n = df.groupby('N')['max_ratio'].max().reset_index()
    p99_by_n = df.groupby('N')['p99_ratio'].max().reset_index()
    
    plt.plot(max_by_n['N'], max_by_n['max_ratio'], marker='o', label='Maximum ratio')
    plt.plot(p99_by_n['N'], p99_by_n['p99_ratio'], marker='s', label='99th percentile')
    
    # Fit curve C = N^a
    from scipy.optimize import curve_fit
    
    def power_law(x, a):
        return x ** a
    
    # Fit maximum ratio
    popt_max, _ = curve_fit(power_law, max_by_n['N'], max_by_n['max_ratio'])
    x_fit = np.linspace(min(max_by_n['N']), max(max_by_n['N']), 100)
    y_fit_max = power_law(x_fit, popt_max[0])
    
    # Fit 99th percentile
    popt_p99, _ = curve_fit(power_law, p99_by_n['N'], p99_by_n['p99_ratio'])
    y_fit_p99 = power_law(x_fit, popt_p99[0])
    
    plt.plot(x_fit, y_fit_max, 'r-', label=f'Fitted curve: C = N^{popt_max[0]:.3f}')
    plt.plot(x_fit, y_fit_p99, 'g-', label=f'Fitted curve: C = N^{popt_p99[0]:.3f}')
    
    plt.xlabel('Number of Perturbation Distributions (N)')
    plt.ylabel('Optimal Constant C')
    plt.title('Relationship between Optimal Constant C and Number of Perturbations N')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('docs/assets/optimal_constant_estimation.png', dpi=300)
    plt.close()
    
    # Save fitting results
    fit_results = {
        'max_ratio_exponent': float(popt_max[0]),
        'p99_ratio_exponent': float(popt_p99[0]),
        'suggested_formula': f"C ≈ N^{max(popt_max[0], popt_p99[0]):.3f}"
    }
    
    with open('experiments/results/optimal_constant_fit.json', 'w') as f:
        json.dump(fit_results, f, indent=4)

def create_summary_tables():
    """
    Create summary tables for experimental results
    """
    # Load all experimental results
    results = load_experiment_results()
    
    # Table 1: Inequality satisfaction rate summary
    summary_data = []
    
    # Position parameter perturbations
    if 'position_only' in results:
        df = results['position_only']
        for n in sorted(df['N'].unique()):
            for pattern in sorted(df['pattern'].unique()):
                subset = df[(df['N'] == n) & (df['pattern'] == pattern)]
                summary_data.append({
                    'Experiment Type': 'Position Parameter Perturbations Only',
                    'Number of Perturbations (N)': n,
                    'Perturbation Pattern': pattern,
                    'Inequality Satisfaction Rate': subset['inequality_holds'].mean(),
                    'Sample Size': len(subset)
                })
    
    # Scale parameter perturbations
    if 'scale_only' in results:
        df = results['scale_only']
        for n in sorted(df['N'].unique()):
            for pattern in sorted(df['pattern'].unique()):
                subset = df[(df['N'] == n) & (df['pattern'] == pattern)]
                summary_data.append({
                    'Experiment Type': 'Scale Parameter Perturbations Only',
                    'Number of Perturbations (N)': n,
                    'Perturbation Pattern': pattern,
                    'Inequality Satisfaction Rate': subset['inequality_holds'].mean(),
                    'Sample Size': len(subset)
                })
    
    # Dual parameter perturbations
    if 'both_params' in results:
        df = results['both_params']
        for n in sorted(df['N'].unique()):
            for pattern in sorted(df['pattern'].unique()):
                subset = df[(df['N'] == n) & (df['pattern'] == pattern)]
                summary_data.append({
                    'Experiment Type': 'Dual Parameter Perturbations',
                    'Number of Perturbations (N)': n,
                    'Perturbation Pattern': pattern,
                    'Inequality Satisfaction Rate': subset['inequality_holds'].mean(),
                    'Sample Size': len(subset)
                })
    
    # Multivariate cases
    if 'multivariate' in results:
        df = results['multivariate']
        for d in sorted(df['d'].unique()):
            for n in sorted(df['N'].unique()):
                subset = df[(df['d'] == d) & (df['N'] == n)]
                summary_data.append({
                    'Experiment Type': f'Multivariate (d={d})',
                    'Number of Perturbations (N)': n,
                    'Perturbation Pattern': 'random',
                    'Inequality Satisfaction Rate': subset['inequality_holds'].mean(),
                    'Sample Size': len(subset)
                })
    
    # Extreme cases
    if 'extreme_cases' in results:
        df = results['extreme_cases']
        for n in sorted(df['N'].unique()):
            for param_type in sorted(df['param_type'].unique()):
                subset = df[(df['N'] == n) & (df['param_type'] == param_type)]
                summary_data.append({
                    'Experiment Type': f'Extreme Cases ({param_type})',
                    'Number of Perturbations (N)': n,
                    'Perturbation Pattern': 'extreme',
                    'Inequality Satisfaction Rate': subset['inequality_holds'].mean(),
                    'Sample Size': len(subset)
                })
    
    # Create summary table
    summary_df = pd.DataFrame(summary_data)
    if not summary_df.empty:
        summary_df.to_csv('experiments/results/inequality_holds_summary.csv', index=False)
        
        # Create HTML table
        html_table = summary_df.to_html(index=False)
        with open('docs/assets/inequality_holds_summary.html', 'w') as f:
            f.write(html_table)
    
    # Table 2: Optimal constant C estimation
    if 'optimal_constant' in results:
        df = results['optimal_constant']
        
        # Group by N, calculate maximum ratio across all dimensions
        optimal_by_n = df.groupby('N').agg({
            'max_ratio': 'max',
            'p95_ratio': 'max',
            'p99_ratio': 'max',
            'mean_ratio': 'mean'
        }).reset_index()
        
        optimal_by_n.to_csv('experiments/results/optimal_constant_by_N.csv', index=False)
        
        # Create HTML table
        html_table = optimal_by_n.to_html(index=False)
        with open('docs/assets/optimal_constant_by_N.html', 'w') as f:
            f.write(html_table)
    
    # Table 3: Counterexample cases summary
    counterexamples = []
    
    # Check counterexamples in all experiments
    for exp_type, df in results.items():
        if exp_type not in ['summary', 'optimal_constant'] and isinstance(df, pd.DataFrame) and not df.empty:
            # Find cases where inequality doesn't hold
            counter = df[df['inequality_holds'] == False]
            if not counter.empty:
                # Select top counterexamples with highest ratios
                top_counters = counter.nlargest(min(5, len(counter)), 'ratio')
                
                for _, row in top_counters.iterrows():
                    example = {'Experiment Type': exp_type}
                    for col in row.index:
                        if col not in ['inequality_holds']:
                            example[col] = row[col]
                    counterexamples.append(example)
    
    # Create counterexample table
    if counterexamples:
        counter_df = pd.DataFrame(counterexamples)
        counter_df.to_csv('experiments/results/counterexamples.csv', index=False)
        
        # Create HTML table
        html_table = counter_df.to_html(index=False)
        with open('docs/assets/counterexamples.html', 'w') as f:
            f.write(html_table)

def run_all_visualizations():
    """
    Run all visualization functions
    """
    print("Starting to generate visualization charts and data tables...")
    
    # Load experimental results
    results = load_experiment_results()
    
    # Run visualizations for each experiment
    visualize_position_only(results.get('position_only'))
    visualize_scale_only(results.get('scale_only'))
    visualize_both_params(results.get('both_params'))
    visualize_multivariate(results.get('multivariate'))
    visualize_extreme_cases(results.get('extreme_cases'))
    visualize_optimal_constant(results.get('optimal_constant'))
    
    # Create summary tables
    create_summary_tables()
    
    print("All visualization charts and data tables have been generated successfully")

if __name__ == "__main__":
    run_all_visualizations()
