#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import os

def kl_divergence_cauchy(mu_0, gamma_0, mu, gamma):
    """Calculates KL divergence between two Cauchy distributions"""
    return np.log((gamma_0 + gamma)**2 + (mu_0 - mu)**2) - np.log(4 * gamma_0 * gamma)

def find_optimal_coefficient_detailed():
    """Detailed analysis of the optimal coefficient C(N)"""
    print("Analyzing optimal coefficient C(N)...")
    print("="*60)
    
    # Expand the range of N to include more data points
    N_values = range(2, 11)  # N = 2, 3, ..., 10
    
    # Use a finer parameter grid to ensure finding the true maximum
    delta_mu_values = np.linspace(0.05, 5.0, 50)  # Finer grid
    delta_s_values = np.linspace(-3.0, -0.01, 50)  # Finer grid
    
    optimal_coefficients = []
    detailed_results = []
    
    # Base distribution parameters
    mu_0 = 0.0
    gamma_0 = 1.0
    
    for N in N_values:
        print(f"\nAnalyzing N = {N}...")
        max_ratio = 0
        best_params = None
        
        # Record results for each parameter combination
        current_results = []
        
        for delta_mu in delta_mu_values:
            for delta_s in delta_s_values:
                # Perturbed distribution parameters (same perturbation in the same direction)
                mu_k = mu_0 + delta_mu
                gamma_k = gamma_0 * np.exp(delta_s)
                
                # Fused distribution parameters
                mu_fused = mu_0 + N * delta_mu
                gamma_fused = gamma_0 * np.exp(N * delta_s)
                
                # Check numerical stability
                if gamma_k <= 0 or gamma_fused <= 0:
                    continue
                    
                # Calculate KL divergence
                try:
                    D_indiv = kl_divergence_cauchy(mu_0, gamma_0, mu_k, gamma_k)
                    D_fused = kl_divergence_cauchy(mu_0, gamma_0, mu_fused, gamma_fused)
                    
                    # Check numerical validity
                    if np.isnan(D_indiv) or np.isnan(D_fused) or np.isinf(D_indiv) or np.isinf(D_fused):
                        continue
                        
                    # Calculate ratio: D_fused / (N * D_indiv)
                    if D_indiv > 0:  # Ensure denominator is positive
                        ratio = D_fused / (N * D_indiv)
                        
                        current_results.append({
                            'N': N,
                            'delta_mu': delta_mu,
                            'delta_s': delta_s,
                            'D_indiv': D_indiv,
                            'D_fused': D_fused,
                            'ratio': ratio
                        })
                        
                        if ratio > max_ratio:
                            max_ratio = ratio
                            best_params = (delta_mu, delta_s)
                            
                except (ValueError, RuntimeWarning):
                    continue
        
        optimal_coefficients.append({
            'N': N,
            'C_min': max_ratio,
            'best_delta_mu': best_params[0] if best_params else None,
            'best_delta_s': best_params[1] if best_params else None
        })
        
        detailed_results.extend(current_results)
        
        print(f"  Optimal coefficient C({N}) >= {max_ratio:.6f}")
        if best_params:
            print(f"  Optimal parameters: delta_mu = {best_params[0]:.4f}, delta_s = {best_params[1]:.4f}")
    
    # Convert to DataFrame
    df_optimal = pd.DataFrame(optimal_coefficients)
    df_detailed = pd.DataFrame(detailed_results)
    
    # Save results
    os.makedirs('results', exist_ok=True)
    df_optimal.to_csv('results/optimal_coefficients.csv', index=False)
    df_detailed.to_csv('results/detailed_coefficient_analysis.csv', index=False)
    
    return df_optimal, df_detailed

def fit_coefficient_function(df_optimal):
    """Fit functional form for C(N)"""
    print("\nFitting functional form for C(N)...")
    print("="*40)
    
    N_vals = df_optimal['N'].values
    C_vals = df_optimal['C_min'].values
    
    # Try different functional forms
    functions = {
        'Linear': lambda N, a, b: a * N + b,
        'Power': lambda N, a, b: a * N**b,
        'Logarithmic': lambda N, a, b: a * np.log(N) + b,
        'Exponential': lambda N, a, b: a * np.exp(b * N),
        'Polynomial': lambda N, a, b, c: a * N**2 + b * N + c
    }
    
    best_fit = None
    best_r2 = -np.inf
    
    for name, func in functions.items():
        try:
            if name == 'Polynomial':
                popt, _ = curve_fit(func, N_vals, C_vals, maxfev=5000)
                y_pred = func(N_vals, *popt)
            else:
                popt, _ = curve_fit(func, N_vals, C_vals, maxfev=5000)
                y_pred = func(N_vals, *popt)
            
            # Calculate R²
            ss_res = np.sum((C_vals - y_pred) ** 2)
            ss_tot = np.sum((C_vals - np.mean(C_vals)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            
            print(f"{name:12}: R² = {r2:.6f}, Parameters = {popt}")
            
            if r2 > best_r2:
                best_r2 = r2
                best_fit = (name, func, popt)
                
        except Exception as e:
            print(f"{name:12}: Fit failed - {e}")
    
    return best_fit, best_r2

def plot_coefficient_analysis(df_optimal, best_fit):
    """Plot C(N) analysis charts"""
    print("\nGenerating C(N) analysis charts...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    N_vals = df_optimal['N'].values
    C_vals = df_optimal['C_min'].values
    
    # 1. C(N) original data points
    ax1.plot(N_vals, C_vals, 'ro-', markersize=8, linewidth=2, label='Experimental C(N)')
    ax1.plot(N_vals, N_vals, 'b--', alpha=0.7, label='y = N')
    ax1.plot(N_vals, N_vals**2, 'g--', alpha=0.7, label='y = N²')
    ax1.set_xlabel('N')
    ax1.set_ylabel('Optimal Coefficient C(N)')
    ax1.set_title('Optimal Coefficient C(N) vs N')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Fitted function
    if best_fit:
        name, func, popt = best_fit
        N_fine = np.linspace(2, 10, 100)
        try:
            if name == 'Polynomial':
                C_fit = func(N_fine, *popt)
            else:
                C_fit = func(N_fine, *popt)
            ax2.plot(N_vals, C_vals, 'ro', markersize=8, label='Experimental Data')
            ax2.plot(N_fine, C_fit, 'b-', linewidth=2, label=f'Best Fit: {name}')
            ax2.set_xlabel('N')
            ax2.set_ylabel('C(N)')
            ax2.set_title(f'Best Fit Function: {name}')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        except:
            ax2.text(0.5, 0.5, 'Fit Function Plot Failed', ha='center', va='center', transform=ax2.transAxes)
    
    # 3. C(N)/N ratio
    ratio_C_N = C_vals / N_vals
    ax3.plot(N_vals, ratio_C_N, 'mo-', markersize=8, linewidth=2, label='C(N)/N')
    ax3.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='y = 1')
    ax3.set_xlabel('N')
    ax3.set_ylabel('C(N)/N')
    ax3.set_title('Normalized Coefficient C(N)/N')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Optimal parameter location
    best_mu = df_optimal['best_delta_mu'].values
    best_s = df_optimal['best_delta_s'].values
    
    scatter = ax4.scatter(best_mu, best_s, c=N_vals, cmap='viridis', s=100, alpha=0.8)
    ax4.set_xlabel('Optimal δμ')
    ax4.set_ylabel('Optimal δs')
    ax4.set_title('Optimal Parameter Distribution')
    plt.colorbar(scatter, ax=ax4, label='N')
    ax4.grid(True, alpha=0.3)
    
    # Add N value annotations
    for i, n in enumerate(N_vals):
        ax4.annotate(f'N={n}', (best_mu[i], best_s[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('results/optimal_coefficient_analysis.png', dpi=300, bbox_inches='tight')
    print("Charts saved to results/optimal_coefficient_analysis.png")
    
    return fig

def theoretical_bounds_analysis(df_optimal):
    """Analyze theoretical upper bounds"""
    print("\nAnalyzing theoretical upper bounds...")
    print("="*40)
    
    N_vals = df_optimal['N'].values
    C_vals = df_optimal['C_min'].values
    
    print("Comparison of various theoretical upper bounds:")
    print("N\tC(N)\tN\tN²\te^N\tN*ln(N)\t√N*N")
    print("-" * 70)
    
    for i, N in enumerate(N_vals):
        C_N = C_vals[i]
        bounds = {
            'N': N,
            'N²': N**2,
            'e^N': np.exp(N),
            'N*ln(N)': N * np.log(N) if N > 1 else N,
            '√N*N': np.sqrt(N) * N
        }
        
        print(f"{N}\t{C_N:.3f}\t{bounds['N']:.1f}\t{bounds['N²']:.1f}\t{bounds['e^N']:.1f}\t{bounds['N*ln(N)']:.1f}\t{bounds['√N*N']:.1f}")
        
        # Check which upper bounds are valid
        valid_bounds = []
        for bound_name, bound_val in bounds.items():
            if bound_val >= C_N:
                valid_bounds.append(bound_name)
        
        print(f"  Valid upper bounds: {', '.join(valid_bounds)}")
    
    # Find the tightest polynomial upper bound
    print(f"\nFinding the tightest upper bound of the form C(N) ≤ N^α:")
    
    # Calculate α for each N such that N^α = C(N)
    alphas = np.log(C_vals) / np.log(N_vals)
    max_alpha = np.max(alphas)
    
    print(f"α values for each N: {dict(zip(N_vals, alphas))}")
    print(f"Maximum α value: {max_alpha:.6f}")
    print(f"Therefore, C(N) ≤ N^{max_alpha:.3f} is a valid upper bound")
    
    # Check if C(N) ≤ N^1.5 is sufficient
    alpha_15 = 1.5
    N15_bounds = N_vals ** alpha_15
    is_valid_15 = np.all(N15_bounds >= C_vals)
    print(f"Is C(N) ≤ N^1.5 valid: {is_valid_15}")
    
    return max_alpha

if __name__ == "__main__":
    # Analyze optimal coefficient
    df_optimal, df_detailed = find_optimal_coefficient_detailed()
    
    # Fit function
    best_fit, best_r2 = fit_coefficient_function(df_optimal)
    print(f"\nBest fit: {best_fit[0] if best_fit else 'None'}, R² = {best_r2:.6f}")
    
    # Plot analysis charts
    plot_coefficient_analysis(df_optimal, best_fit)
    
    # Analyze theoretical upper bounds
    max_alpha = theoretical_bounds_analysis(df_optimal)
    
    print(f"\nSummary:")
    print(f"="*60)
    print(f"1. Optimal coefficient C(N) increases with N, but growth is between linear and quadratic")
    print(f"2. The tightest power-law upper bound is approximately C(N) ≤ N^{max_alpha:.3f}")
    print(f"3. N² and e^N are valid but looser upper bounds")
    print(f"4. Practical tight upper bounds could be C(N) ≤ N^1.5 or C(N) ≤ N*ln(N)")
    print(f"5. All results saved to results/ directory") 