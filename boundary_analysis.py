#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os

def kl_divergence_cauchy(mu_0, gamma_0, mu, gamma):
    return np.log((gamma_0 + gamma)**2 + (mu_0 - mu)**2) - np.log(4 * gamma_0 * gamma)

def find_critical_boundary():
    """Numerically find points (delta_s) on the boundary for given delta_mu values where the ratio is approximately 1, for N=2."""
    print("Numerically finding critical boundary points (N=2)...")
    print("="*60)
    
    N = 2
    # Widen delta_mu range, use logspace for better coverage of small and large values
    # Values from ~0.1 to ~10.0. Include points near 2*sqrt(2) and 3.769
    delta_mu_values = sorted(list(set(
        np.logspace(-1, 0.7, 15).tolist() + 
        [2*np.sqrt(2) - 0.1, 2*np.sqrt(2), 2*np.sqrt(2) + 0.1, 3.769 - 0.1, 3.769, 3.769 + 0.1]
    )))
    delta_mu_values = [dm for dm in delta_mu_values if dm > 0.01] # Ensure positive delta_mu

    critical_delta_s_list = []
    
    for delta_mu in delta_mu_values:
        # Search for delta_s where ratio transitions from <=1 to >1
        # Expanded search range for delta_s
        delta_s_low = -7.0  # Start lower for delta_s
        delta_s_high = 0.5   # Allow slightly positive delta_s
        
        # Store the best delta_s found that satisfies ratio <= 1.0
        # and the best delta_s that satisfies ratio > 1.0
        s_leq_1 = None
        s_gt_1 = None

        for _ in range(100):  # Increased iterations for precision
            delta_s_mid = (delta_s_low + delta_s_high) / 2
            if delta_s_mid == delta_s_low or delta_s_mid == delta_s_high: # Avoid infinite loop if precision limit reached
                break
            
            # Calculate k(2x, u^2) and k(x,u)^2
            x = delta_mu
            u_mid = np.exp(delta_s_mid)
            
            # Term for fused distribution: k(N*x, u^N)
            val_fused = kl_divergence_cauchy(0, 1.0, N * x, np.exp(N * delta_s_mid)) # This is log(k(Nx,u^N))
            # Term for individual sum: N * k(x,u)  -- actually N * log(k(x,u))
            val_individual_sum = N * kl_divergence_cauchy(0, 1.0, x, u_mid) # This is N*log(k(x,u))
            
            # We are checking if D_fused > D_individual_sum
            # which is log(k(Nx,u^N)) > N*log(k(x,u))
            # This is equivalent to k(Nx,u^N) > k(x,u)^N if log is base > 1 (it is)
            # The ratio in previous scripts was D_fused / D_individual_sum.
            # Here, more directly, we check the condition of the inequality itself.

            if np.isinf(val_fused) or np.isinf(val_individual_sum) or np.isnan(val_fused) or np.isnan(val_individual_sum):
                # Problematic calculation, try to adjust search space if possible
                # This might happen if u_mid is too small or too large leading to overflow/underflow in exp or log
                if abs(delta_s_mid - delta_s_low) < abs(delta_s_mid - delta_s_high):
                     delta_s_high = delta_s_mid # Problem likely at lower end
                else:
                     delta_s_low = delta_s_mid # Problem likely at higher end
                continue

            if val_fused > val_individual_sum:  # Counterexample region ( D_fused > N * Sum D_indiv )
                s_gt_1 = delta_s_mid
                delta_s_high = delta_s_mid # Try to find a point where inequality holds (move to less negative delta_s)
            else:  # Inequality holds or on boundary
                s_leq_1 = delta_s_mid
                delta_s_low = delta_s_mid # Try to find a counterexample (move to more negative delta_s)
        
        # The critical boundary is between s_leq_1 and s_gt_1
        # We want the point where it transitions from holding to not holding
        # So, s_low should represent the point where it holds, s_high where it doesn't.
        # The reported critical_delta_s should be where it *starts* to fail. That is delta_s_high from the loop.
        # Or, more robustly, if s_leq_1 and s_gt_1 are found, the boundary is between them.
        # The loop finds delta_s_low as the largest s where it holds, and delta_s_high as smallest s where it fails.
        critical_s = (delta_s_low + delta_s_high) / 2.0 # Midpoint of final interval
        
        # If s_leq_1 is None, means all tested points were counterexamples up to delta_s_high (0.5)
        # If s_gt_1 is None, means inequality held for all points down to delta_s_low (-7.0)
        if s_leq_1 is None and s_gt_1 is not None: # All counterexamples in range, boundary is at or above delta_s_high
            critical_s = s_gt_1 # or delta_s_high, which is what it converges to
        elif s_gt_1 is None and s_leq_1 is not None: # All hold in range, boundary is at or below delta_s_low
            critical_s = s_leq_1 # or delta_s_low
        elif s_gt_1 is None and s_leq_1 is None: # Should not happen if range is wide enough
             critical_s = np.nan # Undetermined

        critical_delta_s_list.append(critical_s)
        if not np.isnan(critical_s):
            print(f"For delta_mu = {delta_mu:.4f}: critical delta_s ≈ {critical_s:.6f}")
        else:
            print(f"For delta_mu = {delta_mu:.4f}: critical delta_s could not be determined within search range.")
    
    # Filter out NaNs for plotting and fitting if any
    valid_indices = [i for i, s in enumerate(critical_delta_s_list) if not np.isnan(s)]
    final_delta_mu_values = [delta_mu_values[i] for i in valid_indices]
    final_critical_delta_s = [critical_delta_s_list[i] for i in valid_indices]

    return final_delta_mu_values, final_critical_delta_s

def analyze_boundary_function():
    """Analyze and plot the boundary function."""
    print("\nAnalyzing and plotting the boundary function...")
    print("="*40)
    
    delta_mu_vals, critical_delta_s_vals = find_critical_boundary()

    if not delta_mu_vals: # Check if find_critical_boundary returned empty lists
        print("No valid critical boundary points found. Skipping fitting and plotting.")
        return [], []
    
    # Attempt to fit a function to the boundary points
    # This is mostly for exploratory purposes as the true function is complex
    from scipy.optimize import curve_fit
    
    def boundary_func_simple(delta_mu, a, b, c, d):
        # A more flexible empirical fit, e.g., a rational function or log based
        # For now, use a simple polynomial, or log relationship
        # Example: c * np.log(delta_mu - a) + b (if delta_mu > a)
        # return a * np.log(delta_mu + b) + c # Simple log fit
        return a * delta_mu**b + c # Power law
        # return a / (delta_mu -b) + c # Rational

    # Initial guess for power law
    # params, pcov = curve_fit(boundary_func_simple, delta_mu_vals, critical_delta_s_vals, p0=[-1, -0.5, 0], maxfev=5000)
    # a,b,c = params
    # print(f"Fitted Power Law: delta_s_critical ≈ {a:.4f} * delta_mu^{b:.4f} + {c:.4f}")
    # Fit is often poor, let's just plot the numerical points primarily.

    # Theoretical points for N=2 boundary: k(2x,u^2) = k(x,u)^2
    # Point 1: (delta_mu/gamma_0 = 2sqrt(2), delta_s = 0)
    theory_pt1_dm = 2*np.sqrt(2)
    theory_pt1_ds = 0
    # Asymptote: delta_mu/gamma_0 approaches sqrt(14.211) as delta_s -> -inf
    asymptote_dm = np.sqrt(7 + np.sqrt(52)) # approx 3.769

    # Visualization
    plt.figure(figsize=(14, 7))
    
    # Plot numerically found boundary points
    plt.subplot(1, 2, 1)
    plt.plot(delta_mu_vals, critical_delta_s_vals, 'ro-', label='Critical Boundary (Numerical, N=2)', markersize=5)
    # plt.plot(delta_mu_fine, boundary_func_simple(delta_mu_fine, a,b,c), 'b--', label=f'Fitted Curve')
    
    # Add theoretical points/lines
    plt.axhline(0, color='grey', linestyle='--', linewidth=0.7, label='delta_s = 0')
    plt.scatter([theory_pt1_dm], [theory_pt1_ds], color='blue', s=100, zorder=5, label=f'Theory: (2√2, 0) ≈ ({theory_pt1_dm:.3f}, 0)')
    plt.axvline(asymptote_dm, color='green', linestyle=':', linewidth=1, label=f'Theory: Asymptote δμ ≈ {asymptote_dm:.3f} as δs → -∞')

    plt.xlabel('δμ/γ₀ (Normalized Location Perturbation)')
    plt.ylabel('δs (Log-Scale Perturbation)')
    plt.title('Counterexample Boundary for D_KL(P₀||P_fused) ≤ 2 Σ D_KL(P₀||Pᵢ)')
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.ylim(min(critical_delta_s_vals + [-2.5, 0.6]) , max(critical_delta_s_vals+[-2.5, 0.6])) # Adjust ylim dynamically
    plt.xlim(0, max(delta_mu_vals + [asymptote_dm + 0.5]))

    # Parameter space heatmap for N=2
    plt.subplot(1, 2, 2)
    # Use wider ranges for heatmap based on new insights
    delta_mu_grid = np.logspace(-1, np.log10(max(delta_mu_vals + [asymptote_dm +1, 5.0])), 60) 
    delta_s_grid = np.linspace(min(critical_delta_s_vals+[-5.0]), max(critical_delta_s_vals+[0.5]), 60)
    
    ratio_grid = np.zeros((len(delta_s_grid), len(delta_mu_grid)))
    N_heat = 2

    for i, ds_val in enumerate(delta_s_grid):
        for j, dm_val in enumerate(delta_mu_grid):
            if dm_val <= 0: continue
            u_val = np.exp(ds_val)
            
            # D_fused = log k(N*dm, u^N)
            log_k_fused = kl_divergence_cauchy(0, 1.0, N_heat * dm_val, np.exp(N_heat * ds_val))
            # N * D_indiv = N * log k(dm, u)
            log_k_indiv_sum = N_heat * kl_divergence_cauchy(0, 1.0, dm_val, u_val)
            
            # Ratio of D_KLs: D_fused / (N * D_indiv)
            if log_k_indiv_sum != 0 and not (np.isinf(log_k_fused) or np.isinf(log_k_indiv_sum) or np.isnan(log_k_fused) or np.isnan(log_k_indiv_sum)):
                # To avoid issues with log(0) or small numbers, check if log_k_fused > log_k_indiv_sum for counterexample
                # The ratio is of D_KL values, not k values.
                # Ratio for plotting: exp(log_k_fused - log_k_indiv_sum) if we define ratio as k_fused_eff / k_indiv_sum_eff
                # Or simply, if log_k_fused > log_k_indiv_sum, it's a counterexample (ratio > 1 in KL terms)
                # For plotting actual ratio of D_KLs, handle signs and zeros carefully.
                # Let's plot 1 if counterexample, 0 if not, for clarity, or ratio of k values if meaningful
                # We want to plot D_fused / (N*D_indiv) = log(k(Nx,u^N)) / (N*log(k(x,u)))
                # This ratio can be tricky if denominators are zero or logs are negative (k < 1 is not possible here)
                # Let's plot a heatmap of (log_k_fused - log_k_indiv_sum) -> positive means counterexample
                diff_logs = log_k_fused - log_k_indiv_sum
                ratio_grid[i, j] = diff_logs # Positive means counterexample
            else:
                ratio_grid[i, j] = -1 # Indicate problem or non-counterexample by default
    
    # Plot difference of logs. Contour at 0 indicates the boundary.
    vmax = np.percentile(ratio_grid[ratio_grid > -1], 95) # Robust vmax
    vmin = np.percentile(ratio_grid[ratio_grid > -1], 5)  # Robust vmin
    im = plt.imshow(ratio_grid, extent=[delta_mu_grid[0], delta_mu_grid[-1], 
                                       delta_s_grid[0], delta_s_grid[-1]], 
                   aspect='auto', origin='lower', cmap='RdYlBu_r', vmin=min(0,vmin)-0.1, vmax=max(0,vmax)+0.1)
    plt.colorbar(im, label='log(D_fused) - log(N·ΣD_indiv)') # Label reflects what is plotted
    
    # Plot numerically found boundary again for comparison
    plt.plot(delta_mu_vals, critical_delta_s_vals, 'k--', linewidth=2, label='Numerical Boundary (points from left plot)')
    # Plot contour at 0 for the heatmap data
    plt.contour(delta_mu_grid, delta_s_grid, ratio_grid, levels=[0.0], colors='magenta', linewidths=2, linestyles='dotted')
    
    plt.scatter([theory_pt1_dm], [theory_pt1_ds], color='cyan', s=100, zorder=5, edgecolor='black', label=f'Theory: (2√2, 0)')
    plt.axvline(asymptote_dm, color='lime', linestyle=':', linewidth=2, label=f'Theory: Asymptote δμ ≈ {asymptote_dm:.3f}')

    plt.xlabel('δμ/γ₀ (Normalized Location Perturbation)')
    plt.ylabel('δs (Log-Scale Perturbation)')
    plt.title('Counterexample Region Heatmap (N=2)')
    plt.legend(fontsize=8)
    plt.xscale('log') # Use log scale for delta_mu if it covers wide range
    # plt.yscale('symlog', linthresh=0.1) # If delta_s also covers range near zero
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle('In-depth Analysis of Counterexample Boundary (N=2)', fontsize=16)
    
    # 确保results目录存在
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/counterexample_boundary_deep_dive.png', dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形而不显示
    
    return delta_mu_vals, critical_delta_s_vals

if __name__ == "__main__":
    delta_mu_vals, critical_delta_s_vals = analyze_boundary_function()
    # theoretical_insight() function call can be here if its print output is desired.
    # The main insights are now integrated into the plotting and printouts of analyze_boundary_function.
    
    print("\n" + "="*60)
    print("Theoretical Insights Summary (N=2 for D_fused <= 2*Sum D_indiv)")
    print("="*60)
    print(f"Boundary Equation: 4((1+exp(2s))^2 + 4x^2) = ((1+exp(s))^2 + x^2)^2, where x=δμ/γ₀, s=δs")
    print(f"- Point on boundary: (x = 2√2 ≈ {2*np.sqrt(2):.4f}, s = 0)")
    print(f"  For s=0: Counterexamples if x > 2√2.")
    print(f"- As s → -∞: Boundary for x approaches ≈ {np.sqrt(7 + np.sqrt(52)):.4f}")
    print(f"  For x < {np.sqrt(7 + np.sqrt(52)):.4f}: Counterexamples if s is sufficiently negative.")
    print("The script `boundary_analysis.py` has been updated to find and plot this boundary numerically.")

# Remove or comment out the old theoretical_insight() if its content is now covered or superseded.
# def theoretical_insight(): ... 