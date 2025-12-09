#!/usr/bin/env python3
"""
Assignment 2 Solution - GPU-Optimized Version
Statistical Finance - Stock Return Distribution Fitting

Fits multiple IID models to percentage log returns and compares:
- In-sample fits (Log-Likelihood, AIC, BIC)
- 1% Value at Risk calculations

GPU acceleration via CuPy with automatic CPU fallback.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, optimize
from sklearn.mixture import GaussianMixture
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ==========================================
# GPU/CPU Backend Selection
# ==========================================
try:
    import cupy as cp
    GPU_AVAILABLE = cp.cuda.is_available()
    if GPU_AVAILABLE:
        print("✓ CuPy detected, using GPU acceleration")
        xp = cp  # Use CuPy for array operations
    else:
        print("✗ CuPy installed but CUDA not available, using CPU")
        xp = np
        GPU_AVAILABLE = False
except ImportError:
    print("✗ CuPy not installed, using CPU (NumPy)")
    xp = np
    GPU_AVAILABLE = False

def to_numpy(arr):
    """Convert array to NumPy (for scipy compatibility)"""
    if GPU_AVAILABLE and hasattr(arr, 'get'):
        return arr.get()
    return np.asarray(arr)

def to_backend(arr):
    """Convert array to current backend (GPU or CPU)"""
    return xp.asarray(arr)

# Set random seed for reproducibility
np.random.seed(42)
if GPU_AVAILABLE:
    cp.random.seed(42)

# ==========================================
# 1. Model Density and Likelihood Functions
# ==========================================

# --- 1.1 Gaussian ---
def gaussian_loglik(params, data):
    mu, sigma = params
    if sigma <= 0:
        return np.inf
    return -np.sum(stats.norm.logpdf(data, loc=mu, scale=sigma))

def fit_gaussian(data):
    mu_init, sigma_init = np.mean(data), np.std(data)
    result = optimize.minimize(
        gaussian_loglik, x0=[mu_init, sigma_init], args=(data,),
        method='L-BFGS-B', bounds=[(-np.inf, np.inf), (1e-6, np.inf)]
    )
    return result.x, -result.fun

def gaussian_1pct_var(params):
    mu, sigma = params
    return stats.norm.ppf(0.01, loc=mu, scale=sigma)

# --- 1.2 Mixture of Gaussians ---
def mixture_gaussian_pdf(x, params, K):
    weights = np.zeros(K)
    weights[:-1] = params[:K-1]
    weights[-1] = 1 - np.sum(weights[:-1])
    mus = params[K-1: 2*K-1]
    sigmas = params[2*K-1: 3*K-1]
    
    pdf = np.zeros_like(x)
    for i in range(K):
        pdf += weights[i] * stats.norm.pdf(x, loc=mus[i], scale=sigmas[i])
    return pdf

def mixture_gaussian_loglik(params, data, K):
    weights = np.zeros(K)
    weights[:-1] = params[:K-1]
    weights[-1] = 1 - np.sum(weights[:-1])
    
    if np.any(weights <= 1e-3) or np.any(weights >= 1 - 1e-3) or not np.isclose(np.sum(weights), 1.0):
        return np.inf
        
    mus = params[K-1: 2*K-1]
    sigmas = params[2*K-1: 3*K-1]
    
    if np.any(sigmas <= 1e-3):
        return np.inf
    
    pdf_vals = mixture_gaussian_pdf(data, params, K)
    pdf_vals = np.maximum(pdf_vals, 1e-300)
    return -np.sum(np.log(pdf_vals))

def fit_mixture_gaussian(data, K):
    # Initialize with sklearn GMM
    gmm = GaussianMixture(n_components=K, random_state=42, max_iter=100).fit(data.reshape(-1, 1))
    weights_init = gmm.weights_[:-1]
    mus_init = gmm.means_.flatten()
    sigmas_init = np.sqrt(gmm.covariances_).flatten()
    
    x0 = np.concatenate([weights_init, mus_init, sigmas_init])
    bounds = [(1e-6, 1-1e-6)]*(K-1) + [(-np.inf, np.inf)]*K + [(1e-6, np.inf)]*K
    
    result = optimize.minimize(
        mixture_gaussian_loglik, x0=x0, args=(data, K),
        method='L-BFGS-B', bounds=bounds
    )
    return result.x, -result.fun

def mixture_gaussian_1pct_var(params, K):
    weights = np.zeros(K)
    weights[:-1] = params[:K-1]
    weights[-1] = 1 - np.sum(weights[:-1])
    mus = params[K-1: 2*K-1]
    sigmas = params[2*K-1: 3*K-1]
    
    def mix_cdf(x):
        cdf_val = 0
        for i in range(K):
            cdf_val += weights[i] * stats.norm.cdf(x, loc=mus[i], scale=sigmas[i])
        return cdf_val - 0.01
    
    try:
        var_val = optimize.brentq(mix_cdf, -20.0, 5.0)
    except:
        var_val = np.nan
    return var_val

# --- 1.3 Weighted Sum of Chi-Squared (GPU-Optimized) ---
TMAX = 100.0
N_T = 4096  # Higher resolution for CF inversion

def weighted_chisq_cf_gpu(t_grid, weights, location):
    """
    Characteristic function of weighted sum of chi-squared RVs.
    GPU-accelerated version using CuPy.
    
    CF(t) = exp(i*t*location) * prod_k( (1 - 2*i*t*w_k)^(-0.5) )
    """
    t = xp.asarray(t_grid)
    phi = xp.exp(1j * t * location)
    
    for w in weights:
        term = (1 - 2j * t * w)
        phi = phi * xp.power(term, -0.5)
    
    return phi

def weighted_chisq_pdf_grid(x_array, weights, location):
    """
    Compute PDF via characteristic function inversion.
    f(x) = (1/pi) * Re[ integral_0^inf exp(-itx) phi(t) dt ]
    
    Uses GPU if available for the large matrix operations.
    """
    # Create grids on the current backend
    t_grid = xp.linspace(0.0, TMAX, N_T)
    dt = float(TMAX / (N_T - 1))
    
    # Compute CF on grid
    phi_grid = weighted_chisq_cf_gpu(t_grid, weights, location)
    
    # Convert x_array to backend
    x = xp.asarray(x_array)
    
    # Compute phase matrix: exp(-i * outer(x, t))
    # This is the expensive operation that benefits from GPU
    phase = xp.exp(-1j * xp.outer(x, t_grid))
    integrand = phase * phi_grid
    
    # Trapezoidal weights
    w_trapz = xp.ones(N_T)
    w_trapz[0] = 0.5
    w_trapz[-1] = 0.5
    
    # Integration
    vals = xp.real(xp.sum(integrand * w_trapz, axis=1) * dt) / np.pi
    vals = xp.maximum(vals, 1e-300)
    
    return to_numpy(vals)

def weighted_chisq_loglik(params, data, K):
    weights = params[:K]
    location = params[K]
    
    if np.any(weights <= 1e-8):
        return np.inf
    
    pdf_vals = weighted_chisq_pdf_grid(data, weights, location)
    return -np.sum(np.log(pdf_vals))

def fit_weighted_chisq(data, K):
    # Initial guess
    weights_init = np.ones(K) * 0.5
    location_init = np.min(data) - 0.5  # X = loc + sum(positive), so loc < min(data)
    
    x0 = np.concatenate([weights_init, [location_init]])
    bounds = [(1e-6, 100.0)]*K + [(-100.0, 100.0)]
    
    try:
        result = optimize.minimize(
            weighted_chisq_loglik, x0=x0, args=(data, K),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 500}
        )
        return result.x, -result.fun
    except:
        return None, -np.inf

def weighted_chisq_1pct_var(params, K):
    """
    Compute 1% VaR via Monte Carlo simulation.
    GPU-accelerated for large sample generation.
    """
    weights = params[:K]
    location = params[K]
    
    n_sim = 1000000
    
    if GPU_AVAILABLE:
        # GPU-accelerated simulation
        sim_data = xp.zeros(n_sim) + location
        for w in weights:
            sim_data += w * (cp.random.normal(size=n_sim) ** 2)
        result = float(xp.percentile(sim_data, 1))
    else:
        # CPU simulation
        sim_data = np.zeros(n_sim) + location
        for w in weights:
            sim_data += w * (np.random.normal(size=n_sim) ** 2)
        result = np.percentile(sim_data, 1)
    
    return result

# --- 1.4 Noncentral t (NCT) ---
def nct_loglik(params, data):
    df, nc, loc, scale = params
    if df <= 0 or scale <= 0:
        return np.inf
    
    vals = stats.nct.pdf(data, df, nc, loc=loc, scale=scale)
    vals = np.maximum(vals, 1e-300)
    return -np.sum(np.log(vals))

def fit_nct(data):
    df_init = 5.0
    nc_init = 0.0
    loc_init, scale_init = stats.norm.fit(data)
    
    x0 = [df_init, nc_init, loc_init, scale_init]
    bounds = [(2.1, 100), (-10, 10), (-np.inf, np.inf), (1e-6, np.inf)]
    
    result = optimize.minimize(
        nct_loglik, x0=x0, args=(data,),
        method='L-BFGS-B', bounds=bounds
    )
    return result.x, -result.fun

def nct_1pct_var(params):
    df, nc, loc, scale = params
    return stats.nct.ppf(0.01, df, nc, loc=loc, scale=scale)


# ==========================================
# 2. Main Execution Block
# ==========================================

def run_analysis():
    print("\n" + "="*60)
    print("Assignment 2: IID Model Fitting for Stock Returns")
    print("="*60)
    
    print("\nLoading data...")
    # Load data - handle both possible locations
    data_file = "DJIA30stockreturns.csv"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, data_file)
    
    if os.path.exists(data_path):
        returns = pd.read_csv(data_path, header=None)
    elif os.path.exists(data_file):
        returns = pd.read_csv(data_file, header=None)
    else:
        raise FileNotFoundError(f"Could not find {data_file}")
    
    print(f"Data Loaded. Shape: {returns.shape}")
    print(f"  - {returns.shape[0]} time periods")
    print(f"  - {returns.shape[1]} stocks")
    
    all_results = {}
    stocks_to_run = returns.columns
    
    for col in stocks_to_run:
        print(f"\nProcessing Stock {col}...", end=" ", flush=True)
        data = returns[col].values
        
        stock_res = {}
        
        # --- Fit Models ---
        
        # Gaussian
        try:
            p, ll = fit_gaussian(data)
            stock_res['Gaussian'] = {
                'params': p, 'loglik': ll, 'n_params': 2,
                'var': gaussian_1pct_var(p)
            }
            print("G", end="", flush=True)
        except Exception as e:
            stock_res['Gaussian'] = {'loglik': -np.inf, 'AIC': np.inf, 'BIC': np.inf, 'var': np.nan}
        
        # Mixture of Gaussians
        for K in [2, 3]:
            name = f'MixGauss_K{K}'
            try:
                p, ll = fit_mixture_gaussian(data, K)
                stock_res[name] = {
                    'params': p, 'loglik': ll, 'n_params': 3*K-1,
                    'var': mixture_gaussian_1pct_var(p, K)
                }
                print(f"M{K}", end="", flush=True)
            except Exception as e:
                stock_res[name] = {'loglik': -np.inf, 'AIC': np.inf, 'BIC': np.inf, 'var': np.nan}
        
        # Weighted Chi-Squared
        for K in [2, 3, 4, 5]:
            name = f'WeightedChiSq_K{K}'
            try:
                p, ll = fit_weighted_chisq(data, K)
                if p is None:
                    raise ValueError("Fit returned None")
                stock_res[name] = {
                    'params': p, 'loglik': ll, 'n_params': K+1,
                    'var': weighted_chisq_1pct_var(p, K)
                }
                print(f"C{K}", end="", flush=True)
            except Exception as e:
                stock_res[name] = {
                    'params': None, 'loglik': -np.inf,
                    'AIC': np.inf, 'BIC': np.inf, 'var': np.nan
                }
        
        # NCT
        try:
            p, ll = fit_nct(data)
            stock_res['NCT'] = {
                'params': p, 'loglik': ll, 'n_params': 4,
                'var': nct_1pct_var(p)
            }
            print("N", end="", flush=True)
        except Exception as e:
            stock_res['NCT'] = {'loglik': -np.inf, 'AIC': np.inf, 'BIC': np.inf, 'var': np.nan}
        
        # Empirical VaR
        stock_res['Empirical'] = {'var': np.percentile(data, 1)}
        
        # --- Calculate AIC/BIC ---
        n_obs = len(data)
        for m_name in stock_res:
            if m_name == 'Empirical':
                continue
            
            res = stock_res[m_name]
            if np.isinf(res['loglik']):
                res['AIC'] = np.inf
                res['BIC'] = np.inf
            else:
                k = res['n_params']
                res['AIC'] = 2*k - 2*res['loglik']
                res['BIC'] = k*np.log(n_obs) - 2*res['loglik']
        
        all_results[col] = stock_res
        print(" ✓")
    
    # ==========================================
    # 3. Visualization (First Stock)
    # ==========================================
    print("\n" + "-"*60)
    print("Generating visualization for first stock...")
    
    first_stock = returns.columns[0]
    res_first = all_results[first_stock]
    data_first = returns[first_stock].values
    
    plt.figure(figsize=(14, 9))
    
    # Kernel Density
    kde = stats.gaussian_kde(data_first)
    x_grid = np.linspace(data_first.min()-1, data_first.max()+1, 1000)
    plt.plot(x_grid, kde(x_grid), 'k-', lw=2.5, label='Kernel Density (Empirical)')
    
    # Gaussian
    if not np.isinf(res_first['Gaussian']['loglik']):
        p = res_first['Gaussian']['params']
        plt.plot(x_grid, stats.norm.pdf(x_grid, *p), '--', lw=1.5, label='Gaussian')
    
    # Mixture of Gaussians
    colors_gmm = ['tab:orange', 'tab:green']
    for idx, K in enumerate([2, 3]):
        name = f'MixGauss_K{K}'
        if not np.isinf(res_first[name]['loglik']):
            y = mixture_gaussian_pdf(x_grid, res_first[name]['params'], K)
            plt.plot(x_grid, y, ':', lw=1.5, color=colors_gmm[idx], label=f'Mixture Gaussian (K={K})')
    
    # Weighted Chi-Squared
    colors_chi = ['tab:red', 'tab:purple', 'tab:brown', 'tab:pink']
    for idx, K in enumerate([2, 3, 4, 5]):
        name = f'WeightedChiSq_K{K}'
        if res_first[name].get('params') is not None:
            weights = res_first[name]['params'][:K]
            loc = res_first[name]['params'][K]
            y = weighted_chisq_pdf_grid(x_grid, weights, loc)
            plt.plot(x_grid, y, '-', alpha=0.7, lw=1.2, color=colors_chi[idx], 
                     label=f'Weighted χ² (K={K})')
    
    # NCT
    if not np.isinf(res_first['NCT']['loglik']):
        p = res_first['NCT']['params']
        y = stats.nct.pdf(x_grid, p[0], p[1], loc=p[2], scale=p[3])
        plt.plot(x_grid, y, '-.', lw=2, color='tab:cyan', label='Noncentral t')
    
    plt.legend(loc='upper right', fontsize=9)
    plt.title(f"Fitted Distributions for Stock {first_stock}", fontsize=14)
    plt.xlabel("Percentage Log Return", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_plot = 'fitted_distributions.png'
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_plot}")
    
    # ==========================================
    # 4. Reporting Tables
    # ==========================================
    print("\n" + "-"*60)
    print("Generating Reports...")
    
    # --- Model Selection Table ---
    selection_rows = []
    for col in stocks_to_run:
        res = all_results[col]
        models = [k for k in res.keys() if k != 'Empirical']
        
        def get_best(metric, minimize=True):
            valid = [(k, res[k][metric]) for k in models if not np.isinf(res[k][metric])]
            if not valid:
                return "None"
            if minimize:
                return min(valid, key=lambda x: x[1])[0]
            else:
                return max(valid, key=lambda x: x[1])[0]
        
        best_ll = get_best('loglik', minimize=False)
        best_aic = get_best('AIC', minimize=True)
        best_bic = get_best('BIC', minimize=True)
        
        selection_rows.append([col, best_ll, best_aic, best_bic])
    
    sel_df = pd.DataFrame(selection_rows, columns=['Stock', 'Best_LogLik', 'Best_AIC', 'Best_BIC'])
    
    print("\n" + "="*60)
    print("MODEL SELECTION TABLE (LaTeX)")
    print("="*60)
    print(sel_df.to_latex(index=False))
    
    # --- VaR Table ---
    models_order = [
        'Gaussian', 'MixGauss_K2', 'MixGauss_K3',
        'WeightedChiSq_K2', 'WeightedChiSq_K3', 'WeightedChiSq_K4', 'WeightedChiSq_K5',
        'NCT'
    ]
    
    header = ['Stock', 'Empirical'] + models_order
    
    latex_lines = []
    latex_lines.append("\\begin{tabular}{" + "l" * len(header) + "}")
    latex_lines.append("\\hline")
    latex_lines.append(" & ".join(header).replace('_', '\\_') + " \\\\")
    latex_lines.append("\\hline")
    
    for col in stocks_to_run:
        res = all_results[col]
        emp_var = res['Empirical']['var']
        
        row_vals = [str(col), f"{emp_var:.4f}"]
        
        model_vars = {}
        for m in models_order:
            val = res.get(m, {}).get('var', np.nan)
            model_vars[m] = val
        
        valid_diffs = {m: abs(v - emp_var) for m, v in model_vars.items() if not np.isnan(v)}
        closest_model = min(valid_diffs, key=valid_diffs.get) if valid_diffs else None
        
        for m in models_order:
            val = model_vars[m]
            if np.isnan(val):
                row_vals.append("NaN")
            else:
                fmt_val = f"{val:.4f}"
                if m == closest_model:
                    row_vals.append(f"\\textbf{{{fmt_val}}}")
                else:
                    row_vals.append(fmt_val)
        
        latex_lines.append(" & ".join(row_vals) + " \\\\")
    
    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    
    print("\n" + "="*60)
    print("VaR COMPARISON TABLE (LaTeX)")
    print("="*60)
    print("\n".join(latex_lines))
    
    # Save results summary
    output_file = 'results_summary.txt'
    with open(output_file, 'w') as f:
        f.write("Assignment 2 - Analysis Results\n")
        f.write("="*60 + "\n\n")
        f.write("Model Selection Summary:\n")
        f.write(sel_df.to_string())
        f.write("\n\n")
        f.write("VaR Comparison (LaTeX):\n")
        f.write("\n".join(latex_lines))
    
    print(f"\nResults saved to: {output_file}")
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)


if __name__ == "__main__":
    run_analysis()
