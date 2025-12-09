
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, optimize, integrate
from sklearn.mixture import GaussianMixture
from scipy.stats import norm, nct
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# ==========================================
# 1. Model Density and Likelihood Functions
# ==========================================

# --- 1.1 Gaussian ---
def gaussian_loglik(params, data):
    mu, sigma = params
    if sigma <= 0: return np.inf
    return -np.sum(stats.norm.logpdf(data, loc=mu, scale=sigma))

def fit_gaussian(data):
    mu_init, sigma_init = np.mean(data), np.std(data)
    result = optimize.minimize(gaussian_loglik, x0=[mu_init, sigma_init], args=(data,),
                               method='L-BFGS-B', bounds=[(-np.inf, np.inf), (1e-6, np.inf)])
    return result.x, -result.fun

def gaussian_1pct_var(params):
    mu, sigma = params
    # Returns the 1st percentile
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
    
    if np.any(sigmas <= 1e-3): return np.inf
    
    pdf_vals = mixture_gaussian_pdf(data, params, K)
    # Avoid log(0)
    pdf_vals = np.maximum(pdf_vals, 1e-300)
    return -np.sum(np.log(pdf_vals))

def fit_mixture_gaussian(data, K):
    # Init with sklearn
    gmm = GaussianMixture(n_components=K, random_state=42, max_iter=100).fit(data.reshape(-1, 1))
    weights_init = gmm.weights_[:-1]
    mus_init = gmm.means_.flatten()
    sigmas_init = np.sqrt(gmm.covariances_).flatten()
    
    x0 = np.concatenate([weights_init, mus_init, sigmas_init])
    
    # Simple bounds
    bounds = [(1e-6, 1-1e-6)]*(K-1) + [(-np.inf, np.inf)]*K + [(1e-6, np.inf)]*K
    
    result = optimize.minimize(mixture_gaussian_loglik, x0=x0, args=(data, K),
                               method='L-BFGS-B', bounds=bounds)
    return result.x, -result.fun

def mixture_gaussian_1pct_var(params, K):
    # Numerical root finding for CDF(x) = 0.01
    
    # Reconstruct params
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
        
    # Search range appropriate for returns (e.g. -50% to +50%)
    # Usually -10% is a safe start for 1% VaR for daily returns
    try:
        var_val = optimize.brentq(mix_cdf, -20.0, 5.0) # Search range 
    except:
        var_val = np.nan
    return var_val


# --- 1.3 Weighted Sum of Chi-Squared ---
TMAX = 100.0
N_T = 4096 # Higher resolution
t_grid = np.linspace(0.0, TMAX, N_T)
dt = t_grid[1] - t_grid[0]

def weighted_chisq_cf(t, weights, location):
    # CF of weighted sum: exp(i*t*loc) * prod( (1 - 2*i*t*w_k)^(-0.5) )
    # Note: Using t_grid global for grid ops, but here we might need scalar t support?
    # Actually let's keep the grid approach for PDF
    phi = np.exp(1j * t * location)
    for w in weights:
        # Complex power: (1 - 2itw)^(-0.5)
        # Use np.power to handle complex numbers correctly
        term = (1 - 2j * t * w)
        phi *= np.power(term, -0.5)
    return phi

def weighted_chisq_pdf_grid(x_array, weights, location):
    # Compute CF on grid
    phi_grid = weighted_chisq_cf(t_grid, weights, location)
    
    # Inversion
    # f(x) = (1/pi) * Re[ integral_0^inf exp(-itx) phi(t) dt ]
    x_array = np.asarray(x_array)
    phase = np.exp(-1j * np.outer(x_array, t_grid))
    integrand = phase * phi_grid
    
    # Trapz rule
    w_trapz = np.ones(N_T)
    w_trapz[0] = 0.5; w_trapz[-1] = 0.5
    
    vals = np.real(np.sum(integrand * w_trapz, axis=1) * dt) / np.pi
    # Ensure non-negative
    return np.maximum(vals, 1e-300)

def weighted_chisq_loglik(params, data, K):
    weights = params[:K]
    location = params[K]
    if np.any(weights <= 1e-8): return np.inf
    
    pdf_vals = weighted_chisq_pdf_grid(data, weights, location)
    return -np.sum(np.log(pdf_vals))

def fit_weighted_chisq(data, K):
    # Weights usually small?
    weights_init = np.ones(K) * 0.5
    location_init = np.min(data) # It's a sum of positive vars + loc, so loc approx min?
    # Actually X = loc + sum, sum > 0. So loc is the minimum support.
    # Returns can be negative.
    location_init = np.min(data) - 0.5 
    
    x0 = np.concatenate([weights_init, [location_init]])
    bounds = [(1e-6, 100.0)]*K + [(-100.0, 100.0)]
    
    # This optimization is unstable. 
    # Let's try Nelder-Mead first then BFGS? Or just L-BFGS-B
    try:
        result = optimize.minimize(weighted_chisq_loglik, x0=x0, args=(data, K),
                                   method='L-BFGS-B', bounds=bounds, 
                                   options={'maxiter': 500})
        return result.x, -result.fun
    except:
        return None, -np.inf

def weighted_chisq_1pct_var(params, K):
    # Simulation approach for robustness
    weights = params[:K]
    location = params[K]
    
    n_sim = 1000000
    sim_data = np.zeros(n_sim) + location
    for w in weights:
        # Z^2 where Z ~ N(0,1) is ChiSq(1)
        sim_data += w * (np.random.normal(size=n_sim)**2)
        
    return np.percentile(sim_data, 1)


# --- 1.4 Noncentral t (NCT) ---
def nct_loglik(params, data):
    df, nc, loc, scale = params
    if df <= 0 or scale <= 0: return np.inf
    
    # Use scipy nct pdf: nct.pdf(x, df, nc, loc, scale)
    # Note: scipy implements (x-loc)/scale
    vals = stats.nct.pdf(data, df, nc, loc=loc, scale=scale)
    vals = np.maximum(vals, 1e-300)
    return -np.sum(np.log(vals))

def fit_nct(data):
    # Initial guess: standard t approx
    df_init = 5.0
    nc_init = 0.0
    loc_init, scale_init = stats.norm.fit(data)
    
    x0 = [df_init, nc_init, loc_init, scale_init]
    bounds = [(2.1, 100), (-10, 10), (-np.inf, np.inf), (1e-6, np.inf)]
    
    result = optimize.minimize(nct_loglik, x0=x0, args=(data,),
                               method='L-BFGS-B', bounds=bounds)
    return result.x, -result.fun

def nct_1pct_var(params):
    df, nc, loc, scale = params
    return stats.nct.ppf(0.01, df, nc, loc=loc, scale=scale)


# ==========================================
# 2. Main Execution Block
# ==========================================

def run_analysis():
    print("Loading data...")
    # Adjust path if needed or assume local execution
    try:
        returns = pd.read_csv("DJIA30stockreturns.csv", header=None)
    except FileNotFoundError:
        # Fallback for testing path
        returns = pd.read_csv("/Users/samwallace/Library/Mobile Documents/com~apple~CloudDocs/Statistical Finance/Take Home 2/DJIA30stockreturns.csv", header=None)
        
    print(f"Data Loaded. Shape: {returns.shape}")
    
    all_results = {}
    
    # Limit stocks for testing if needed, but instructions say "all"
    limit_stocks = None 
    
    stocks_to_run = returns.columns[:limit_stocks] if limit_stocks else returns.columns
    
    for col in stocks_to_run:
        print(f"\\nProcessing Stock {col}...")
        data = returns[col].values
        
        # --- Fit Models ---
        stock_res = {}
        
        # Gaussian
        try:
            p, ll = fit_gaussian(data)
            stock_res['Gaussian'] = {'params': p, 'loglik': ll, 'n_params': 2, 'var': gaussian_1pct_var(p)}
        except Exception as e:
            print(f"  Gaussian fail: {e}")
            stock_res['Gaussian'] = {'loglik': -np.inf, 'AIC': np.inf, 'BIC': np.inf, 'var': np.nan}
            
        # GMM
        for K in [2, 3]:
            name = f'MixGauss_K{K}'
            try:
                p, ll = fit_mixture_gaussian(data, K)
                stock_res[name] = {'params': p, 'loglik': ll, 'n_params': 3*K-1, 'var': mixture_gaussian_1pct_var(p, K)}
            except Exception as e:
                print(f"  {name} fail: {e}")
                stock_res[name] = {'loglik': -np.inf, 'AIC': np.inf, 'BIC': np.inf, 'var': np.nan}
                
        # Weighted ChiSq
        for K in [2, 3, 4, 5]:
            name = f'WeightedChiSq_K{K}'
            try:
                p, ll = fit_weighted_chisq(data, K)
                if p is None: raise ValueError("Fit returned None")
                stock_res[name] = {'params': p, 'loglik': ll, 'n_params': K+1, 'var': weighted_chisq_1pct_var(p, K)}
            except Exception as e:
                print(f"  {name} fail: {e}")
                stock_res[name] = {'params': None, 'loglik': -np.inf, 'AIC': np.inf, 'BIC': np.inf, 'var': np.nan}
                
        # NCT
        try:
            p, ll = fit_nct(data)
            stock_res['NCT'] = {'params': p, 'loglik': ll, 'n_params': 4, 'var': nct_1pct_var(p)}
        except Exception as e:
            print(f"  NCT fail: {e}")
            stock_res['NCT'] = {'loglik': -np.inf, 'AIC': np.inf, 'BIC': np.inf, 'var': np.nan}
            
        # Empirical VaR
        stock_res['Empirical'] = {'var': np.percentile(data, 1)}
        
        # --- Calculate AIC/BIC ---
        n_obs = len(data)
        for m_name in stock_res:
            if m_name == 'Empirical': continue
            
            res = stock_res[m_name]
            if np.isinf(res['loglik']):
                res['AIC'] = np.inf
                res['BIC'] = np.inf
            else:
                k = res['n_params']
                res['AIC'] = 2*k - 2*res['loglik']
                res['BIC'] = k*np.log(n_obs) - 2*res['loglik']
        
        all_results[col] = stock_res

    # ==========================================
    # 3. Visualization (First Stock)
    # ==========================================
    print("\\nGenerating visualization for first stock...")
    first_stock = returns.columns[0]
    res_first = all_results[first_stock]
    data_first = returns[first_stock].values
    
    plt.figure(figsize=(12, 8))
    
    # Kernel Density
    kde = stats.gaussian_kde(data_first)
    x_grid = np.linspace(data_first.min()-1, data_first.max()+1, 1000)
    plt.plot(x_grid, kde(x_grid), 'k-', lw=2, label='Kernel Density')
    
    # Models
    # Gaussian
    p = res_first['Gaussian']['params']
    plt.plot(x_grid, stats.norm.pdf(x_grid, *p), label='Gaussian', linestyle='--')
    
    # GMM
    for K in [2, 3]:
        name = f'MixGauss_K{K}'
        if not np.isinf(res_first[name]['loglik']):
            y = mixture_gaussian_pdf(x_grid, res_first[name]['params'], K)
            plt.plot(x_grid, y, label=name, linestyle=':')
            
    # Weighted ChiSq
    for K in [2, 3, 4, 5]:
        name = f'WeightedChiSq_K{K}'
        if res_first[name]['params'] is not None:
             weights = res_first[name]['params'][:K]
             loc = res_first[name]['params'][K]
             y = weighted_chisq_pdf_grid(x_grid, weights, loc)
             plt.plot(x_grid, y, label=name, alpha=0.6)
             
    # NCT
    if not np.isinf(res_first['NCT']['loglik']):
        p = res_first['NCT']['params'] # df, nc, loc, scale
        y = stats.nct.pdf(x_grid, p[0], p[1], loc=p[2], scale=p[3])
        plt.plot(x_grid, y, label='NCT', linestyle='-.')
        
    plt.legend()
    plt.title(f"Density Fit for Stock {first_stock}")
    plt.xlabel("Percentage Return")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.3)
    plt.savefig('fitted_distributions.png', dpi=300)
    print("Plot saved to fitted_distributions.png")
    
    # ==========================================
    # 4. Reporting Tables
    # ==========================================
    print("\\nGenerating Reports...")
    
    # --- Model Selection Table ---
    # Rows: Stocks, Cols: Best Model (LL), Best (AIC), Best (BIC)
    selection_rows = []
    for col in stocks_to_run:
        res = all_results[col]
        # Filter out empirical
        models = [k for k in res.keys() if k != 'Empirical']
        
        # Helper to find best
        def get_best(metric, minimize=True):
            valid = [(k, res[k][metric]) for k in models if not np.isinf(res[k][metric])]
            if not valid: return "None"
            if minimize:
                return min(valid, key=lambda x: x[1])[0]
            else:
                return max(valid, key=lambda x: x[1])[0]
                
        best_ll = get_best('loglik', minimize=False)
        best_aic = get_best('AIC', minimize=True)
        best_bic = get_best('BIC', minimize=True)
        
        selection_rows.append([col, best_ll, best_aic, best_bic])
        
    sel_df = pd.DataFrame(selection_rows, columns=['Stock', 'Best_LogLik', 'Best_AIC', 'Best_BIC'])
    print("\\nModel Selection Table (LaTeX):")
    print(sel_df.to_latex(index=False))
    
    # --- VaR Table ---
    # Rows: Stocks
    # Cols: Empirical, then models. Bold the closest to empirical.
    
    var_rows = []
    models_order = ['Gaussian', 'MixGauss_K2', 'MixGauss_K3', 
                    'WeightedChiSq_K2', 'WeightedChiSq_K3', 'WeightedChiSq_K4', 'WeightedChiSq_K5',
                    'NCT']
    
    header = ['Stock', 'Empirical'] + models_order
    
    # We will construct the LaTeX body manually to handle bolding
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
            
        # Find closest
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
    
    print("\\nVaR Comparison Table (LaTeX):")
    print("\n".join(latex_lines))
    
    # Save results to file as well for inspection
    with open('results_summary.txt', 'w') as f:
        f.write("Analysis Complete.\n")
        f.write(sel_df.to_string())

if __name__ == "__main__":
    run_analysis()
