#!/usr/bin/env python3
"""
GPU-Accelerated Stock Model Fitting

Fits multiple distribution models to DJIA30 stock returns:
- Gaussian
- Mixture of Gaussians (K=2,3)
- Weighted Chi-Squared (K=2,3,4,5) [GPU-accelerated via CuPy]
- Noncentral t (location-scale)

Computes AIC, BIC, and 1% VaR for all models.
Saves results to results.pkl for plotting.

Requirements: CuPy, NumPy, SciPy, pandas, sklearn
Target: Ubuntu 22 with NVIDIA RTX 4080
"""

import numpy as np
import pandas as pd
from scipy import stats, optimize
from sklearn.mixture import GaussianMixture
import pickle
import time
import warnings

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("CuPy available - using GPU acceleration")
except ImportError:
    GPU_AVAILABLE = False
    print("CuPy not available - falling back to NumPy")

warnings.filterwarnings('ignore')

# =============================================================================
# Constants
# =============================================================================
TMAX = 1000.0   # Max t for characteristic function integration
N_T = 20000     # Number of points in t-grid
ALPHA = 0.01  # VaR level (1%)

# =============================================================================
# Utility Functions
# =============================================================================

def create_t_grid(tmax=TMAX, n_t=N_T):
    """Create t-grid for characteristic function integration."""
    t_grid = np.linspace(0.0, tmax, n_t)
    dt = t_grid[1] - t_grid[0]
    return t_grid, dt

t_grid, dt = create_t_grid()

# =============================================================================
# Gaussian Distribution
# =============================================================================

def gaussian_pdf(x, mu, sigma):
    """PDF of Gaussian distribution."""
    return stats.norm.pdf(x, loc=mu, scale=sigma)

def gaussian_loglik(params, data):
    """Negative log-likelihood for Gaussian."""
    mu, sigma = params
    if sigma <= 0:
        return np.inf
    return -np.sum(stats.norm.logpdf(data, loc=mu, scale=sigma))

def fit_gaussian(data):
    """Fit Gaussian via MLE."""
    mu_init = np.mean(data)
    sigma_init = np.std(data)
    
    result = optimize.minimize(
        gaussian_loglik,
        x0=[mu_init, sigma_init],
        args=(data,),
        method='L-BFGS-B',
        bounds=[(-np.inf, np.inf), (1e-6, np.inf)]
    )
    return result.x, -result.fun

# =============================================================================
# Mixture of Gaussians
# =============================================================================

def mixture_gaussian_pdf(x, params, K):
    """PDF of K-component Gaussian mixture."""
    weights = np.zeros(K)
    weights[:-1] = params[:K-1]
    weights[-1] = 1 - np.sum(weights[:-1])
    
    mus = params[K-1:2*K-1]
    sigmas = params[2*K-1:3*K-1]
    
    pdf = np.zeros_like(x, dtype=float)
    for i in range(K):
        pdf += weights[i] * stats.norm.pdf(x, loc=mus[i], scale=sigmas[i])
    return pdf

def mixture_gaussian_loglik(params, data, K):
    """Negative log-likelihood for K-component Gaussian mixture."""
    weights = np.zeros(K)
    weights[:-1] = params[:K-1]
    weights[-1] = 1 - np.sum(weights[:-1])
    
    if np.any(weights <= 1e-3) or np.any(weights >= 1 - 1e-3):
        return np.inf
    if not np.isclose(np.sum(weights), 1.0, atol=1e-6):
        return np.inf
    
    mus = params[K-1:2*K-1]
    sigmas = params[2*K-1:3*K-1]
    
    if np.any(sigmas <= 1e-3):
        return np.inf
    
    pdf_vals = mixture_gaussian_pdf(data, params, K)
    pdf_vals = np.maximum(pdf_vals, 1e-300)
    
    if np.any(pdf_vals <= 0) or np.any(~np.isfinite(pdf_vals)):
        return np.inf
    
    log_lik = np.sum(np.log(pdf_vals))
    return -log_lik if np.isfinite(log_lik) else np.inf

def fit_mixture_gaussian(data, K):
    """Fit K-component Gaussian mixture via MLE."""
    # Use sklearn for initialization
    gmm = GaussianMixture(n_components=K, random_state=42, max_iter=100)
    gmm.fit(data.reshape(-1, 1))
    
    weights_init = gmm.weights_[:-1]
    mus_init = gmm.means_.flatten()
    sigmas_init = np.sqrt(gmm.covariances_).flatten()
    
    x0 = np.concatenate([weights_init, mus_init, sigmas_init])
    
    bounds = []
    for _ in range(K-1):
        bounds.append((1e-6, 1 - 1e-6))
    for _ in range(K):
        bounds.append((-np.inf, np.inf))
    for _ in range(K):
        bounds.append((1e-6, np.inf))
    
    result = optimize.minimize(
        mixture_gaussian_loglik,
        x0=x0,
        args=(data, K),
        method='L-BFGS-B',
        bounds=bounds
    )
    return result.x, -result.fun

# =============================================================================
# Weighted Chi-Squared (GPU-accelerated)
# =============================================================================

def weighted_chisq_cf_grid_cpu(weights, location, t_grid):
    """CPU version: Characteristic function of weighted chi-squared (log-form)."""
    # log phi(t) = i*t*loc - 0.5 * sum(log(1 - 2*i*t*w))
    z = 1 - 2j * np.outer(t_grid, weights)
    log_phi = 1j * t_grid * location - 0.5 * np.sum(np.log(z), axis=1)
    return np.exp(log_phi)

def weighted_chisq_pdf_from_phi_cpu(x_array, phi, t_grid, dt):
    """CPU version: PDF via Gil-Pelaez formula."""
    x_array = np.asarray(x_array)
    phase = np.exp(-1j * np.outer(x_array, t_grid))
    integrand = phase * phi
    weights = np.ones(len(t_grid))
    weights[0] = weights[-1] = 0.5
    vals = np.real(np.sum(integrand * weights, axis=1) * dt)
    return vals / np.pi

if GPU_AVAILABLE:
    def weighted_chisq_cf_grid_gpu(weights, location, t_grid_gpu):
        """GPU version: Characteristic function of weighted chi-squared (log-form)."""
        z = 1 - 2j * cp.outer(t_grid_gpu, weights)
        log_phi = 1j * t_grid_gpu * location - 0.5 * cp.sum(cp.log(z), axis=1)
        return cp.exp(log_phi)
    
    def weighted_chisq_pdf_from_phi_gpu(x_array, phi, t_grid_gpu, dt):
        """GPU version: PDF via Gil-Pelaez formula (batched)."""
        x_gpu = cp.asarray(x_array)
        n_data = len(x_gpu)
        n_t = len(t_grid_gpu)
        
        # Process in batches to avoid OOM
        # x_gpu (n_data) x t_grid_gpu (n_t) -> n_data * n_t * 16 bytes
        # 8000 * 20000 * 16 = 2.5 GB. Should fit, but maybe overhead/fragmentation.
        # Let's batch over x_array
        batch_size = 1000 # Process 1000 data points at a time
        pdf_vals = cp.zeros(n_data, dtype=cp.float64)
        
        weights = cp.ones(n_t)
        weights[0] = weights[-1] = 0.5
        
        for i in range(0, n_data, batch_size):
            end = min(i + batch_size, n_data)
            x_batch = x_gpu[i:end]
            
            # phase: (batch, n_t)
            phase = cp.exp(-1j * cp.outer(x_batch, t_grid_gpu))
            integrand = phase * phi # broadcast phi (n_t)
            
            # Sum over t
            batch_vals = cp.real(cp.sum(integrand * weights, axis=1) * dt)
            pdf_vals[i:end] = batch_vals
            
            # Free memory
            del phase, integrand, batch_vals
            cp.get_default_memory_pool().free_all_blocks()
            
        return cp.asnumpy(pdf_vals) / np.pi

def weighted_chisq_cf_grid(weights, location):
    """Characteristic function computation (auto GPU/CPU)."""
    if GPU_AVAILABLE:
        t_grid_gpu = cp.asarray(t_grid)
        weights_gpu = cp.asarray(weights)
        return weighted_chisq_cf_grid_gpu(weights_gpu, location, t_grid_gpu), t_grid_gpu
    else:
        return weighted_chisq_cf_grid_cpu(weights, location, t_grid), t_grid

def weighted_chisq_pdf_from_phi(x_array, phi, t_grid_arr):
    """PDF computation (auto GPU/CPU)."""
    if GPU_AVAILABLE:
        return weighted_chisq_pdf_from_phi_gpu(x_array, phi, t_grid_arr, dt)
    else:
        return weighted_chisq_pdf_from_phi_cpu(x_array, phi, t_grid_arr, dt)

def weighted_chisq_loglik(params, data, K):
    """Negative log-likelihood for weighted chi-squared (FULL, not binned)."""
    weights = params[:K]
    location = params[K]
    
    # Removed strict positivity check to allow negative weights
    # But we might want to bound them to avoid extreme values
    if np.any(np.abs(weights) > 100):
        return np.inf
    
    phi, t_arr = weighted_chisq_cf_grid(weights, location)
    pdf_vals = weighted_chisq_pdf_from_phi(data, phi, t_arr)
    
    if np.any(pdf_vals <= 0) or np.any(~np.isfinite(pdf_vals)):
        return np.inf
    
    return -np.sum(np.log(pdf_vals))

def fit_weighted_chisq(data, K):
    """Fit weighted chi-squared model via MLE (FULL, GPU-accelerated)."""
    # Initialize with mix of positive and negative weights
    K_half = K // 2
    weights_init = np.concatenate([
        -np.linspace(1, 0.5, K_half),
        np.linspace(0.5, 1, K - K_half)
    ])
    weights_init /= np.std(weights_init) # Scaling
    location_init = np.mean(data)
    
    x0 = np.concatenate([weights_init, [location_init]])
    
    # Bounds: allow negative weights
    w_max = 10.0
    bounds = [(-w_max, w_max) for _ in range(K)] + [(-np.inf, np.inf)]
    
    result = optimize.minimize(
        weighted_chisq_loglik,
        x0=x0,
        args=(data, K),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000}
    )
    return result.x, -result.fun

# =============================================================================
# Difference of Weighted Chi-Squared (K positive + K negative components)
# =============================================================================

def diff_weighted_chisq_cf_grid_cpu(weights_pos, weights_neg, location, t_grid):
    """CPU version: CF for difference of weighted chi-squared.
    
    X = location + sum(a_i * Z_i^2) - sum(b_j * W_j^2)
    where a_i > 0, b_j > 0, Z_i, W_j ~ N(0,1) iid.
    
    phi(t) = exp(it*loc) * prod((1-2it*a_i)^(-1/2)) * prod((1+2it*b_j)^(-1/2))
    """
    # Positive part: (1 - 2it*a)
    z_pos = 1 - 2j * np.outer(t_grid, weights_pos)
    # Negative part: (1 + 2it*b)
    z_neg = 1 + 2j * np.outer(t_grid, weights_neg)
    
    # Log-form for stability
    log_phi = (1j * t_grid * location 
               - 0.5 * np.sum(np.log(z_pos), axis=1)
               - 0.5 * np.sum(np.log(z_neg), axis=1))
    return np.exp(log_phi)

if GPU_AVAILABLE:
    def diff_weighted_chisq_cf_grid_gpu(weights_pos, weights_neg, location, t_grid_gpu):
        """GPU version: CF for difference of weighted chi-squared."""
        z_pos = 1 - 2j * cp.outer(t_grid_gpu, weights_pos)
        z_neg = 1 + 2j * cp.outer(t_grid_gpu, weights_neg)
        
        log_phi = (1j * t_grid_gpu * location
                   - 0.5 * cp.sum(cp.log(z_pos), axis=1)
                   - 0.5 * cp.sum(cp.log(z_neg), axis=1))
        return cp.exp(log_phi)

def diff_weighted_chisq_cf_grid(weights_pos, weights_neg, location):
    """Characteristic function for difference model (auto GPU/CPU)."""
    if GPU_AVAILABLE:
        t_grid_gpu = cp.asarray(t_grid)
        w_pos_gpu = cp.asarray(weights_pos)
        w_neg_gpu = cp.asarray(weights_neg)
        return diff_weighted_chisq_cf_grid_gpu(w_pos_gpu, w_neg_gpu, location, t_grid_gpu), t_grid_gpu
    else:
        return diff_weighted_chisq_cf_grid_cpu(weights_pos, weights_neg, location, t_grid), t_grid

def diff_weighted_chisq_loglik(params, data, K):
    """Negative log-likelihood for difference of weighted chi-squared.
    
    params: [w_pos_1..w_pos_K, w_neg_1..w_neg_K, location]
    """
    weights_pos = params[:K]
    weights_neg = params[K:2*K]
    location = params[2*K]
    
    # Enforce positivity for both weight sets
    if np.any(weights_pos <= 0.01) or np.any(weights_neg <= 0.01):
        return np.inf
    if np.any(weights_pos > 100) or np.any(weights_neg > 100):
        return np.inf
    
    phi, t_arr = diff_weighted_chisq_cf_grid(weights_pos, weights_neg, location)
    pdf_vals = weighted_chisq_pdf_from_phi(data, phi, t_arr)
    
    if np.any(pdf_vals <= 0) or np.any(~np.isfinite(pdf_vals)):
        return np.inf
    
    return -np.sum(np.log(pdf_vals))

def fit_diff_weighted_chisq(data, K):
    """Fit difference of weighted chi-squared model via MLE.
    
    K = number of positive weights = number of negative weights.
    Total weights = 2*K, total params = 2*K + 1 (with location).
    """
    # Initialize weights based on data scale
    scale = np.std(data) / np.sqrt(2 * K)
    weights_pos_init = np.linspace(0.5, 1.5, K) * scale
    weights_neg_init = np.linspace(0.5, 1.5, K) * scale
    location_init = np.mean(data)
    
    x0 = np.concatenate([weights_pos_init, weights_neg_init, [location_init]])
    
    # Bounds: all weights must be positive
    w_max = 10.0
    bounds = [(0.01, w_max) for _ in range(2 * K)] + [(-np.inf, np.inf)]
    
    result = optimize.minimize(
        diff_weighted_chisq_loglik,
        x0=x0,
        args=(data, K),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000}
    )
    return result.x, -result.fun

# =============================================================================
# Noncentral t (location-scale)
# =============================================================================

def nct_pdf(x, df, nc, loc, scale):
    """PDF of noncentral t with location and scale."""
    z = (x - loc) / scale
    return stats.nct.pdf(z, df, nc) / scale

def nct_loglik(params, data):
    """Negative log-likelihood for NCT."""
    df, nc, loc, scale = params
    if df <= 0 or scale <= 0:
        return np.inf
    
    pdf_vals = nct_pdf(data, df, nc, loc, scale)
    if np.any(pdf_vals <= 0) or np.any(~np.isfinite(pdf_vals)):
        return np.inf
    
    return -np.sum(np.log(pdf_vals))

def fit_nct(data):
    """Fit noncentral t via MLE."""
    x0 = [5, 0, np.mean(data), np.std(data)]
    bounds = [(0.1, 100), (-10, 10), (-np.inf, np.inf), (1e-6, np.inf)]
    
    result = optimize.minimize(
        nct_loglik,
        x0=x0,
        args=(data,),
        method='L-BFGS-B',
        bounds=bounds
    )
    return result.x, -result.fun

# =============================================================================
# Model Fitting Orchestration
# =============================================================================

def fit_all_models(data, stock_id):
    """Fit all models to a single returns series."""
    results = {}
    
    print(f"  Fitting Gaussian...")
    params, loglik = fit_gaussian(data)
    results['Gaussian'] = {'params': params, 'loglik': loglik, 'n_params': 2}
    
    print(f"  Fitting Mixture Gaussian (K=2)...")
    params, loglik = fit_mixture_gaussian(data, K=2)
    results['MixGauss_K2'] = {'params': params, 'loglik': loglik, 'n_params': 5}
    
    print(f"  Fitting Mixture Gaussian (K=3)...")
    params, loglik = fit_mixture_gaussian(data, K=3)
    results['MixGauss_K3'] = {'params': params, 'loglik': loglik, 'n_params': 8}
    
    # Weighted Chi-Squared models
    for K in [2, 3, 4, 5]:
        print(f"  Fitting Weighted Chi-Squared (K={K})...")
        try:
            params, loglik = fit_weighted_chisq(data, K)
            results[f'WeightedChiSq_K{K}'] = {
                'params': params, 'loglik': loglik, 'n_params': K + 1
            }
        except Exception as e:
            print(f"    Warning: Failed - {e}")
            results[f'WeightedChiSq_K{K}'] = {
                'params': None, 'loglik': -np.inf, 'n_params': K + 1
            }
    
    # Difference of Weighted Chi-Squared models (K positive + K negative)
    for K in [3]:  # K=3 means 3+3=6 weights total
        print(f"  Fitting Diff Weighted Chi-Squared (K={K}+{K})...")
        try:
            params, loglik = fit_diff_weighted_chisq(data, K)
            results[f'DiffWeightedChiSq_K{K}'] = {
                'params': params, 'loglik': loglik, 'n_params': 2 * K + 1
            }
        except Exception as e:
            print(f"    Warning: Failed - {e}")
            results[f'DiffWeightedChiSq_K{K}'] = {
                'params': None, 'loglik': -np.inf, 'n_params': 2 * K + 1
            }
    
    print(f"  Fitting Noncentral t...")
    params, loglik = fit_nct(data)
    results['NCT'] = {'params': params, 'loglik': loglik, 'n_params': 4}
    
    return results

# =============================================================================
# AIC/BIC Computation
# =============================================================================

def compute_aic_bic(loglik, n_params, n_obs):
    """Compute AIC and BIC."""
    aic = 2 * n_params - 2 * loglik
    bic = n_params * np.log(n_obs) - 2 * loglik
    return aic, bic

# =============================================================================
# VaR Computation
# =============================================================================

def empirical_var(data, alpha=ALPHA):
    """Empirical alpha-quantile."""
    return np.percentile(data, alpha * 100)

def var_gaussian(params, alpha=ALPHA):
    """VaR for Gaussian."""
    mu, sigma = params
    return stats.norm.ppf(alpha, loc=mu, scale=sigma)

def mixture_gaussian_cdf(x, params, K):
    """CDF of K-component Gaussian mixture."""
    weights = np.zeros(K)
    weights[:-1] = params[:K-1]
    weights[-1] = 1 - np.sum(weights[:-1])
    
    mus = params[K-1:2*K-1]
    sigmas = params[2*K-1:3*K-1]
    
    cdf = 0.0
    for i in range(K):
        cdf += weights[i] * stats.norm.cdf(x, loc=mus[i], scale=sigmas[i])
    return cdf

def var_mixture_gaussian(params, K, alpha=ALPHA):
    """VaR for K-component Gaussian mixture."""
    weights = np.zeros(K)
    weights[:-1] = params[:K-1]
    weights[-1] = 1 - np.sum(weights[:-1])
    mus = params[K-1:2*K-1]
    sigmas = params[2*K-1:3*K-1]
    
    mean_approx = np.sum(weights * mus)
    var_approx = np.sum(weights * (sigmas**2 + (mus - mean_approx)**2))
    std_approx = np.sqrt(max(var_approx, 1e-8))
    
    lo = mean_approx - 10 * std_approx
    hi = mean_approx + 10 * std_approx
    
    def f(x):
        return mixture_gaussian_cdf(x, params, K) - alpha
    
    return optimize.brentq(f, lo, hi)

def weighted_chisq_cdf_from_phi(x_array, phi, t_arr):
    """CDF via Gil-Pelaez formula."""
    x_array = np.asarray(x_array)
    
    if GPU_AVAILABLE:
        x_gpu = cp.asarray(x_array)
        t_safe = cp.copy(t_arr)
        t_safe[0] = 1e-12
        phase = cp.exp(-1j * cp.outer(x_gpu, t_arr))
        integrand = cp.imag(phase * phi / t_safe)
        w = cp.ones_like(t_arr)
        w[0] = w[-1] = 0.5
        vals = cp.sum(integrand * w, axis=1) * dt
        return cp.asnumpy(0.5 - vals / np.pi)
    else:
        t_safe = np.copy(t_arr)
        t_safe[0] = 1e-12
        phase = np.exp(-1j * np.outer(x_array, t_arr))
        integrand = np.imag(phase * phi / t_safe)
        w = np.ones_like(t_arr)
        w[0] = w[-1] = 0.5
        vals = np.sum(integrand * w, axis=1) * dt
        return 0.5 - vals / np.pi

def var_weighted_chisq(params, K, alpha=ALPHA, n_sim=100000):
    """VaR for weighted chi-squared model via Monte Carlo simulation."""
    weights = np.asarray(params[:K])
    location = params[K]
    
    # Simulate chi-squared(1) random variables and apply weights
    np.random.seed(42)  # For reproducibility
    chi2_samples = np.random.chisquare(df=1, size=(n_sim, K))
    X = location + np.sum(weights * chi2_samples, axis=1)
    
    return np.percentile(X, alpha * 100)

def var_diff_weighted_chisq(params, K, alpha=ALPHA, n_sim=100000):
    """VaR for difference of weighted chi-squared model via Monte Carlo simulation."""
    weights_pos = np.asarray(params[:K])
    weights_neg = np.asarray(params[K:2*K])
    location = params[2*K]
    
    # Simulate chi-squared(1) random variables
    np.random.seed(42)  # For reproducibility
    chi2_pos = np.random.chisquare(df=1, size=(n_sim, K))
    chi2_neg = np.random.chisquare(df=1, size=(n_sim, K))
    
    # X = location + sum(w_pos * chi2_pos) - sum(w_neg * chi2_neg)
    X = location + np.sum(weights_pos * chi2_pos, axis=1) - np.sum(weights_neg * chi2_neg, axis=1)
    
    return np.percentile(X, alpha * 100)

def var_nct(params, alpha=ALPHA):
    """VaR for noncentral t."""
    df, nc, loc, scale = params
    q_std = stats.nct.ppf(alpha, df, nc)
    return loc + scale * q_std

def compute_var(data, model_results, alpha=ALPHA):
    """Compute VaR for all models."""
    var_dict = {'Empirical': empirical_var(data, alpha)}
    
    for name, res in model_results.items():
        params = res['params']
        if params is None:
            var_dict[name] = np.nan
            continue
        
        if name == 'Gaussian':
            var_dict[name] = var_gaussian(params, alpha)
        elif name.startswith('MixGauss_K'):
            K = int(name.split('K')[1])
            var_dict[name] = var_mixture_gaussian(params, K, alpha)
        elif name.startswith('WeightedChiSq_K'):
            K = int(name.split('K')[1])
            var_dict[name] = var_weighted_chisq(params, K, alpha)
        elif name.startswith('DiffWeightedChiSq_K'):
            K = int(name.split('K')[1])
            var_dict[name] = var_diff_weighted_chisq(params, K, alpha)
        elif name == 'NCT':
            var_dict[name] = var_nct(params, alpha)
        else:
            var_dict[name] = np.nan
    
    return var_dict

# =============================================================================
# Density Evaluation for Plotting
# =============================================================================

def evaluate_densities(x_range, model_results):
    """Evaluate all model densities on x_range for plotting."""
    densities = {}
    
    for name, res in model_results.items():
        params = res['params']
        if params is None:
            continue
        
        if name == 'Gaussian':
            densities[name] = gaussian_pdf(x_range, params[0], params[1])
        elif name.startswith('MixGauss_K'):
            K = int(name.split('K')[1])
            densities[name] = mixture_gaussian_pdf(x_range, params, K)
        elif name.startswith('WeightedChiSq_K'):
            K = int(name.split('K')[1])
            weights = params[:K]
            location = params[K]
            phi, t_arr = weighted_chisq_cf_grid(weights, location)
            densities[name] = weighted_chisq_pdf_from_phi(x_range, phi, t_arr)
        elif name.startswith('DiffWeightedChiSq_K'):
            K = int(name.split('K')[1])
            weights_pos = params[:K]
            weights_neg = params[K:2*K]
            location = params[2*K]
            phi, t_arr = diff_weighted_chisq_cf_grid(weights_pos, weights_neg, location)
            densities[name] = weighted_chisq_pdf_from_phi(x_range, phi, t_arr)
        elif name == 'NCT':
            densities[name] = nct_pdf(x_range, params[0], params[1], params[2], params[3])
    
    return densities

# =============================================================================
# Main Execution
# =============================================================================

def one_stock(returns):
    """Fit models for a single stock (first one)."""
    col = returns.columns[0]
    print(f"\n{'=' * 60}")
    print(f"Fitting models for Stock {col} (Single Stock Mode)")
    print(f"{'=' * 60}")
    
    data = returns[col].values
    n_obs = len(data)
    
    # Fit models
    results = fit_all_models(data, col)
    
    # Compute AIC/BIC
    for model in results:
        loglik = results[model]['loglik']
        n_params = results[model]['n_params']
        aic, bic = compute_aic_bic(loglik, n_params, n_obs)
        results[model]['AIC'] = aic
        results[model]['BIC'] = bic
        
    # Compute VaR
    print("\nComputing VaR...")
    var_dict = compute_var(data, results, alpha=ALPHA)
    
    print("\nResults Summary:")
    print(f"{'Model':<20} {'LogLik':<12} {'AIC':<12} {'BIC':<12} {'VaR (1%)':<12}")
    print("-" * 70)
    
    emp_var = var_dict['Empirical']
    print(f"{'Empirical':<20} {'-':<12} {'-':<12} {'-':<12} {emp_var:.4f}")
    
    for model, res in results.items():
        print(f"{model:<20} {res['loglik']:.2f}        {res['AIC']:.2f}        {res['BIC']:.2f}        {var_dict[model]:.4f}")

def all_stock(returns):
    """Fit models for all stocks and save results."""
    n_obs = len(returns)
    all_results = {}
    all_densities = {}
    
    start_time = time.time()
    
    for col in returns.columns:
        print(f"\n{'=' * 60}")
        print(f"Fitting models for Stock {col}")
        print(f"{'=' * 60}")
        
        data = returns[col].values
        all_results[col] = fit_all_models(data, col)
        
        # Compute AIC/BIC
        for model in all_results[col]:
            loglik = all_results[col][model]['loglik']
            n_params = all_results[col][model]['n_params']
            aic, bic = compute_aic_bic(loglik, n_params, n_obs)
            all_results[col][model]['AIC'] = aic
            all_results[col][model]['BIC'] = bic
        
        # Evaluate densities for plotting
        q_low, q_high = np.quantile(data, [0.01, 0.99])
        x_range = np.linspace(q_low, q_high, 500)
        all_densities[col] = {
            'x_range': x_range,
            'data': data,
            'densities': evaluate_densities(x_range, all_results[col])
        }
    
    elapsed = time.time() - start_time
    print(f"\nTotal fitting time: {elapsed:.2f} seconds")
    
    # Compute VaR table
    print("\nComputing VaR table...")
    var_table = []
    for stock in returns.columns:
        data = returns[stock].values
        var_dict = compute_var(data, all_results[stock], alpha=ALPHA)
        
        emp_var = var_dict['Empirical']
        model_names = [m for m in var_dict.keys() if m != 'Empirical']
        diffs = {m: abs(var_dict[m] - emp_var) for m in model_names if np.isfinite(var_dict[m])}
        best_model = min(diffs.items(), key=lambda x: x[1])[0] if diffs else None
        
        row = {
            'Stock': stock,
            'Empirical_VaR': emp_var,
            'Best_Model_VaR': best_model,
            **{f'VaR_{m}': var_dict[m] for m in model_names}
        }
        var_table.append(row)
    
    df_var = pd.DataFrame(var_table)
    
    # Create summary table
    print("\nCreating model comparison summary...")
    summary_rows = []
    for stock in returns.columns:
        row = {'Stock': stock}
        for model, res in all_results[stock].items():
            row[f'{model}_LogLik'] = res['loglik']
            row[f'{model}_AIC'] = res['AIC']
            row[f'{model}_BIC'] = res['BIC']
        
        # Find best by each criterion
        models = list(all_results[stock].keys())
        row['Best_LogLik'] = max(models, key=lambda m: all_results[stock][m]['loglik'] if np.isfinite(all_results[stock][m]['loglik']) else -np.inf)
        row['Best_AIC'] = min(models, key=lambda m: all_results[stock][m]['AIC'] if np.isfinite(all_results[stock][m]['AIC']) else np.inf)
        row['Best_BIC'] = min(models, key=lambda m: all_results[stock][m]['BIC'] if np.isfinite(all_results[stock][m]['BIC']) else np.inf)
        
        summary_rows.append(row)
    
    df_summary = pd.DataFrame(summary_rows)
    
    # Save results
    results = {
        'all_results': all_results,
        'all_densities': all_densities,
        'var_table': df_var,
        'summary_table': df_summary,
        'returns': returns,
        'n_obs': n_obs,
        'alpha': ALPHA
    }
    
    output_path = "results.pkl"
    print(f"\nSaving results to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    print("\n" + "=" * 60)
    print("DONE! Results saved to results.pkl")
    print("=" * 60)
    
    # Print summary
    print("\nBest models by criterion (per stock):")
    print(df_summary[['Stock', 'Best_LogLik', 'Best_AIC', 'Best_BIC']].to_string(index=False))
    
    print("\n\nVaR (1%) table:")
    print(df_var.to_string(index=False))

def main():
    print("=" * 60)
    print("GPU-Accelerated Stock Model Fitting")
    print("=" * 60)
    
    # Load data
    data_path = "DJIA30stockreturns.csv"
    print(f"\nLoading data from {data_path}...")
    returns = pd.read_csv(data_path, header=None)
    print(f"Data shape: {returns.shape}")
    
    # Choose mode
    # one_stock(returns)
    
    all_stock(returns)

if __name__ == "__main__":
    main()
