#!/usr/bin/env python3
"""
Chi-Squared Model Investigation

This script investigates three different chi-squared model formulations:
1. Weighted Chi-Squared (allows positive AND negative weights)
2. Positive-Only Weighted Chi-Squared (only positive weights)
3. Difference of Chi-Squared (explicit positive and negative terms)

The goal is to understand why the difference formulation allows the location
parameter to move freely while the positive-only version does not.

Target: First stock (column 0) from DJIA30stockreturns.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, optimize
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# Constants
# =============================================================================
TMAX = 100.0   # Max t for characteristic function integration
N_T = 8000     # Number of points in t-grid

# Global t-grid
t_grid = np.linspace(0.0, TMAX, N_T)
dt = t_grid[1] - t_grid[0]

# =============================================================================
# Shared PDF from Characteristic Function
# =============================================================================

def weighted_chisq_pdf_from_phi(x_array, phi):
    """
    Given data x_array and phi(t_grid),
    approximate the pdf at all x via numeric integration.
    
    Uses Gil-Pelaez formula:
    f(x) = (1/π) * Re[ ∫_0^∞ exp(-itx) * φ(t) dt ]
    """
    x_array = np.asarray(x_array)
    phase = np.exp(-1j * np.outer(x_array, t_grid))
    integrand = phase * phi
    weights = np.ones(N_T)
    weights[0] = weights[-1] = 0.5
    vals = np.real(np.sum(integrand * weights, axis=1) * dt)
    return vals / np.pi

# =============================================================================
# 1. Weighted Chi-Squared (Mixed Signs Allowed)
# =============================================================================

def weighted_chisq_cf_grid(weights, location):
    """
    Characteristic function of weighted sum of chi-squared(1) + location
    X = location + sum(weights[k] * Z_k) where Z_k ~ chi^2(1)
    
    ALLOWS NEGATIVE weights (which gives a flipped chi-squared contribution)
    
    phi_X(t) = exp(i*t*location) * Prod(1 - 2*i*t*weights[j])^(-1/2)
    """
    t = t_grid
    w = np.asarray(weights)
    z = 1 - 2j * np.outer(t, w)
    log_phi = 1j * t * location - 0.5 * np.sum(np.log(z), axis=1)
    return np.exp(log_phi)


def weighted_chisq_loglik(params, data, K):
    """Negative log-likelihood for mixed-sign weighted chi-squared."""
    weights = params[:K]
    location = params[K]
    
    phi = weighted_chisq_cf_grid(weights, location)
    pdf_vals = weighted_chisq_pdf_from_phi(data, phi)
    
    if np.any(pdf_vals <= 0) or np.any(~np.isfinite(pdf_vals)):
        return np.inf
    
    return -np.sum(np.log(pdf_vals))


def fit_weighted_chisq(data, K):
    """
    Fit weighted chi-squared model via MLE.
    ALLOWS negative weights (mixed signs).
    """
    # Initialize with negative and positive weights
    K_half = K // 2
    weights_init = np.concatenate([
        -np.linspace(1, 0.5, K_half),
        np.linspace(0.5, 1, K - K_half)
    ])
    weights_init /= np.std(weights_init)
    location_init = np.mean(data)
    
    x0 = np.concatenate([weights_init, [location_init]])
    
    w_max = 5.0
    bounds = [(-w_max, w_max) for _ in range(K)] + [(-np.inf, np.inf)]
    
    result = optimize.minimize(
        weighted_chisq_loglik,
        x0=x0,
        args=(data, K),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 100, 'disp': True}
    )
    
    return result.x, -result.fun

# =============================================================================
# 2. Positive-Only Weighted Chi-Squared
# =============================================================================

def positive_weighted_chisq_cf_grid(weights, location):
    """
    Characteristic function of weighted sum of chi-squared(1) + location
    X = location + sum(weights[k] * Z_k) where Z_k ~ chi^2(1)
    
    ALL weights must be POSITIVE.
    
    phi_X(t) = exp(i*t*location) * Prod(1 - 2*i*t*weights[j])^(-1/2)
    """
    t = t_grid
    w = np.asarray(weights)
    z = 1 - 2j * np.outer(t, w)
    log_phi = 1j * t * location - 0.5 * np.sum(np.log(z), axis=1)
    return np.exp(log_phi)


def positive_weighted_chisq_loglik(params, data, K):
    """Negative log-likelihood for positive-only weighted chi-squared."""
    weights = params[:K]
    location = params[K]
    
    # Ensure all weights are positive (handled by bounds, but double-check)
    if np.any(weights <= 0):
        return np.inf
    
    phi = positive_weighted_chisq_cf_grid(weights, location)
    pdf_vals = weighted_chisq_pdf_from_phi(data, phi)
    
    if np.any(pdf_vals <= 0) or np.any(~np.isfinite(pdf_vals)):
        return np.inf
    
    return -np.sum(np.log(pdf_vals))


def fit_positive_weighted_chisq(data, K):
    """
    Fit weighted chi-squared model via MLE.
    ALL weights constrained to be POSITIVE.
    """
    # Initialize all positive weights
    weights_init = np.linspace(0.5, 1.5, K)
    scale = np.std(data) / (K ** 0.5)
    weights_init *= scale
    location_init = np.mean(data)
    
    x0 = np.concatenate([weights_init, [location_init]])
    
    w_max = 10.0
    # All weights bounded to be positive
    bounds = [(0.01, w_max) for _ in range(K)] + [(-np.inf, np.inf)]
    
    result = optimize.minimize(
        positive_weighted_chisq_loglik,
        x0=x0,
        args=(data, K),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 100, 'disp': True}
    )
    
    return result.x, -result.fun

# =============================================================================
# 3. Difference of Chi-Squared
# =============================================================================

def difference_weighted_chisq_cf_grid(weights_pos, weights_neg, location):
    """
    Characteristic function for: X = location + Σ a⁺·Z⁺ - Σ a⁻·Z⁻
    
    φ_X(t) = exp(it·location) · Π(1-2it·a⁺)^(-1/2) · Π(1+2it·a⁻)^(-1/2)
    
    Note the sign flip: for the negative part we have (1 + 2it·a⁻)
    """
    t = t_grid
    w_pos = np.asarray(weights_pos)
    w_neg = np.asarray(weights_neg)
    
    # Positive part: 1 - 2i·a⁺·t
    z_pos = 1 - 2j * np.outer(t, w_pos)
    
    # Negative part: 1 + 2i·a⁻·t (note the PLUS)
    z_neg = 1 + 2j * np.outer(t, w_neg)
    
    log_phi = (1j * t * location
               - 0.5 * np.sum(np.log(z_pos), axis=1)
               - 0.5 * np.sum(np.log(z_neg), axis=1))
    
    return np.exp(log_phi)


def difference_weighted_chisq_loglik(params, data, K_pos, K_neg):
    """Negative log-likelihood for difference model."""
    weights_pos = params[:K_pos]
    weights_neg = params[K_pos:K_pos+K_neg]
    location = params[-1]
    
    phi = difference_weighted_chisq_cf_grid(weights_pos, weights_neg, location)
    pdf_vals = weighted_chisq_pdf_from_phi(data, phi)
    
    if np.any(pdf_vals <= 0) or np.any(~np.isfinite(pdf_vals)):
        return np.inf
    
    return -np.sum(np.log(pdf_vals))


def fit_difference_weighted_chisq(data, K_pos, K_neg):
    """Fit difference of weighted chi-squared model via MLE."""
    # Initialize all weights positive
    weights_pos_init = np.linspace(0.5, 1.5, K_pos)
    weights_neg_init = np.linspace(0.5, 1.5, K_neg)
    
    # Scale based on data
    scale = np.std(data) / (K_pos + K_neg)**0.5
    weights_pos_init *= scale
    weights_neg_init *= scale
    
    location_init = np.mean(data)
    
    x0 = np.concatenate([weights_pos_init, weights_neg_init, [location_init]])
    
    # All weights must be positive
    w_max = 10.0
    bounds = ([(0.01, w_max) for _ in range(K_pos)] +
              [(0.01, w_max) for _ in range(K_neg)] +
              [(-np.inf, np.inf)])
    
    result = optimize.minimize(
        difference_weighted_chisq_loglik,
        x0=x0,
        args=(data, K_pos, K_neg),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 100, 'disp': True}
    )
    
    return result.x, -result.fun

# =============================================================================
# Main Investigation
# =============================================================================

def main():
    print("=" * 70)
    print("Chi-Squared Model Investigation")
    print("=" * 70)
    
    # Load data
    returns = pd.read_csv("DJIA30stockreturns.csv", header=None)
    data = returns[0].values  # First stock only
    
    print(f"\nData: First stock from DJIA30")
    print(f"  Shape: {data.shape}")
    print(f"  Mean: {np.mean(data):.6f}")
    print(f"  Std: {np.std(data):.6f}")
    print(f"  Min: {np.min(data):.6f}")
    print(f"  Max: {np.max(data):.6f}")
    
    K = 6  # Total number of chi-squared components
    K_pos = K // 2
    K_neg = K - K_pos
    
    results = {}
    
    # ---------------------------------------------------------------------------
    # 1. Mixed-Sign Weighted Chi-Squared
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("1. MIXED-SIGN WEIGHTED CHI-SQUARED (K=6)")
    print("   Weights can be positive OR negative")
    print("=" * 70 + "\n")
    
    params_mixed, loglik_mixed = fit_weighted_chisq(data, K)
    
    weights_mixed = params_mixed[:K]
    location_mixed = params_mixed[-1]
    
    print(f"\nFitted Parameters:")
    print(f"  Weights: {weights_mixed}")
    print(f"  Location: {location_mixed:.6f}")
    print(f"  Log-likelihood: {loglik_mixed:.2f}")
    
    results['mixed'] = {
        'params': params_mixed,
        'loglik': loglik_mixed,
        'weights': weights_mixed,
        'location': location_mixed,
        'K': K
    }
    
    # ---------------------------------------------------------------------------
    # 2. Positive-Only Weighted Chi-Squared
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("2. POSITIVE-ONLY WEIGHTED CHI-SQUARED (K=6)")
    print("   All weights must be positive")
    print("=" * 70 + "\n")
    
    params_positive, loglik_positive = fit_positive_weighted_chisq(data, K)
    
    weights_positive = params_positive[:K]
    location_positive = params_positive[-1]
    
    print(f"\nFitted Parameters:")
    print(f"  Weights: {weights_positive}")
    print(f"  Location: {location_positive:.6f}")
    print(f"  Log-likelihood: {loglik_positive:.2f}")
    
    results['positive_only'] = {
        'params': params_positive,
        'loglik': loglik_positive,
        'weights': weights_positive,
        'location': location_positive,
        'K': K
    }
    
    # ---------------------------------------------------------------------------
    # 3. Difference of Chi-Squared
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"3. DIFFERENCE OF CHI-SQUARED (K_pos={K_pos}, K_neg={K_neg})")
    print("   X = location + Σ a⁺·Z⁺ - Σ a⁻·Z⁻")
    print("=" * 70 + "\n")
    
    params_diff, loglik_diff = fit_difference_weighted_chisq(data, K_pos, K_neg)
    
    weights_pos_diff = params_diff[:K_pos]
    weights_neg_diff = params_diff[K_pos:K_pos+K_neg]
    location_diff = params_diff[-1]
    
    print(f"\nFitted Parameters:")
    print(f"  Positive weights: {weights_pos_diff}")
    print(f"  Negative weights: {weights_neg_diff}")
    print(f"  Location: {location_diff:.6f}")
    print(f"  Log-likelihood: {loglik_diff:.2f}")
    
    results['difference'] = {
        'params': params_diff,
        'loglik': loglik_diff,
        'weights_pos': weights_pos_diff,
        'weights_neg': weights_neg_diff,
        'location': location_diff,
        'K_pos': K_pos,
        'K_neg': K_neg
    }
    
    # ---------------------------------------------------------------------------
    # Summary Comparison
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    
    print(f"\n{'Model':<30} {'Location':<15} {'Log-Likelihood':<15}")
    print("-" * 60)
    print(f"{'Mixed-Sign Chi-Sq':<30} {location_mixed:<15.6f} {loglik_mixed:<15.2f}")
    print(f"{'Positive-Only Chi-Sq':<30} {location_positive:<15.6f} {loglik_positive:<15.2f}")
    print(f"{'Difference Chi-Sq':<30} {location_diff:<15.6f} {loglik_diff:<15.2f}")
    
    print(f"\nData mean for reference: {np.mean(data):.6f}")
    
    # ---------------------------------------------------------------------------
    # Generate and Save Plots
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    
    # Create x-range for PDF evaluation
    x_plot = np.linspace(data.min() - 1, data.max() + 1, 500)
    
    # Compute PDFs
    phi_mixed = weighted_chisq_cf_grid(weights_mixed, location_mixed)
    pdf_mixed = weighted_chisq_pdf_from_phi(x_plot, phi_mixed)
    
    phi_positive = positive_weighted_chisq_cf_grid(weights_positive, location_positive)
    pdf_positive = weighted_chisq_pdf_from_phi(x_plot, phi_positive)
    
    phi_diff = difference_weighted_chisq_cf_grid(weights_pos_diff, weights_neg_diff, location_diff)
    pdf_diff = weighted_chisq_pdf_from_phi(x_plot, phi_diff)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Mixed-Sign
    ax1 = axes[0]
    ax1.hist(data, bins=50, density=True, alpha=0.6, color='steelblue', edgecolor='black', label='Data')
    ax1.plot(x_plot, pdf_mixed, 'r-', linewidth=2.5, label=f'Mixed-Sign χ² (K={K})')
    ax1.axvline(location_mixed, color='red', linestyle='--', alpha=0.7, label=f'Location={location_mixed:.3f}')
    ax1.axvline(np.mean(data), color='green', linestyle=':', alpha=0.7, label=f'Data Mean={np.mean(data):.3f}')
    ax1.set_xlabel('Returns', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Mixed-Sign Weighted χ²', fontsize=14)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. Positive-Only
    ax2 = axes[1]
    ax2.hist(data, bins=50, density=True, alpha=0.6, color='steelblue', edgecolor='black', label='Data')
    ax2.plot(x_plot, pdf_positive, 'orange', linewidth=2.5, label=f'Positive-Only χ² (K={K})')
    ax2.axvline(location_positive, color='orange', linestyle='--', alpha=0.7, label=f'Location={location_positive:.3f}')
    ax2.axvline(np.mean(data), color='green', linestyle=':', alpha=0.7, label=f'Data Mean={np.mean(data):.3f}')
    ax2.set_xlabel('Returns', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Positive-Only Weighted χ²', fontsize=14)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Difference
    ax3 = axes[2]
    ax3.hist(data, bins=50, density=True, alpha=0.6, color='steelblue', edgecolor='black', label='Data')
    ax3.plot(x_plot, pdf_diff, 'purple', linewidth=2.5, label=f'Difference χ² ({K_pos}+{K_neg})')
    ax3.axvline(location_diff, color='purple', linestyle='--', alpha=0.7, label=f'Location={location_diff:.3f}')
    ax3.axvline(np.mean(data), color='green', linestyle=':', alpha=0.7, label=f'Data Mean={np.mean(data):.3f}')
    ax3.set_xlabel('Returns', fontsize=12)
    ax3.set_ylabel('Density', fontsize=12)
    ax3.set_title('Difference of χ²', fontsize=14)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('chi_squared_comparison.png', dpi=150, bbox_inches='tight')
    print("\nSaved: chi_squared_comparison.png")
    
    # Additional plot: All three on same axes
    fig2, ax = plt.subplots(figsize=(12, 7))
    ax.hist(data, bins=50, density=True, alpha=0.5, color='steelblue', edgecolor='black', label='Data Histogram')
    ax.plot(x_plot, pdf_mixed, 'r-', linewidth=2, label=f'Mixed-Sign (LL={loglik_mixed:.0f})')
    ax.plot(x_plot, pdf_positive, 'orange', linewidth=2, label=f'Positive-Only (LL={loglik_positive:.0f})')
    ax.plot(x_plot, pdf_diff, 'purple', linewidth=2, label=f'Difference (LL={loglik_diff:.0f})')
    ax.axvline(np.mean(data), color='green', linestyle=':', linewidth=2, label=f'Data Mean')
    ax.set_xlabel('Returns', fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.set_title('Chi-Squared Model Comparison (First Stock)', fontsize=16)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('chi_squared_overlay.png', dpi=150, bbox_inches='tight')
    print("Saved: chi_squared_overlay.png")
    
    plt.show()
    
    print("\n" + "=" * 70)
    print("INVESTIGATION COMPLETE")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    results = main()
