#!/usr/bin/env python3
"""
Plot Results from GPU-Accelerated Stock Model Fitting

Loads results.pkl and generates:
- Density overlay plots for each stock
- Model comparison tables (AIC, BIC, Log-Likelihood)
- VaR comparison table

Uses seaborn for visualization.
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

# Set seaborn style
sns.set_theme(style="whitegrid", palette="husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

# =============================================================================
# Load Results
# =============================================================================

def load_results(path="results.pkl"):
    """Load saved results from compute_models.py."""
    with open(path, 'rb') as f:
        return pickle.load(f)

# =============================================================================
# Density Plots
# =============================================================================

import os

def plot_first_stock_detailed(stock, densities_info, output_dir="."):
    """
    Create 3 detailed plots for the first stock:
    1. Full distribution
    2. Zoomed center 
    3. Right-hand side tail
    """
    data = densities_info['data']
    x_range = densities_info['x_range']
    densities = densities_info['densities']
    
    # Kernel density estimate
    kde = gaussian_kde(data)
    
    # Color palette for models
    colors = sns.color_palette("husl", len(densities))
    
    paths = []
    
    # --- Plot 1: Full Distribution ---
    fig, ax = plt.subplots(figsize=(14, 9))
    
    ax.plot(x_range, kde(x_range), 'k-', linewidth=2.5, label='Kernel Density', alpha=0.8)
    
    for i, (model_name, pdf_vals) in enumerate(densities.items()):
        label = model_name.replace('_', ' ').replace('MixGauss K', 'Mix Gauss K=')
        label = label.replace('WeightedChiSq K', 'Weighted χ² K=')
        ax.plot(x_range, pdf_vals, '--', linewidth=1.8, color=colors[i], label=label, alpha=0.85)
    
    ax.set_xlabel('Return (%)', fontsize=13)
    ax.set_ylabel('Density', fontsize=13)
    ax.set_title(f'Fitted Distributions for Stock {stock} - Full Distribution', fontsize=15, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = f"{output_dir}/stock_{stock}_full.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    paths.append(fig_path)
    print(f"Saved: {fig_path}")
    
    # --- Plot 2: Zoomed Center ---
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Zoom to center: typically between -2 and 2 std or similar
    center_mask = (x_range >= -3) & (x_range <= 3)
    x_center = x_range[center_mask]
    
    ax.plot(x_center, kde(x_center), 'k-', linewidth=2.5, label='Kernel Density', alpha=0.8)
    
    for i, (model_name, pdf_vals) in enumerate(densities.items()):
        label = model_name.replace('_', ' ').replace('MixGauss K', 'Mix Gauss K=')
        label = label.replace('WeightedChiSq K', 'Weighted χ² K=')
        ax.plot(x_center, pdf_vals[center_mask], '--', linewidth=1.8, color=colors[i], label=label, alpha=0.85)
    
    ax.set_xlabel('Return (%)', fontsize=13)
    ax.set_ylabel('Density', fontsize=13)
    ax.set_title(f'Fitted Distributions for Stock {stock} - Center (Zoomed)', fontsize=15, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-3, 3)
    
    plt.tight_layout()
    fig_path = f"{output_dir}/stock_{stock}_center.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    paths.append(fig_path)
    print(f"Saved: {fig_path}")
    
    # --- Plot 3: Right-Hand Side Tail ---
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # RHS tail: values > 1.5 std (approximately)
    rhs_mask = x_range >= 1.5
    x_rhs = x_range[rhs_mask]
    
    ax.plot(x_rhs, kde(x_rhs), 'k-', linewidth=2.5, label='Kernel Density', alpha=0.8)
    
    for i, (model_name, pdf_vals) in enumerate(densities.items()):
        label = model_name.replace('_', ' ').replace('MixGauss K', 'Mix Gauss K=')
        label = label.replace('WeightedChiSq K', 'Weighted χ² K=')
        ax.plot(x_rhs, pdf_vals[rhs_mask], '--', linewidth=1.8, color=colors[i], label=label, alpha=0.85)
    
    ax.set_xlabel('Return (%)', fontsize=13)
    ax.set_ylabel('Density', fontsize=13)
    ax.set_title(f'Fitted Distributions for Stock {stock} - Right Tail', fontsize=15, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = f"{output_dir}/stock_{stock}_rhs_tail.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    paths.append(fig_path)
    print(f"Saved: {fig_path}")
    
    return paths


def plot_stock_full(stock, densities_info, output_dir="stocks"):
    """
    Plot a single stock's full distribution with all model estimates.
    Uses same styling as the first stock plots.
    Saves to the stocks/ subdirectory.
    """
    data = densities_info['data']
    x_range = densities_info['x_range']
    densities = densities_info['densities']
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Kernel density estimate of the data
    kde = gaussian_kde(data)
    ax.plot(x_range, kde(x_range), 'k-', linewidth=2.5, label='Kernel Density', alpha=0.8)
    
    # Color palette for models - same as first stock
    colors = sns.color_palette("husl", len(densities))
    
    # Plot each model density
    for i, (model_name, pdf_vals) in enumerate(densities.items()):
        label = model_name.replace('_', ' ').replace('MixGauss K', 'Mix Gauss K=')
        label = label.replace('WeightedChiSq K', 'Weighted χ² K=')
        ax.plot(x_range, pdf_vals, '--', linewidth=1.8, color=colors[i], label=label, alpha=0.85)
    
    ax.set_xlabel('Return (%)', fontsize=13)
    ax.set_ylabel('Density', fontsize=13)
    ax.set_title(f'Fitted Distributions for Stock {stock}', fontsize=15, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    fig_path = f"{output_dir}/stock_{stock}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return fig_path


def plot_all_densities(all_densities, output_dir="."):
    """Plot densities for all stocks."""
    paths = []
    for stock, info in all_densities.items():
        path = plot_stock_full(stock, info, output_dir)
        paths.append(path)
        print(f"Saved: {path}")
    return paths

# =============================================================================
# Summary Tables
# =============================================================================

def create_model_comparison_table(all_results, returns):
    """Create table showing best model by LogLik, AIC, BIC for each stock with values."""
    rows = []
    
    for stock in returns.columns:
        stock_res = all_results[stock]
        models = list(stock_res.keys())
        
        # Extract metrics
        logliks = {m: stock_res[m]['loglik'] for m in models}
        aics = {m: stock_res[m]['AIC'] for m in models}
        bics = {m: stock_res[m]['BIC'] for m in models}
        
        # Find best
        valid_logliks = {m: v for m, v in logliks.items() if np.isfinite(v)}
        valid_aics = {m: v for m, v in aics.items() if np.isfinite(v)}
        valid_bics = {m: v for m, v in bics.items() if np.isfinite(v)}
        
        best_loglik = max(valid_logliks, key=valid_logliks.get) if valid_logliks else 'N/A'
        best_aic = min(valid_aics, key=valid_aics.get) if valid_aics else 'N/A'
        best_bic = min(valid_bics, key=valid_bics.get) if valid_bics else 'N/A'
        
        # Get values for best models
        best_loglik_val = valid_logliks.get(best_loglik, np.nan) if best_loglik != 'N/A' else np.nan
        best_aic_val = valid_aics.get(best_aic, np.nan) if best_aic != 'N/A' else np.nan
        best_bic_val = valid_bics.get(best_bic, np.nan) if best_bic != 'N/A' else np.nan
        
        rows.append({
            'Stock': stock,
            'Best (LogLik)': best_loglik,
            'LogLik Value': f"{best_loglik_val:.2f}" if np.isfinite(best_loglik_val) else '--',
            'Best (AIC)': best_aic,
            'AIC Value': f"{best_aic_val:.2f}" if np.isfinite(best_aic_val) else '--',
            'Best (BIC)': best_bic,
            'BIC Value': f"{best_bic_val:.2f}" if np.isfinite(best_bic_val) else '--'
        })
    
    return pd.DataFrame(rows)

def create_detailed_metrics_table(all_results, returns):
    """Create detailed table with all metrics for all models."""
    rows = []
    
    for stock in returns.columns:
        stock_res = all_results[stock]
        
        for model, res in stock_res.items():
            rows.append({
                'Stock': stock,
                'Model': model,
                'Log-Likelihood': res['loglik'],
                'AIC': res['AIC'],
                'BIC': res['BIC'],
                'n_params': res['n_params']
            })
    
    return pd.DataFrame(rows)

def plot_model_frequency_chart(comparison_df, output_dir="."):
    """Bar chart showing how often each model is best by each criterion."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    criteria = ['Best (LogLik)', 'Best (AIC)', 'Best (BIC)']
    titles = ['Best by Log-Likelihood', 'Best by AIC', 'Best by BIC']
    
    for ax, crit, title in zip(axes, criteria, titles):
        counts = comparison_df[crit].value_counts()
        colors = sns.color_palette("husl", len(counts))
        
        bars = ax.bar(range(len(counts)), counts.values, color=colors)
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Add value labels on bars
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                   str(val), ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    fig_path = f"{output_dir}/model_frequency.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_path}")
    return fig_path

def plot_chisq_comparison(stock, densities_info, output_dir="."):
    """
    Compare WeightedChiSq and DiffWeightedChiSq models for a stock.
    Shows the difference in how these models fit the data.
    """
    data = densities_info['data']
    x_range = densities_info['x_range']
    densities = densities_info['densities']
    
    # Filter for chi-squared models only
    weighted_models = {k: v for k, v in densities.items() if k.startswith('WeightedChiSq_K')}
    diff_models = {k: v for k, v in densities.items() if k.startswith('DiffWeightedChiSq_K')}
    
    if not weighted_models and not diff_models:
        return None
    
    # Kernel density estimate
    kde = gaussian_kde(data)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # --- Left Panel: WeightedChiSq (negative weights allowed) ---
    ax1 = axes[0]
    ax1.plot(x_range, kde(x_range), 'k-', linewidth=2.5, label='Kernel Density', alpha=0.8)
    
    colors_w = sns.color_palette("Blues_d", len(weighted_models))
    for i, (model_name, pdf_vals) in enumerate(weighted_models.items()):
        K = model_name.split('K')[1]
        label = f'Weighted χ² K={K}'
        # Clip negative values for plotting (numerical artifacts)
        pdf_clipped = np.clip(pdf_vals, 0, None)
        ax1.plot(x_range, pdf_clipped, '--', linewidth=1.8, color=colors_w[i], label=label, alpha=0.85)
    
    ax1.set_xlabel('Return (%)', fontsize=13)
    ax1.set_ylabel('Density', fontsize=13)
    ax1.set_title(f'Weighted Chi-Squared (neg weights allowed)', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    # --- Right Panel: DiffWeightedChiSq (positive weights, sign in CF) ---
    ax2 = axes[1]
    ax2.plot(x_range, kde(x_range), 'k-', linewidth=2.5, label='Kernel Density', alpha=0.8)
    
    colors_d = sns.color_palette("Oranges_d", len(diff_models))
    for i, (model_name, pdf_vals) in enumerate(diff_models.items()):
        K = model_name.split('K')[1]
        label = f'Diff Weighted χ² K={K}+{K}'
        pdf_clipped = np.clip(pdf_vals, 0, None)
        ax2.plot(x_range, pdf_clipped, '--', linewidth=1.8, color=colors_d[i], label=label, alpha=0.85)
    
    ax2.set_xlabel('Return (%)', fontsize=13)
    ax2.set_ylabel('Density', fontsize=13)
    ax2.set_title(f'Difference of Weighted Chi-Squared', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)
    
    fig.suptitle(f'Chi-Squared Model Comparison for Stock {stock}', fontsize=15, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    fig_path = f"{output_dir}/stock_{stock}_chisq_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_path}")
    return fig_path

# =============================================================================
# VaR Table
# =============================================================================

def format_var_table(df_var):
    """Format VaR table for display and LaTeX output."""
    # Get VaR columns
    var_cols = [c for c in df_var.columns if c.startswith('VaR_')]
    emp_col = 'Empirical_VaR'
    
    # Create formatted output
    formatted_rows = []
    for _, row in df_var.iterrows():
        stock = row['Stock']
        emp_var = row[emp_col]
        best_model = row['Best_Model_VaR']
        
        fmt_row = {'Stock': stock, 'Empirical': f"{emp_var:.3f}"}
        for col in var_cols:
            val = row[col]
            model = col.replace('VaR_', '')
            if np.isnan(val):
                fmt_row[model] = '--'
            elif model == best_model:
                fmt_row[model] = f"**{val:.3f}**"  # Bold for best
            else:
                fmt_row[model] = f"{val:.3f}"
        
        formatted_rows.append(fmt_row)
    
    return pd.DataFrame(formatted_rows)

def plot_var_heatmap(df_var, output_dir="."):
    """Heatmap showing distance from empirical VaR for each model, with best model highlighted."""
    var_cols = [c for c in df_var.columns if c.startswith('VaR_')]
    
    # Get empirical VaR for each stock
    emp_var = df_var['Empirical_VaR'].values
    
    # Prepare data - compute absolute distance from empirical VaR
    var_data = df_var[var_cols].copy()
    for i, col in enumerate(var_cols):
        var_data[col] = np.abs(df_var[col].values - emp_var)
    
    var_data.index = df_var['Stock']
    var_data.columns = [c.replace('VaR_', '').replace('_', ' ') for c in var_cols]
    
    # Find minima per stock (axis=1 for each row)
    min_per_stock = var_data.idxmin(axis=1)  # Model name with minimum distance
    
    fig, ax = plt.subplots(figsize=(18, 10))
    
    # Transposed view (models on y-axis, stocks on x-axis)
    heatmap_data = var_data.T
    
    # Use a diverging colormap - green for low (good), red for high (bad)
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn_r',
                ax=ax, annot_kws={'size': 9, 'weight': 'bold'},
                linewidths=0.5, linecolor='white',
                cbar_kws={'label': 'Absolute Distance from Empirical VaR', 'shrink': 0.8})
    
    # Add thick borders around the best model for each stock
    from matplotlib.patches import Rectangle
    for stock_idx, stock in enumerate(var_data.index):
        best_model = min_per_stock[stock]
        model_idx = list(heatmap_data.index).index(best_model)
        
        # Draw a thick rectangle border around the best cell
        rect = Rectangle((stock_idx, model_idx), 1, 1, 
                         fill=False, edgecolor='blue', linewidth=3)
        ax.add_patch(rect)
    
    ax.set_xlabel('Stock', fontsize=13, fontweight='bold')
    ax.set_ylabel('Model', fontsize=13, fontweight='bold')
    ax.set_title('Distance from Empirical 1% VaR\n(Green = Close, Red = Far, Blue Border = Best per Stock)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    fig_path = f"{output_dir}/var_heatmap.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_path}")
    return fig_path

# =============================================================================
# Main Execution
# =============================================================================

def main():
    print("=" * 60)
    print("Generating Plots from Saved Results")
    print("=" * 60)
    
    # Load results
    print("\nLoading results.pkl...")
    try:
        results = load_results("results.pkl")
    except FileNotFoundError:
        print("Error: results.pkl not found. Run compute_models.py first.")
        return
    
    all_results = results['all_results']
    all_densities = results['all_densities']
    df_var = results['var_table']
    returns = results['returns']
    
    output_dir = "."
    
    # Plot 3 detailed views for the first stock
    first_stock = returns.columns[0]
    print(f"\nPlotting detailed views for Stock {first_stock}...")
    plot_first_stock_detailed(first_stock, all_densities[first_stock], output_dir)
    
    # Plot chi-squared model comparison for first stock
    print(f"\nPlotting chi-squared model comparison for Stock {first_stock}...")
    plot_chisq_comparison(first_stock, all_densities[first_stock], output_dir)
    
    # Plot remaining 25 stocks with normal comparison, save to stocks/ folder
    stocks_dir = os.path.join(output_dir, "stocks")
    os.makedirs(stocks_dir, exist_ok=True)
    print(f"\nPlotting remaining stocks with normal comparison to {stocks_dir}/...")
    remaining_stocks = returns.columns[1:]
    for stock in remaining_stocks:
        path = plot_stock_full(stock, all_densities[stock], stocks_dir)
        print(f"  Saved: {path}")
    
    # Create model comparison tables
    print("\nCreating model comparison tables...")
    comparison_df = create_model_comparison_table(all_results, returns)
    print("\nBest Model by Criterion:")
    print(comparison_df.to_string(index=False))
    
    # Plot model frequency chart
    plot_model_frequency_chart(comparison_df, output_dir)
    
    # Format and print VaR table
    print("\n\n1% VaR Table (** = best model for each stock):")
    formatted_var = format_var_table(df_var)
    print(formatted_var.to_string(index=False))
    
    # VaR heatmap
    print("\nCreating VaR heatmap...")
    plot_var_heatmap(df_var, output_dir)
    
    # Generate LaTeX tables
    print("\n" + "=" * 60)
    print("LaTeX Tables")
    print("=" * 60)
    
    # Model comparison LaTeX
    print("\n% Model Comparison Table")
    print(comparison_df.to_latex(index=False, escape=True, caption="Best model by criterion for each stock"))
    
    print("\nDONE!")

if __name__ == "__main__":
    main()
