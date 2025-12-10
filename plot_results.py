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

def plot_densities_for_stock(stock, densities_info, output_dir="."):
    """Plot density comparison for a single stock."""
    data = densities_info['data']
    x_range = densities_info['x_range']
    densities = densities_info['densities']
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Kernel density estimate
    kde = gaussian_kde(data)
    ax.plot(x_range, kde(x_range), 'k-', linewidth=2.5, label='Kernel Density', alpha=0.8)
    
    # Color palette for models
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
    
    # Save figure
    fig_path = f"{output_dir}/density_stock_{stock}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return fig_path

def plot_all_densities(all_densities, output_dir="."):
    """Plot densities for all stocks."""
    paths = []
    for stock, info in all_densities.items():
        path = plot_densities_for_stock(stock, info, output_dir)
        paths.append(path)
        print(f"Saved: {path}")
    return paths

# =============================================================================
# Summary Tables
# =============================================================================

def create_model_comparison_table(all_results, returns):
    """Create table showing best model by LogLik, AIC, BIC for each stock."""
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
        
        rows.append({
            'Stock': stock,
            'Best (LogLik)': best_loglik,
            'Best (AIC)': best_aic,
            'Best (BIC)': best_bic
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
    """Heatmap comparing VaR estimates across stocks and models."""
    var_cols = [c for c in df_var.columns if c.startswith('VaR_')]
    
    # Prepare data for heatmap
    var_data = df_var[var_cols].copy()
    var_data.index = df_var['Stock']
    var_data.columns = [c.replace('VaR_', '').replace('_', ' ') for c in var_cols]
    
    # Add empirical VaR
    emp_var = df_var['Empirical_VaR'].values
    var_data.insert(0, 'Empirical', emp_var)
    
    fig, ax = plt.subplots(figsize=(16, 12))
    
    sns.heatmap(var_data.T, annot=True, fmt='.2f', cmap='RdYlBu_r',
                center=0, ax=ax, annot_kws={'size': 8})
    
    ax.set_xlabel('Stock', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    ax.set_title('1% VaR Comparison Across Stocks and Models', fontsize=14, fontweight='bold')
    
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
    
    # Plot densities for first stock only (as per assignment)
    first_stock = returns.columns[0]
    print(f"\nPlotting density for Stock {first_stock}...")
    plot_densities_for_stock(first_stock, all_densities[first_stock], output_dir)
    
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
