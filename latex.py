#!/usr/bin/env python3
"""
Generate LaTeX Tables from Saved Results

Outputs:
1. VaR table with stocks as columns, models as rows
   - Empirical VaR in first row
   - Bold for model closest to empirical VaR (per stock)
   
2. Model comparison table (best by LogLik, AIC, BIC)
"""

import numpy as np
import pandas as pd
import pickle

# =============================================================================
# Load Results
# =============================================================================

def load_results(path="results.pkl"):
    """Load saved results from compute_models.py."""
    with open(path, 'rb') as f:
        return pickle.load(f)

# =============================================================================
# LaTeX VaR Table
# =============================================================================

def generate_var_latex(df_var):
    """
    Generate LaTeX VaR table:
    - Rows = models
    - Columns = stocks  
    - Empirical VaR in first row
    - Bold = model closest to empirical VaR for each stock
    """
    stocks = df_var['Stock'].tolist()
    
    # VaR columns (excluding Best_Model_VaR)
    var_cols = [c for c in df_var.columns if 'VaR' in c and c != 'Best_Model_VaR']
    emp_col = 'Empirical_VaR'
    model_cols = [c for c in var_cols if c != emp_col]
    
    # LaTeX row labels
    row_labels = {
        'Empirical_VaR': r'Empirical',
        'VaR_Gaussian': r'Gaussian',
        'VaR_MixGauss_K2': r'Mix Gauss $K=2$',
        'VaR_MixGauss_K3': r'Mix Gauss $K=3$',
        'VaR_WeightedChiSq_K2': r'Weighted $\chi^2$, $K=2$',
        'VaR_WeightedChiSq_K3': r'Weighted $\chi^2$, $K=3$',
        'VaR_WeightedChiSq_K4': r'Weighted $\chi^2$, $K=4$',
        'VaR_WeightedChiSq_K5': r'Weighted $\chi^2$, $K=5$',
        'VaR_NCT': r'Noncentral $t$'
    }
    
    def get_label(colname):
        return row_labels.get(colname, colname.replace('_', r'\_'))
    
    def fmt(x):
        if np.isnan(x):
            return r'--'
        return f'{x:.3f}'
    
    # Find closest model to empirical for each stock
    closest = {}
    for _, row in df_var.iterrows():
        stock = row['Stock']
        emp = row[emp_col]
        best_model = None
        best_diff = np.inf
        for mc in model_cols:
            val = row[mc]
            if not np.isnan(val):
                diff = abs(val - emp)
                if diff < best_diff:
                    best_diff = diff
                    best_model = mc
        closest[stock] = best_model
    
    # Build LaTeX
    lines = []
    col_spec = 'l' + 'r' * len(stocks)
    
    lines.append(r'\begin{table}[ht]')
    lines.append(r'\centering')
    lines.append(r'\small')
    lines.append(r'\begin{tabular}{' + col_spec + r'}')
    lines.append(r'\toprule')
    
    # Header
    header = ['Model'] + [str(s) for s in stocks]
    lines.append(' & '.join(header) + r' \\')
    lines.append(r'\midrule')
    
    # Empirical row
    row_vals = [get_label(emp_col)]
    for _, row in df_var.iterrows():
        row_vals.append(fmt(row[emp_col]))
    lines.append(' & '.join(row_vals) + r' \\')
    
    # Model rows
    for mc in model_cols:
        row_vals = [get_label(mc)]
        for _, row in df_var.iterrows():
            stock = row['Stock']
            val = row[mc]
            s = fmt(val)
            # Bold if closest to empirical
            if closest[stock] == mc and s != r'--':
                s = r'\textbf{' + s + '}'
            row_vals.append(s)
        lines.append(' & '.join(row_vals) + r' \\')
    
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\caption{1\% VaR (returns) for each stock and model. '
                 r'Empirical VaR in first row; bold values indicate the model '
                 r'closest to the empirical VaR for each stock.}')
    lines.append(r'\label{tab:var_1pct}')
    lines.append(r'\end{table}')
    
    return '\n'.join(lines)

# =============================================================================
# LaTeX Model Comparison Table
# =============================================================================

def generate_comparison_latex(all_results, returns):
    """
    Generate LaTeX table showing best model by LogLik, AIC, BIC per stock.
    """
    rows = []
    for stock in returns.columns:
        stock_res = all_results[stock]
        models = list(stock_res.keys())
        
        logliks = {m: stock_res[m]['loglik'] for m in models}
        aics = {m: stock_res[m]['AIC'] for m in models}
        bics = {m: stock_res[m]['BIC'] for m in models}
        
        valid_ll = {m: v for m, v in logliks.items() if np.isfinite(v)}
        valid_aic = {m: v for m, v in aics.items() if np.isfinite(v)}
        valid_bic = {m: v for m, v in bics.items() if np.isfinite(v)}
        
        best_ll = max(valid_ll, key=valid_ll.get) if valid_ll else 'N/A'
        best_aic = min(valid_aic, key=valid_aic.get) if valid_aic else 'N/A'
        best_bic = min(valid_bic, key=valid_bic.get) if valid_bic else 'N/A'
        
        rows.append((stock, best_ll, best_aic, best_bic))
    
    lines = []
    lines.append(r'\begin{table}[ht]')
    lines.append(r'\centering')
    lines.append(r'\begin{tabular}{lrrr}')
    lines.append(r'\toprule')
    lines.append(r'Stock & Best (Log-Lik) & Best (AIC) & Best (BIC) \\')
    lines.append(r'\midrule')
    
    for stock, ll, aic, bic in rows:
        lines.append(f'{stock} & {ll} & {aic} & {bic} ' + r'\\')
    
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\caption{Best model for each stock according to in-sample log-likelihood, AIC, and BIC.}')
    lines.append(r'\label{tab:best_models}')
    lines.append(r'\end{table}')
    
    return '\n'.join(lines)

# =============================================================================
# Main
# =============================================================================

def main():
    print("Loading results.pkl...")
    try:
        results = load_results("results.pkl")
    except FileNotFoundError:
        print("Error: results.pkl not found. Run compute_models.py first.")
        return
    
    df_var = results['var_table']
    all_results = results['all_results']
    returns = results['returns']
    
    # Generate VaR LaTeX table
    print("\n" + "=" * 60)
    print("VaR Table (LaTeX)")
    print("=" * 60)
    var_latex = generate_var_latex(df_var)
    print(var_latex)
    
    # Save to file
    with open("var_table.tex", "w") as f:
        f.write(var_latex)
    print("\nSaved to: var_table.tex")
    
    # Generate model comparison LaTeX table
    print("\n" + "=" * 60)
    print("Model Comparison Table (LaTeX)")
    print("=" * 60)
    comp_latex = generate_comparison_latex(all_results, returns)
    print(comp_latex)
    
    with open("model_comparison.tex", "w") as f:
        f.write(comp_latex)
    print("\nSaved to: model_comparison.tex")

if __name__ == "__main__":
    main()
