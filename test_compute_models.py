
import numpy as np
import pandas as pd
import compute_models
import time

def test_fitting():
    print("Testing Weighted Chi-Squared (with Negative Weights) Fitting...")
    
    # Generate dummy returns data (Gaussian-ish but with tails)
    np.random.seed(42)
    data = np.random.standard_t(df=5, size=1000) * 0.01
    
    print(f"Data shape: {data.shape}")
    
    # Test CF function directly
    print("Testing CF function...")
    weights = np.array([0.5, -0.5])
    location = 0.0
    phi, t_arr = compute_models.weighted_chisq_cf_grid(weights, location)
    print(f"CF computed. Shape: {phi.shape}")
    
    # Test Fitting
    print("Testing fit_weighted_chisq (K=2)...")
    start = time.time()
    params, loglik = compute_models.fit_weighted_chisq(data, K=2)
    end = time.time()
    
    print(f"Fitting complete in {end-start:.4f}s")
    print(f"Log-likelihood: {loglik}")
    print(f"Params: {params}")
    
    # Check if weights contain negative values (they might or might not, but should be allowed)
    K = 2
    weights = params[:K]
    print(f"Weights: {weights}")
    
    # Test VaR
    print("Testing VaR computation...")
    var = compute_models.var_weighted_chisq(params, K=2, alpha=0.01)
    print(f"VaR (1%): {var}")
    
    # Run full pipeline on dummy data
    print("\nRunning full pipeline on dummy data...")
    results = compute_models.fit_all_models(data, "DummyStock")
    print("Results keys:", results.keys())
    
    if 'WeightedChiSq_K2' in results:
        print("WeightedChiSq_K2 found in results.")
    else:
        print("WeightedChiSq_K2 NOT found in results.")

if __name__ == "__main__":
    test_fitting()
