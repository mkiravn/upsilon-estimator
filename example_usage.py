#!/usr/bin/env python3
"""Example: estimating upsilon_k as a Python module."""

from upsilon_estimator import estimate_upsilon_k

# Basic usage: estimate upsilon_k for default k values (2-10)
results = estimate_upsilon_k(
    zarr_path="data/processed/founders_panel.npz",
    marker_threshold=0.01,       # allele freq > 1% = marker
    causal_threshold=0.05,       # allele freq < 5% = causal
    n_draws=150_000,              # sample size for locus pairs
    seed=1,                        # reproducibility
)

print("Basic estimation (k=2-10):")
print(results)
print()

# Subset to specific k values
results_subset = estimate_upsilon_k(
    zarr_path="data/processed/founders_panel.npz",
    n_draws=50_000,
    k_values=[2, 4, 6, 8, 10],
)

print("Subset k values:")
print(results_subset)
print()

# With distance binning
results_binned = estimate_upsilon_k(
    zarr_path="data/processed/founders_panel.npz",
    n_draws=50_000,
    k_values=[4, 8],
    bin_by_distance=True,
    n_distance_bins=5,
)

print("With distance binning (5 bins):")
print(results_binned[["k", "upsilon_k_crude", "upsilon_k_full"]])
print("Binned columns:", [c for c in results_binned.columns if "bin_" in c])

# Save to CSV
results.to_csv("upsilon_k_estimates.csv", index=False)
print("\nSaved to upsilon_k_estimates.csv")
