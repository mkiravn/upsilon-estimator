# Upsilon K Estimator

Estimate upsilon_k (marker-causal relatedness ratio) from founder panels in zarr or npz format.

Outputs both crude (debiased r² + Φₖ) and full (exact hₖ components) estimates, with optional distance binning.

## Quick Start

### CLI

```bash
python3 run_upsilon.py <panel_path> --output results.csv [options]
```

**Required arguments:**
- `panel_path`: path to zarr directory or .npz file
- `--output, -o`: output CSV file path

**Options:**
- `--marker-threshold`: allele frequency threshold for marker sites (default 0.01)
- `--causal-threshold`: allele frequency threshold for causal sites (default 0.05)
- `--n-draws, -n`: number of locus pairs to sample (default 150,000)
- `--seed, -s`: random seed (default 1)
- `--k-values, -k`: kinship classes to compute (default 2 3 4 5 6 7 8 9 10)
- `--bin-by-distance`: bin results by genetic distance
- `--n-distance-bins`: number of distance bins (default 10)
- `--verbose, -v`: print progress information

**Examples:**

```bash
# Basic run with default parameters
python3 run_upsilon.py founders.npz -o results.csv

# Custom thresholds and smaller sample
python3 run_upsilon.py founders.npz -o results.csv \
  --marker-threshold 0.005 \
  --causal-threshold 0.02 \
  --n-draws 50000

# Distance-binned results
python3 run_upsilon.py founders.npz -o results.csv \
  --bin-by-distance \
  --n-distance-bins 5 \
  -k 4 6 8 10

# Verbose output
python3 run_upsilon.py founders.npz -o results.csv -v
```

### Python Module

```python
from upsilon_estimator import estimate_upsilon_k

results = estimate_upsilon_k(
    zarr_path="founders.npz",
    marker_threshold=0.01,
    causal_threshold=0.05,
    n_draws=150_000,
    seed=1,
    k_values=[2, 4, 6, 8, 10],
    bin_by_distance=False,
)

# Save to CSV
results.to_csv("results.csv", index=False)

# Access individual estimates
print(results[["k", "upsilon_k_crude", "upsilon_k_full"]])
```

See `example_usage.py` for more examples.

## Output Format

The CSV contains one row per kinship class with columns:

- `k`: kinship class (2=full sibs, 3=half-sibs, ..., 10=fourth cousins)
- `upsilon_k_crude`: crude estimate (mean r² + Φₖ ratio)
- `upsilon_k_full`: full estimate (exact hₖ decomposition)
- `r2_MM`: mean debiased r² (marker-marker pairs)
- `r2_MQ`: mean debiased r² (marker-causal pairs)
- `shared_ibd_part`: mean shared IBD component (Φₖ)

When `--bin-by-distance` is used, additional columns appear for each bin:
- `bin_{i}_upsilon_crude`, `bin_{i}_upsilon_full`: estimates for bin i
- `bin_{i}_r2_MM`, `bin_{i}_r2_MQ`: r² values for bin i
- `bin_{i}_distance`: genetic distance range (cM) for bin i

## Input Format

Supports both **zarr** and **npz** formats.

**Zarr directory** must contain:
- `genotypes` (or `haplotypes`): array of shape (n_sites, n_haplotypes)
- `positions`: physical positions
- `allele_freq`: allele frequencies
- `genetic_position`: genetic positions in cM (optional; computed from `recombination_rate` if missing)

**NPZ file** must contain:
- `haplotypes` (or `genotypes`): array of shape (n_sites, n_haplotypes)
- `positions`: physical positions
- `allele_freq`: allele frequencies
- `recombination_rate` (optional): used to compute genetic position if `genetic_position` missing

## Parameters Explained

- **marker_threshold**: sites with allele frequency > this are "markers" (typically common variants)
- **causal_threshold**: sites with allele frequency < this are "causal-like" (rare variants)
- **n_draws**: sample size for locus pairs; larger = more stable estimates (default 150k takes ~30s)
- **seed**: for reproducibility of random draws
- **k_values**: which kinship classes to compute (skipping speeds things up)
- **bin_by_distance**: split estimates by genetic distance to see LD decay

## Module API

### `estimate_upsilon_k(...)`

Main estimation function. Returns pandas DataFrame.

**Arguments:**
- `zarr_path`: str, path to panel
- `marker_threshold`: float (default 0.01)
- `causal_threshold`: float (default 0.05)
- `n_draws`: int (default 150,000)
- `seed`: int (default 1)
- `k_values`: list of int or None for default [2..10]
- `bin_by_distance`: bool (default False)
- `n_distance_bins`: int (default 10)

### `load_panel(path)`

Load and validate a panel. Returns dict with keys:
- `panel`: LazyPanel object
- `hap`, `pos`, `af`, `gpos`: raw arrays
- `n_hap`, `n_sites`: dimensions

### `paired_draws(marker_idx, causal_idx, n, rng, panel, gpos)`

Draw n paired loci with common random numbers. Returns tuple (num, den) dicts.

### `estimate_components(k, pair_data, n_hap)`

Compute hₖ decomposition for a pair dataset. Returns dict with components.

## Computational Notes

- Zarr format is preferred for large panels (lazy row loading)
- NPZ loaded fully into memory
- n_hap is auto-detected from the panel
- Debiased r² removes the 1/(n-2) finite-sample bias
- Distance bins skip if < 10 pairs
- Requires ld_sim package in `../src` (or add src to PYTHONPATH)

## Example: Full Pipeline

```python
# Script to estimate upsilon_k for multiple panels
from pathlib import Path
from upsilon_estimator import estimate_upsilon_k

panels = ["panel_chr1.npz", "panel_chr2.npz"]

for panel_path in panels:
    results = estimate_upsilon_k(
        panel_path,
        n_draws=100_000,
        k_values=[2, 4, 6, 8, 10],
    )
    output = f"results_{Path(panel_path).stem}.csv"
    results.to_csv(output, index=False)
    print(f"Saved {output}")
```
