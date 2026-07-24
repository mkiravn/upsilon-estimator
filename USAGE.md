# Upsilon K Estimator

Estimate upsilon_k (marker-causal relatedness ratio) from founder panels in zarr or npz format.

Outputs both crude (debiased r² + Φₖ) and full (exact hₖ components) estimates, with optional distance binning.

## Quick Start

### CLI

```bash
python3 run_upsilon.py <panel_path> --output results.csv [options]
```

**Required arguments:**
- `panel_path`: path to zarr directory or .npz file (auto-detects xftsim format)
- `--output, -o`: output CSV file path

**Specifying markers vs causal sites (choose one approach):**
- `--marker-indices`: path to file with marker site indices (.npy, .npz, or single-column CSV)
- `--causal-indices`: path to file with causal site indices (.npy, .npz, or single-column CSV)
- **OR** use allele frequency thresholds (if indices not provided):
  - `--marker-threshold`: AF threshold for markers (default 0.01)
  - `--causal-threshold`: AF threshold for causal (default 0.05)

**Panel options:**
- `--chromosome`: load specific chromosome (xftsim zarr only; default loads first chromosome)

**Sampling options:**
- `--n-draws, -n`: number of locus pairs to sample (default 150,000)
- `--seed, -s`: random seed (default 1)
- `--k-values, -k`: kinship classes to compute (default 2 3 4 5 6 7 8 9 10)

**Output options:**
- `--bin-by-distance`: bin results by genetic distance
- `--n-distance-bins`: number of distance bins (default 10)
- `--verbose, -v`: print progress information

**Examples:**

```bash
# Basic run with default AF thresholds
python3 run_upsilon.py founders.npz -o results.csv

# With explicit marker and causal site indices
python3 run_upsilon.py founders.npz -o results.csv \
  --marker-indices markers.npy \
  --causal-indices causal.npy

# Load single chromosome from xftsim zarr
python3 run_upsilon.py /path/to/xftsim.zarr -o chr1_results.csv \
  --chromosome 1 \
  --marker-indices markers.npy \
  --causal-indices causal.npy

# Custom thresholds and smaller sample
python3 run_upsilon.py founders.npz -o results.csv \
  --marker-threshold 0.005 \
  --causal-threshold 0.02 \
  --n-draws 50000

# Distance-binned results by genetic distance
python3 run_upsilon.py founders.npz -o results.csv \
  --bin-by-distance \
  --n-distance-bins 5 \
  -k 4 6 8 10 \
  --marker-indices markers.npy \
  --causal-indices causal.npy

# Verbose output with chromosome selection
python3 run_upsilon.py founders.zarr -o results.csv \
  --chromosome 1 \
  --n-draws 100000 \
  -v
```

### Python Module

```python
import numpy as np
from upsilon_estimator import estimate_upsilon_k

# With explicit site indices
marker_idx = np.load("markers.npy")
causal_idx = np.load("causal.npy")

results = estimate_upsilon_k(
    zarr_path="founders.npz",
    marker_indices=marker_idx,
    causal_indices=causal_idx,
    n_draws=150_000,
    seed=1,
    k_values=[2, 4, 6, 8, 10],
)

# Save to CSV
results.to_csv("results.csv", index=False)
print(results[["k", "upsilon_k_crude", "upsilon_k_full"]])

# Or with AF thresholds (if indices not available)
results_thresh = estimate_upsilon_k(
    zarr_path="founders.npz",
    marker_threshold=0.01,
    causal_threshold=0.05,
    n_draws=150_000,
)

# For xftsim zarr with chromosome selection
results_chr1 = estimate_upsilon_k(
    zarr_path="/path/to/xftsim.zarr",
    chromosome=1,
    marker_indices=marker_idx,
    causal_indices=causal_idx,
    bin_by_distance=True,
    n_distance_bins=5,
)
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

Supports **xftsim zarr** (lazy-loaded, preferred for large panels), **standard zarr**, and **npz** formats. Auto-detects format.

**xftsim zarr directory** (auto-detected) contains:
- `HaplotypeArray`: (n_individuals, 2×n_true_sites) blosc-compressed genotypes
- `af`, `chrom`, `pos_bp`, `pos_cM`: metadata arrays (de-duplicated per true site)
- Supports `--chromosome` selection for memory efficiency
- Requires `blosc` package: `pip install blosc`

**Standard zarr directory** contains:
- `genotypes` (or `haplotypes`): array of shape (n_sites, n_haplotypes)
- `positions`: physical positions
- `allele_freq`: allele frequencies
- `genetic_position`: genetic positions in cM (optional)
- Optional: `chrom` array for chromosome filtering

**NPZ file** contains:
- `haplotypes` (or `genotypes`): array of shape (n_sites, n_haplotypes)
- `positions`: physical positions
- `allele_freq`: allele frequencies
- `recombination_rate` (optional): used to compute genetic position if `genetic_position` missing
- Optional: `chrom` array for chromosome filtering

**Index files** (for `--marker-indices` / `--causal-indices`):
- `.npy` format: single numpy integer array
- `.npz` format: single array inside
- `.csv` format: single column of integer indices

## Parameters Explained

**Site selection** (choose one):
- **marker_indices / causal_indices**: explicit arrays of site indices (preferred, avoids AF threshold ambiguity)
- **marker_threshold / causal_threshold**: allele frequency thresholds (simpler but less control over density/spacing)

**Panel**:
- **chromosome**: load single chromosome from xftsim zarr for faster runs on large panels

**Sampling**:
- **n_draws**: sample size for locus pairs; larger = more stable estimates (150k ≈ 30s on single chr)
- **seed**: for reproducibility of random draws
- **k_values**: which kinship classes to compute (skipping unused k values speeds things up)

**Output**:
- **bin_by_distance**: split estimates by genetic distance bins to see LD decay and spacing effects

## Module API

### `estimate_upsilon_k(...)`

Main estimation function. Returns pandas DataFrame.

**Arguments:**
- `zarr_path`: str, path to panel (zarr or npz)
- `marker_indices`: np.ndarray, explicit marker site indices (preferred)
- `causal_indices`: np.ndarray, explicit causal site indices (preferred)
- `marker_threshold`: float (default 0.01, used only if marker_indices=None)
- `causal_threshold`: float (default 0.05, used only if causal_indices=None)
- `chromosome`: str, chromosome to load (xftsim zarr only)
- `n_draws`: int (default 150,000)
- `seed`: int (default 1)
- `k_values`: list of int or None for default [2..10]
- `bin_by_distance`: bool (default False)
- `n_distance_bins`: int (default 10)

### `load_panel(path, chromosome=None)`

Load and validate a panel. Auto-detects xftsim/standard zarr or npz.

**Returns dict with keys:**
- `panel`: LazyPanel object (lazy-loaded genotypes)
- `pos`, `af`, `gpos`: coordinate arrays
- `chrom`: chromosome array (if available)
- `n_hap`, `n_sites`: dimensions

### `paired_draws(marker_idx, causal_idx, n, rng, panel, gpos)`

Draw n paired loci with common random numbers. Returns tuple (num, den) dicts.

### `estimate_components(k, pair_data, n_hap)`

Compute hₖ decomposition for a pair dataset. Returns dict with components.

## Performance & Practical Notes

**Recommended workflow for large genome-wide panels:**

1. **Load one chromosome at a time**: 
   ```bash
   upsilon run_upsilon.py founders.zarr --chromosome 1 --output chr1.csv \
     --marker-indices markers.npy --causal-indices causal.npy
   ```
   This keeps memory usage low and can parallelize across chromosomes.

2. **Use explicit site indices** (preferred over AF thresholds):
   - Gives direct control over marker vs causal density/spacing
   - Faster than scanning AF each time
   - More interpretable (know exactly which SNPs are in each pool)

3. **Memory and speed**:
   - xftsim zarr: lazy-loads only needed sites (efficient for large panels)
   - Standard zarr: lazy-loads genotypes on-demand
   - NPZ: loads full genotype matrix into RAM (OK for single chromosomes)
   - `n_hap` auto-detected from genotype array
   - Runtime ~30s per chromosome with 150k draws on modern hardware
   - Reduce `--n-draws` if needed for quicker exploratory runs

4. **Reproducibility**:
   - Use `--seed` for reproducible random draws
   - Debiased r² removes finite-sample 1/(n-2) bias
   - Distance bins skip if < 10 pairs (logged if verbose)

**Internals:**
   - ld_sim functions are vendored (no external dependencies needed)
   - Supports blosc for xftsim zarr decompression: `pip install blosc`

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
