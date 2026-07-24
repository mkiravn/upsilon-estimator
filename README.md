# Upsilon K Estimator

Estimate upsilon_k (marker-causal relatedness ratio) from founder panels in zarr or npz format.

Outputs both crude (debiased r² + Φₖ) and full (exact hₖ components) estimates, with optional distance binning.

## Installation

```bash
# Install in editable mode
pip install -e .

# With zarr support (for large panels)
pip install -e ".[zarr]"
```

## Quick Start

### Command-line

```bash
upsilon founders.npz --output results.csv
```

Or run directly:

```bash
python3 run_upsilon.py founders.npz --output results.csv
```

### Python module

```python
from upsilon_estimator import estimate_upsilon_k

results = estimate_upsilon_k("founders.npz")
results.to_csv("results.csv", index=False)
```

## Documentation

See [USAGE.md](USAGE.md) for full documentation, including:
- CLI arguments and examples
- Python module API
- Input format specifications
- Parameter explanations

## Requirements

- Python ≥3.9
- numpy, pandas
- zarr (optional, for zarr-backed panels)
- ld_sim package (must be available in Python path)

## Example

```bash
python3 run_upsilon.py panel.npz \
  --output results.csv \
  --n-draws 100000 \
  --k-values 2 4 6 8 10 \
  --bin-by-distance \
  --n-distance-bins 5 \
  --verbose
```

This estimates upsilon_k for k={2,4,6,8,10} using 100k locus pair samples, binned by genetic distance into 5 bins.

## Output

CSV with columns:
- `k`: kinship class
- `upsilon_k_crude`: crude estimate
- `upsilon_k_full`: full estimate
- `r2_MM`, `r2_MQ`, `shared_ibd_part`: components

Distance bins (if `--bin-by-distance`) add columns like `bin_0_upsilon_crude`, `bin_0_r2_MM`, etc.

## License

MIT
