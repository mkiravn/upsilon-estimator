# Upsilon K Estimator

Predict **υ_k**, the relationship-class-dependent attenuation of marker-tagged
heritability (`h²_{v,k} = υ_k · h²`), from a phased founder panel in zarr or npz
format. For a marker set 𝓜 and a causal set 𝓒,

```
υ_k = Cov_k(G_M, G_C) / Var_k(G_M)
```

is estimated from the two-locus decomposition `Cov_k(Z_a, Z_b) = A_k r² + B_k r λ_a λ_b + Φ_k`
averaged over locus pairs. The primary estimator uses the robust model
**(r² + Φₖ) / (r² + Φₖ)** — the cross term `B_k r λ_a λ_b` is dropped because it
adds negligible signal but has kurtosis in the thousands and destabilises the ratio.

For the derivation see `../supplement_imperfect_LD.tex`.

## Installation

```bash
pip install -e .            # core (numpy, pandas)
pip install -e ".[zarr]"    # + zarr, for large / genome-scale panels
```

`ld_sim` is only needed to read xftsim-format zarr panels; standard zarr and npz
panels work without it.

## Recommended estimator (genome-scale)

`estimate_upsilon_k_scale_aware` is the one to use for genome-scale simulations.
It samples locus pairs **stratified by genetic distance** (so every scale is
covered) with **density-weighted importance weights** (so the estimate is
invariant to SNP density, and a single chromosome reproduces the genome-wide
value), and returns a **block-jackknife standard error** per class.

### Command line

```bash
# scale-aware estimate with jackknife SEs
upsilon panel.zarr -o upsilon.csv -k 2 3 4 5 6 7 8 9 \
  --marker-threshold 0.01 --causal-threshold 0.05 -v

# with explicit marker/causal index files
upsilon panel.zarr -o upsilon.csv \
  --marker-indices markers.npy --causal-indices causal.npy

# Speed alpha-model weighting (adds weighted columns; alpha=-1 is GCTA)
upsilon panel.zarr -o upsilon.csv --alpha -0.25

# quick point estimate, no jackknife
upsilon panel.zarr -o upsilon.csv --fast
```

Key options: `--n-draws` (pairs, default 200k), `--n-strata` (distance strata),
`--n-blocks` (jackknife blocks, default 200), `--d-max` (max pair distance in cM;
default is the panel's full map span — set this to the chromosome length for a
single-chromosome panel), `--alpha`, `--include-a-k`, `--chromosome`, `--seed`.

### Python

```python
from upsilon_estimator import estimate_upsilon_k_scale_aware

df = estimate_upsilon_k_scale_aware(
    "panel.zarr",
    marker_threshold=0.01, causal_threshold=0.05,
    k_values=range(2, 10), n_draws=200_000, n_blocks=200,
    alpha=None,          # or e.g. -0.25 for the Speed model
)
# columns: k, upsilon_k, upsilon_k_se, n_marker, n_causal
```

Provide `marker_indices=` / `causal_indices=` (arrays) to define the sets
explicitly instead of by MAF threshold.

## Other functions

- `estimate_upsilon_k` — fast robust point estimate (no jackknife); used by `--fast`.
- `estimate_upsilon_k_weighted` — Speed alpha-model on the legacy uniform sampler.
- `sampling.stratified_draws`, `sampling.block_jackknife_se`, `sampling.effect_size_weights` — building blocks.

## Output

CSV with `k`, `upsilon_k`, `upsilon_k_se`, `n_marker`, `n_causal`. With `--alpha`,
adds `upsilon_k_weighted`, `upsilon_k_weighted_se`, `alpha`. `--fast` instead
returns `k, upsilon_k, r2_MM, r2_MQ, shared_ibd_part`.

## Notes for genome-scale runs

- **Scale.** υ_k depends on the maximum pair genetic distance. Leave `--d-max`
  unset to use the panel's full map span (the right choice for a single
  chromosome); for a multi-chromosome zarr, load one chromosome at a time with
  `--chromosome` so pairs stay within-chromosome.
- **Density invariance.** Thinning SNPs leaves υ_k unchanged within the SE, so
  runtime can be tuned via `--n-draws` without biasing the estimate.
- **Uncertainty.** The block jackknife recomputes the whole ratio per deleted
  block, so the numerator/denominator correlation and LD between pairs are handled;
  run a couple of seeds to confirm Monte-Carlo error is below the jackknife SE.

## License

MIT
