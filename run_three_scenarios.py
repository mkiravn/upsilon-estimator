"""Estimate upsilon_k for all imperfect-LD scenarios in config/config.yaml,
using the converted founders panel and each scenario's alpha (Speed et al.
causal-effect weighting) via estimate_upsilon_k(..., alpha=...).

Threshold mapping mirrors config/config.yaml's imperfect_ld_experiments:
  marker_min_maf / causal_max_maf select marker/causal pools by MAF;
  disjoint_markers excludes causal sites from the marker pool (sc4-style).
n_causal/n_markers subsampling and marker_mode="sample_noncausal" vs
"sample_any" beyond the disjoint/overlap distinction are not replicated here
(this script full-thresholds rather than subsamples, as before); it is a
diagnostic sanity check, not a stand-in for the Snakemake pipeline.
"""
from pathlib import Path
import numpy as np
import yaml

from upsilon_estimator import estimate_upsilon_k

PANEL = "/home/mianniravn/claude-822397757/-project2-jnovembre-mianniravn-glitch-0526/a892de0d-7b26-411d-b8bb-9a87234dd901/scratchpad/founders_n5000_gbr_converted.npz"
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
OUTDIR = Path("results")
OUTDIR.mkdir(exist_ok=True)

# Load allele frequencies to compute explicit indices
d = np.load(PANEL)
af = d['allele_freq']
maf = np.minimum(af, 1.0 - af)  # Minor allele frequency (always <= 0.5)
all_idx = np.arange(len(maf))

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

SCENARIOS = {}
for exp in cfg["imperfect_ld_experiments"]:
    name = exp["name"]
    causal_max_maf = exp.get("causal_max_maf")
    marker_min_maf = exp["marker_min_maf"]
    disjoint = exp.get("disjoint_markers", False)
    alpha = exp["alpha"]

    if causal_max_maf is None:
        causal_idx = np.nonzero(maf > 0)[0]
    else:
        causal_idx = np.nonzero(maf < causal_max_maf)[0]

    marker_mask = maf > marker_min_maf
    if disjoint:
        marker_mask &= ~np.isin(all_idx, causal_idx)
    marker_idx = np.nonzero(marker_mask)[0]

    SCENARIOS[name] = dict(
        marker_indices=marker_idx,
        causal_indices=causal_idx,
        alpha=alpha,
        name=f"{name} (alpha={alpha}, marker_min_maf={marker_min_maf}, "
             f"causal_max_maf={causal_max_maf}, disjoint={disjoint})",
    )

K_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10]
N_DRAWS = 150_000

for name, params in SCENARIOS.items():
    print(f"=== {name}: {params.get('name', name)} ===")

    results = estimate_upsilon_k(
        PANEL,
        marker_indices=params["marker_indices"],
        causal_indices=params["causal_indices"],
        alpha=params["alpha"],
        n_draws=N_DRAWS,
        seed=1,
        k_values=K_VALUES,
        bin_by_distance=True,
        n_distance_bins=5,
    )
    results.insert(0, "scenario", name)
    out_path = OUTDIR / f"upsilon_{name}.csv"
    results.to_csv(out_path, index=False)
    print(results[["scenario", "k", "upsilon_k_crude", "upsilon_k_full", "r2_MM", "r2_MQ", "shared_ibd_part"]].to_string(index=False))
    print(f"saved {out_path}\n")
