"""Quick check: does the scale-aware estimator give consistent upsilon_k
across different chromosomes for one scenario (sc4_gcta)? The README claims
density/scale invariance means a single chromosome reproduces the genome-wide
value -- this is a sanity check of that claim, not a full genome-wide run.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from upsilon_estimator import estimate_upsilon_k_scale_aware
from upsilon_estimator.estimators import load_panel

REPO_ROOT = Path(__file__).resolve().parent.parent
PANEL = REPO_ROOT / "results" / "founders" / "founders_n5000_gbr.zarr"
OUTDIR = Path(__file__).resolve().parent / "results"

CHROMOSOMES = [1, 6, 12, 18, 22]  # spread across the genome (varying size / gene density)
K_VALUES = list(range(2, 10))
N_DRAWS = 30_000
N_BLOCKS = 30
SEED = 1

# sc4_gcta parameters
MARKER_MIN_MAF = 0.01
CAUSAL_MAX_MAF = 0.05
DISJOINT = True
ALPHA = -1.0


def build_indices(af, marker_min_maf, causal_max_maf, disjoint):
    maf = np.minimum(af, 1.0 - af)
    all_idx = np.arange(len(maf))
    if causal_max_maf is None:
        causal_idx = np.nonzero(maf > 0)[0]
    else:
        causal_idx = np.nonzero((maf > 0) & (maf < causal_max_maf))[0]
    marker_mask = maf > marker_min_maf
    if disjoint:
        marker_mask &= ~np.isin(all_idx, causal_idx)
    marker_idx = np.nonzero(marker_mask)[0]
    return marker_idx, causal_idx


def main():
    rows = []
    for chrom in CHROMOSOMES:
        panel_data = load_panel(str(PANEL), chromosome=chrom)
        af = panel_data["af"]
        gpos = panel_data["gpos"]
        marker_idx, causal_idx = build_indices(af, MARKER_MIN_MAF, CAUSAL_MAX_MAF, DISJOINT)
        print(f"=== chr{chrom}: {panel_data['n_sites']} sites, map span {gpos.max()-gpos.min():.1f} cM, "
              f"{len(marker_idx)} markers, {len(causal_idx)} causal ===")

        est = estimate_upsilon_k_scale_aware(
            str(PANEL),
            marker_indices=marker_idx,
            causal_indices=causal_idx,
            n_draws=N_DRAWS,
            seed=SEED,
            k_values=K_VALUES,
            chromosome=chrom,
            n_blocks=N_BLOCKS,
            alpha=ALPHA,
        )
        est.insert(0, "chromosome", chrom)
        rows.append(est)
        print(est[["k", "upsilon_k", "upsilon_k_se"]].to_string(index=False))
        print()

    combined = pd.concat(rows, ignore_index=True)
    combined.to_csv(OUTDIR / "sc4_gcta_chromosome_check.csv", index=False)
    print(f"Saved {OUTDIR / 'sc4_gcta_chromosome_check.csv'}")


if __name__ == "__main__":
    main()
