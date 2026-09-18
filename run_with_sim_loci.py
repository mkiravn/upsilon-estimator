"""Estimate upsilon_k using the exact SNP indices from the imperfect_ld simulations."""
from pathlib import Path
import numpy as np
import pandas as pd

from upsilon_estimator import estimate_upsilon_k

PANEL = "/home/mianniravn/claude-822397757/-project2-jnovembre-mianniravn-glitch-0526/a892de0d-7b26-411d-b8bb-9a87234dd901/scratchpad/founders_n5000_gbr_converted.npz"
OUTDIR = Path("results")
OUTDIR.mkdir(exist_ok=True)

K_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10]
N_DRAWS = 150_000

# Read SNP indices from simulation effects files
scenarios_sim = {}

for sc in ['sc0_gcta', 'sc3_gcta', 'sc4_gcta']:
    effects = pd.read_csv(f'../results/imperfect_ld/{sc}/effects/rep0.effects.tsv', sep='\t')

    causal_idx = effects[effects['is_causal'] == True].index.values

    # For sc3/sc4, filter markers by MAF>0 and MAF>=0.01 (not all non-causal SNPs or fixed alleles)
    if sc in ['sc3_gcta', 'sc4_gcta']:
        marker_idx = effects[(effects['is_causal'] == False) & (effects['maf'] > 0) & (effects['maf'] >= 0.01)].index.values
    else:
        # For sc0, also exclude fixed alleles
        marker_idx = effects[(effects['is_causal'] == False) & (effects['maf'] > 0)].index.values

    scenarios_sim[sc] = {
        'marker_indices': marker_idx.astype(int),
        'causal_indices': causal_idx.astype(int),
        'name': f"{sc} (sim loci, MAF-filtered)"
    }
    print(f"{sc}: {len(marker_idx)} markers (MAF>=0.01), {len(causal_idx)} causals")

# Run estimation with simulation loci
for name, params in scenarios_sim.items():
    print(f"\n=== {name}: {params.get('name', name)} ===")

    results = estimate_upsilon_k(
        PANEL,
        marker_indices=params["marker_indices"],
        causal_indices=params["causal_indices"],
        n_draws=N_DRAWS,
        seed=1,
        k_values=K_VALUES,
        bin_by_distance=False,
    )
    results.insert(0, "scenario", name)
    out_path = OUTDIR / f"upsilon_{name}_sim_loci.csv"
    results.to_csv(out_path, index=False)
    print(results[["scenario", "k", "upsilon_k_crude", "upsilon_k_full", "r2_MM", "r2_MQ", "shared_ibd_part"]].to_string(index=False))
    print(f"saved {out_path}\n")
