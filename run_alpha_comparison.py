"""Compare the estimator's alpha-weighted upsilon_k against simulation for the
three alpha variants that share sc1 topology (marker_min_maf=0, causal_max_maf=
None, disjoint=False): sc1_gcta (alpha=-1, GCTA/uniform), sc1_alpha_m05
(alpha=-0.5), sc1_alpha_m15 (alpha=-1.5).

predictor=GRM_marker vs response=GRM_causal is alpha-invariant (both GRMs are
plain genotype GRMs, independent of effect sizes) -- confirmed identical
across these three scenarios in every prior run. The alpha-sensitive
comparison instead uses:
  - estimator: upsilon_k_weighted (Speed alpha-model weighted, from
    estimate_upsilon_k_scale_aware(..., alpha=...))
  - simulation: predictor=GRM_marker, response=pheno_crossprod slope, divided
    by h2=0.6 (the source_experiment's heritability). Within one pedigree
    class k, MGSE (b2=0.2 here) shifts only the intercept, not the slope (see
    refs/glitch-paper/paper/main.tex), and mu=0 so there's no AM confound, so
    slope_mean(k) / h2 is a valid empirical estimate of upsilon_k_weighted(k)
    for this alpha.

Uses density-matched marker/causal sets (see run_density_matched_comparison.py).
"""
from pathlib import Path

import numpy as np
import pandas as pd

import run_density_matched_comparison as dm
import run_imperfect_ld_comparison as base
from upsilon_estimator import estimate_upsilon_k_scale_aware

H2 = 0.6
ALPHA_SCENARIOS = ["sc1_gcta", "sc1_alpha_m05", "sc1_alpha_m15"]
N_REPS = 3


def load_sim_pheno_slope(scenario):
    path = base.SIM_ROOT / scenario / "summaries" / "slope_summary.csv"
    df = pd.read_csv(path)
    df = df[(df["predictor"] == "GRM_marker") & (df["response"] == "pheno_crossprod")]
    df = df[np.isfinite(pd.to_numeric(df["k"], errors="coerce"))].copy()
    df["k"] = df["k"].astype(int)
    df["sim_upsilon_k_weighted"] = df["slope_mean"] / H2
    df["sim_upsilon_k_weighted_se"] = df["slope_se"] / H2
    return df.set_index("k")[["sim_upsilon_k_weighted", "sim_upsilon_k_weighted_se"]]


def run_scenario_weighted(name, params):
    reps = base.available_reps(name, N_REPS)
    print(f"=== {name}: alpha={params['alpha']}, density-matched, reps={reps} ===")

    per_rep = []
    for rep in reps:
        counts = dm.SCENARIO_COUNTS[name]
        marker_idx, causal_idx = dm.matched_indices_for_rep(
            name, rep, params["marker_min_maf"], counts["marker_mode"],
            counts["n_markers"], seed=dm.MARKER_SEED_BASE + rep,
        )
        est = estimate_upsilon_k_scale_aware(
            str(base.PANEL),
            marker_indices=marker_idx,
            causal_indices=causal_idx,
            n_draws=base.N_DRAWS,
            seed=base.SEED,
            k_values=base.K_VALUES,
            chromosome=base.CHROMOSOME,
            n_blocks=base.N_BLOCKS,
            alpha=params["alpha"],
        )
        est["rep"] = rep
        per_rep.append(est)

    all_reps = pd.concat(per_rep, ignore_index=True)
    agg = all_reps.groupby("k").agg(
        upsilon_k_weighted=("upsilon_k_weighted", "mean"),
        upsilon_k_weighted_se=("upsilon_k_weighted", "sem"),
    ).reset_index()
    return agg


def main():
    outdir = base.OUTDIR
    all_rows = []
    for name in ALPHA_SCENARIOS:
        params = base.SCENARIOS[name]
        est = run_scenario_weighted(name, params)
        sim = load_sim_pheno_slope(name)
        merged = est.set_index("k").join(sim, how="left").reset_index()
        merged.insert(0, "scenario", name)
        merged.insert(1, "alpha", params["alpha"])
        merged["diff"] = merged["upsilon_k_weighted"] - merged["sim_upsilon_k_weighted"]
        all_rows.append(merged)
        cols = ["k", "upsilon_k_weighted", "upsilon_k_weighted_se",
                "sim_upsilon_k_weighted", "sim_upsilon_k_weighted_se", "diff"]
        print(merged[cols].to_string(index=False))
        print()

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(outdir / "alpha_comparison.csv", index=False)
    print(f"Saved {outdir / 'alpha_comparison.csv'}")


if __name__ == "__main__":
    main()
