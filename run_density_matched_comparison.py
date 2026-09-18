"""Density-matched vs full-MAF-pool comparison, for all 10 imperfect-LD
scenarios (chr1 only).

"Full pool" (see run_imperfect_ld_comparison.py) uses every chr1 site passing
the MAF thresholds -- much denser than what the Snakemake pipeline actually
fed into GRM_causal/GRM_marker, which subsamples down to fixed n_causal /
n_markers out of a much larger genome-wide pool.

"Density-matched" instead:
  - uses each replicate's *real* causal SNP set (effects.tsv's is_causal,
    restricted to chr1 -- see run_imperfect_ld_comparison.load_real_causal_indices)
  - downsamples the chr1 marker-eligible pool (maf > marker_min_maf, and
    non-causal if marker_mode == sample_noncausal) to the same proportional
    density the genome-wide n_markers subsample implies for chr1: frac =
    n_markers / genome-wide eligible pool size (computed per replicate from
    effects.tsv), target = round(frac * chr1 pool size).

This isolates whether matching the simulation's actual marker/causal density
(not just its MAF thresholds) changes upsilon_k, for every scenario -- not
just sc4_gcta, where the effect was large.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import run_imperfect_ld_comparison as base
from upsilon_estimator import estimate_upsilon_k_scale_aware
from upsilon_estimator.estimators import load_panel

N_REPS = 3
MARKER_SEED_BASE = 2000

# name -> (n_causal, n_markers, marker_mode)
SCENARIO_COUNTS = {
    "sc0_gcta":       dict(n_causal=800_000, n_markers=800_000, marker_mode="sample_any"),
    "sc1_gcta":       dict(n_causal=100_000, n_markers=800_000, marker_mode="sample_any"),
    "sc3_gcta":       dict(n_causal=100_000, n_markers=800_000, marker_mode="sample_any"),
    "sc4_gcta":       dict(n_causal=100_000, n_markers=800_000, marker_mode="sample_noncausal"),
    "sc3_maf05_gcta": dict(n_causal=100_000, n_markers=800_000, marker_mode="sample_any"),
    "sc4_maf05_gcta": dict(n_causal=100_000, n_markers=800_000, marker_mode="sample_noncausal"),
    "sc1_maf5":       dict(n_causal=100_000, n_markers=800_000, marker_mode="sample_any"),
    "sc4_maf5":       dict(n_causal=100_000, n_markers=800_000, marker_mode="sample_noncausal"),
    "sc1_alpha_m05":  dict(n_causal=100_000, n_markers=800_000, marker_mode="sample_any"),
    "sc1_alpha_m15":  dict(n_causal=100_000, n_markers=800_000, marker_mode="sample_any"),
}


def matched_indices_for_rep(scenario, rep, marker_min_maf, marker_mode, n_markers, seed):
    """Real causal set (chr1) + chr1 marker pool downsampled to the density
    the genome-wide n_markers subsample implies for chr1, for one replicate."""
    eff_path = base.SIM_ROOT / scenario / "effects" / f"rep{rep}.effects.tsv"
    eff = pd.read_csv(eff_path, sep="\t", usecols=["SNP", "maf", "is_causal"])
    eff["SNP"] = eff["SNP"].astype(int)

    if marker_mode == "sample_noncausal":
        genome_pool = eff[(eff["maf"] > marker_min_maf) & (~eff["is_causal"])]
    else:
        genome_pool = eff[eff["maf"] > marker_min_maf]
    frac = n_markers / len(genome_pool)

    causal_idx = base.load_real_causal_indices(scenario, rep)

    chr1_pool_mask = genome_pool["SNP"].to_numpy() < base.CHROM_N_SITES
    chr1_pool = genome_pool["SNP"].to_numpy()[chr1_pool_mask]
    target_n = int(round(frac * len(chr1_pool)))

    rng = np.random.default_rng(seed)
    marker_idx = rng.choice(chr1_pool, size=min(target_n, len(chr1_pool)), replace=False)
    marker_idx.sort()
    return marker_idx, causal_idx


def run_scenario_matched(name, params):
    counts = SCENARIO_COUNTS[name]
    reps = base.available_reps(name, N_REPS)
    print(f"=== {name}: density-matched, reps={reps} ===")

    per_rep = []
    for rep in reps:
        marker_idx, causal_idx = matched_indices_for_rep(
            name, rep, params["marker_min_maf"], counts["marker_mode"],
            counts["n_markers"], seed=MARKER_SEED_BASE + rep,
        )
        print(f"  rep{rep}: {len(marker_idx)} markers, {len(causal_idx)} causal")
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
        upsilon_k=("upsilon_k", "mean"),
        upsilon_k_se=("upsilon_k", "sem"),
        n_marker=("n_marker", "mean"),
        n_causal=("n_causal", "mean"),
    ).reset_index()
    return agg


def main():
    outdir = base.OUTDIR
    full_pool = pd.read_csv(outdir / "imperfect_ld_comparison.csv")

    all_rows = []
    for name, params in base.SCENARIOS.items():
        matched = run_scenario_matched(name, params)
        matched.insert(0, "scenario", name)
        matched.to_csv(outdir / f"upsilon_{name}_matched.csv", index=False)

        fp = full_pool[full_pool["scenario"] == name][["k", "upsilon_k", "upsilon_k_se", "sim_upsilon_k", "sim_upsilon_k_se"]]
        fp = fp.rename(columns={"upsilon_k": "full_pool_upsilon_k", "upsilon_k_se": "full_pool_se"})

        merged = matched.merge(fp, on="k", how="left")
        merged["diff_matched_vs_sim"] = merged["upsilon_k"] - merged["sim_upsilon_k"]
        merged["diff_fullpool_vs_sim"] = merged["full_pool_upsilon_k"] - merged["sim_upsilon_k"]
        all_rows.append(merged)

        cols = ["k", "full_pool_upsilon_k", "upsilon_k", "sim_upsilon_k",
                "diff_fullpool_vs_sim", "diff_matched_vs_sim"]
        print(merged[cols].to_string(index=False))
        print()

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(outdir / "imperfect_ld_density_matched_comparison.csv", index=False)
    print(f"Saved {outdir / 'imperfect_ld_density_matched_comparison.csv'}")


if __name__ == "__main__":
    main()
