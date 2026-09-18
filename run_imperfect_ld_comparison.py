"""Compare estimator upsilon_k against simulated slope regressions for the
imperfect-LD scenarios in results/imperfect_ld/.

For each scenario, builds explicit marker/causal index sets from the shared
founders panel (results/founders/founders_n5000_gbr.zarr, chromosome 1 --
the scale-aware estimator is density/scale-invariant, so a single chromosome
reproduces the genome-wide value; see upsilon-estimator/README.md) according
to that scenario's marker_min_maf / causal_max_maf / disjoint_markers, runs
estimate_upsilon_k_scale_aware with the scenario's Speed alpha, and joins the
result against results/imperfect_ld/{scenario}/summaries/slope_summary.csv
(predictor=GRM_marker, response=GRM_causal), which is the simulation-side
empirical upsilon_k.

Two ways to build the causal set (USE_REAL_CAUSAL toggles which):
  - MAF-threshold pool (default): every chr1 site passing causal_max_maf.
    Not resampled down to n_causal as the Snakemake pipeline does -- the
    estimator's density invariance means this doesn't bias upsilon_k, only
    its Monte Carlo noise floor.
  - Real per-replicate causal set: results/imperfect_ld/{scenario}/effects/
    rep{rep}.effects.tsv persists an is_causal flag per SNP, and its SNP
    column is the *global* founders-zarr site index (verified against the
    panel's .bim/.pos_cM: chr1 spans SNP IDs 0..103207, chr2 starts at
    103208, etc.), so for a chr1-restricted run these IDs are directly usable
    as causal indices -- the exact SNPs that replicate's simulation used,
    not an approximation. Marker identity isn't recoverable this way (the
    sampled marker SNP list is written to a deleted tmpdir in the Snakemake
    rule), so markers are always built from the MAF-threshold pool, using
    that replicate's real causal set for disjointness. Runs one estimate per
    replicate and reports the across-replicate mean/SE alongside the
    jackknife SE.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from upsilon_estimator import estimate_upsilon_k_scale_aware
from upsilon_estimator.estimators import load_panel

REPO_ROOT = Path(__file__).resolve().parent.parent
PANEL = REPO_ROOT / "results" / "founders" / "founders_n5000_gbr.zarr"
SIM_ROOT = REPO_ROOT / "results" / "imperfect_ld"
OUTDIR = Path(__file__).resolve().parent / "results"
OUTDIR.mkdir(exist_ok=True)

CHROMOSOME = 1
CHROM_N_SITES = 103208  # chr1 site count in the founders zarr / global SNP-ID offset for chr2
K_VALUES = list(range(2, 10))
N_DRAWS = 30_000
N_BLOCKS = 30
SEED = 1

USE_REAL_CAUSAL = False   # optional: use each replicate's actual causal SNP set (see docstring)
REAL_CAUSAL_N_REPS = 3    # how many replicates to average over when USE_REAL_CAUSAL=True

# name -> (marker_min_maf, causal_max_maf, disjoint_markers, alpha)
# sc*_gcta / sc1_alpha_* from config/config.yaml; sc1_maf5 / sc4_maf5 only
# exist under those names in config/config_figures.yaml (same parameters as
# sc3_maf05_gcta / sc4_maf05_gcta, but a distinct simulation replicate batch).
SCENARIOS = {
    "sc0_gcta":       dict(marker_min_maf=0.00, causal_max_maf=None, disjoint=False, alpha=-1.0),
    "sc1_gcta":       dict(marker_min_maf=0.00, causal_max_maf=None, disjoint=False, alpha=-1.0),
    "sc3_gcta":       dict(marker_min_maf=0.01, causal_max_maf=None, disjoint=False, alpha=-1.0),
    "sc4_gcta":       dict(marker_min_maf=0.01, causal_max_maf=0.05, disjoint=True,  alpha=-1.0),
    "sc3_maf05_gcta": dict(marker_min_maf=0.05, causal_max_maf=None, disjoint=False, alpha=-1.0),
    "sc4_maf05_gcta": dict(marker_min_maf=0.05, causal_max_maf=0.05, disjoint=True,  alpha=-1.0),
    "sc1_maf5":       dict(marker_min_maf=0.05, causal_max_maf=None, disjoint=False, alpha=-1.0),
    "sc4_maf5":       dict(marker_min_maf=0.05, causal_max_maf=0.05, disjoint=True,  alpha=-1.0),
    "sc1_alpha_m05":  dict(marker_min_maf=0.00, causal_max_maf=None, disjoint=False, alpha=-0.5),
    "sc1_alpha_m15":  dict(marker_min_maf=0.00, causal_max_maf=None, disjoint=False, alpha=-1.5),
}


def build_indices(af, marker_min_maf, causal_max_maf, disjoint, real_causal_idx=None):
    """real_causal_idx overrides the MAF-derived causal set (still used for
    marker disjointness) when provided -- see USE_REAL_CAUSAL."""
    maf = np.minimum(af, 1.0 - af)
    all_idx = np.arange(len(maf))

    if real_causal_idx is not None:
        causal_idx = real_causal_idx
    elif causal_max_maf is None:
        causal_idx = np.nonzero(maf > 0)[0]
    else:
        causal_idx = np.nonzero((maf > 0) & (maf < causal_max_maf))[0]

    marker_mask = maf > marker_min_maf
    if disjoint:
        marker_mask &= ~np.isin(all_idx, causal_idx)
    marker_idx = np.nonzero(marker_mask)[0]
    return marker_idx, causal_idx


def load_real_causal_indices(scenario, rep, max_site_id=CHROM_N_SITES):
    """Real causal SNP set for one replicate, restricted to chromosome 1's
    global SNP-ID range [0, max_site_id)."""
    path = SIM_ROOT / scenario / "effects" / f"rep{rep}.effects.tsv"
    df = pd.read_csv(path, sep="\t", usecols=["SNP", "is_causal"])
    df["SNP"] = df["SNP"].astype(int)
    causal = df.loc[df["is_causal"] & (df["SNP"] < max_site_id), "SNP"].to_numpy()
    return np.sort(causal)


def available_reps(scenario, n_wanted):
    effects_dir = SIM_ROOT / scenario / "effects"
    reps = sorted(
        int(p.stem.split(".")[0].replace("rep", ""))
        for p in effects_dir.glob("rep*.effects.tsv")
    )
    return reps[:n_wanted]


def load_sim_slopes(scenario):
    path = SIM_ROOT / scenario / "summaries" / "slope_summary.csv"
    df = pd.read_csv(path)
    df = df[(df["predictor"] == "GRM_marker") & (df["response"] == "GRM_causal")]
    df = df[np.isfinite(pd.to_numeric(df["k"], errors="coerce"))].copy()
    df["k"] = df["k"].astype(int)
    return df.set_index("k")[["slope_mean", "slope_se"]].rename(
        columns={"slope_mean": "sim_upsilon_k", "slope_se": "sim_upsilon_k_se"}
    )


def run_scenario_maf_pool(af, name, params):
    marker_idx, causal_idx = build_indices(
        af, params["marker_min_maf"], params["causal_max_maf"], params["disjoint"]
    )
    print(f"=== {name}: alpha={params['alpha']}, marker_min_maf={params['marker_min_maf']}, "
          f"causal_max_maf={params['causal_max_maf']}, disjoint={params['disjoint']} "
          f"-> {len(marker_idx)} markers, {len(causal_idx)} causal (MAF-pool) ===")

    return estimate_upsilon_k_scale_aware(
        str(PANEL),
        marker_indices=marker_idx,
        causal_indices=causal_idx,
        n_draws=N_DRAWS,
        seed=SEED,
        k_values=K_VALUES,
        chromosome=CHROMOSOME,
        n_blocks=N_BLOCKS,
        alpha=params["alpha"],
    )


def run_scenario_real_causal(af, name, params):
    reps = available_reps(name, REAL_CAUSAL_N_REPS)
    print(f"=== {name}: real-causal mode, reps={reps} ===")

    per_rep = []
    for rep in reps:
        real_causal = load_real_causal_indices(name, rep)
        marker_idx, causal_idx = build_indices(
            af, params["marker_min_maf"], params["causal_max_maf"], params["disjoint"],
            real_causal_idx=real_causal,
        )
        print(f"  rep{rep}: {len(marker_idx)} markers, {len(causal_idx)} real causal")
        est = estimate_upsilon_k_scale_aware(
            str(PANEL),
            marker_indices=marker_idx,
            causal_indices=causal_idx,
            n_draws=N_DRAWS,
            seed=SEED,
            k_values=K_VALUES,
            chromosome=CHROMOSOME,
            n_blocks=N_BLOCKS,
            alpha=params["alpha"],
        )
        est["rep"] = rep
        per_rep.append(est)

    all_reps = pd.concat(per_rep, ignore_index=True)
    agg = all_reps.groupby("k").agg(
        upsilon_k=("upsilon_k", "mean"),
        upsilon_k_se=("upsilon_k", "sem"),          # across-replicate SE
        upsilon_k_jk_se=("upsilon_k_se", "mean"),   # mean jackknife SE, for reference
        n_marker=("n_marker", "mean"),
        n_causal=("n_causal", "mean"),
    ).reset_index()
    return agg


def main():
    print(f"Loading founders panel (chromosome {CHROMOSOME}): {PANEL}")
    panel_data = load_panel(str(PANEL), chromosome=CHROMOSOME)
    af = panel_data["af"]
    print(f"  {panel_data['n_sites']} sites, {panel_data['n_hap']} haplotypes")
    print(f"  mode: {'real per-replicate causal sets' if USE_REAL_CAUSAL else 'MAF-threshold pool'}")

    all_rows = []
    for name, params in SCENARIOS.items():
        if USE_REAL_CAUSAL:
            est = run_scenario_real_causal(af, name, params)
        else:
            est = run_scenario_maf_pool(af, name, params)

        est.insert(0, "scenario", name)
        suffix = "_realcausal" if USE_REAL_CAUSAL else ""
        est.to_csv(OUTDIR / f"upsilon_{name}{suffix}.csv", index=False)

        sim = load_sim_slopes(name)
        merged = est.set_index("k").join(sim, how="left")
        merged["diff"] = merged["upsilon_k"] - merged["sim_upsilon_k"]
        merged = merged.reset_index()
        all_rows.append(merged)

        cols = ["k", "upsilon_k", "upsilon_k_se", "sim_upsilon_k", "sim_upsilon_k_se", "diff"]
        print(merged[cols].to_string(index=False))
        print()

    combined = pd.concat(all_rows, ignore_index=True)
    suffix = "_realcausal" if USE_REAL_CAUSAL else ""
    combined.to_csv(OUTDIR / f"imperfect_ld_comparison{suffix}.csv", index=False)
    print(f"Saved combined comparison to {OUTDIR / f'imperfect_ld_comparison{suffix}.csv'}")


if __name__ == "__main__":
    main()
