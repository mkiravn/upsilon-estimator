"""Compute the k=Inf (unrelated-pairs) reference for each scenario:
upsilon_Inf = r2_MQ / r2_MM, using the same stratified/density-weighted draws
and debiasing as estimate_upsilon_k_scale_aware -- as k->Inf the shared-IBD
term Phi_k vanishes (see sampling._pair_contrib), so the per-k ratio reduces
exactly to this population-level (unrelated) LD ratio. Uses the same
density-matched marker/causal sets as run_density_matched_comparison.py,
averaged over the same replicates.

Also pulls each scenario's own simulated k=Inf slope from slope_summary.csv
(predictor=GRM_marker, response=GRM_causal) for comparison -- theory vs the
simulation's own empirical "unrelated pairs" slope.
"""
import numpy as np
import pandas as pd

import run_density_matched_comparison as dm
import run_imperfect_ld_comparison as base
from upsilon_estimator import sampling as _samp
from upsilon_estimator import _theory as theory
from upsilon_estimator.estimators import load_panel


def r2_mm_mq_ratio(panel, af, gpos, n_hap, marker_idx, causal_idx, n_draws, seed, d_max=None, n_strata=24):
    rng = np.random.default_rng(seed)
    num, den = _samp.stratified_draws(
        marker_idx, causal_idx, n_draws, rng, panel, gpos,
        d_max=d_max, n_strata=n_strata,
    )
    r2_mm = theory.r2_debiased(den["r"] ** 2, n_hap)
    r2_mq = theory.r2_debiased(num["r"] ** 2, n_hap)
    r2_mm_w = _samp._wmean(r2_mm, den["weight"])
    r2_mq_w = _samp._wmean(r2_mq, num["weight"])
    return r2_mm_w, r2_mq_w, r2_mq_w / r2_mm_w


def load_sim_inf(scenario):
    path = base.SIM_ROOT / scenario / "summaries" / "slope_summary.csv"
    df = pd.read_csv(path)
    df = df[(df["predictor"] == "GRM_marker") & (df["response"] == "GRM_causal")]
    inf_row = df[~np.isfinite(pd.to_numeric(df["k"], errors="coerce"))]
    if inf_row.empty:
        return np.nan, np.nan
    return float(inf_row["slope_mean"].iloc[0]), float(inf_row["slope_se"].iloc[0])


def main():
    panel_data = load_panel(str(base.PANEL), chromosome=base.CHROMOSOME)
    panel, af, gpos, n_hap = panel_data["panel"], panel_data["af"], panel_data["gpos"], panel_data["n_hap"]

    rows = []
    for name, params in base.SCENARIOS.items():
        counts = dm.SCENARIO_COUNTS[name]
        reps = base.available_reps(name, dm.N_REPS)
        ratios = []
        for rep in reps:
            marker_idx, causal_idx = dm.matched_indices_for_rep(
                name, rep, params["marker_min_maf"], counts["marker_mode"],
                counts["n_markers"], seed=dm.MARKER_SEED_BASE + rep,
            )
            _, _, ratio = r2_mm_mq_ratio(
                panel, af, gpos, n_hap, marker_idx, causal_idx,
                n_draws=base.N_DRAWS, seed=base.SEED,
            )
            ratios.append(ratio)
        ratios = np.array(ratios)
        theory_inf = ratios.mean()
        theory_inf_se = ratios.std(ddof=1) / np.sqrt(len(ratios)) if len(ratios) > 1 else np.nan

        sim_inf, sim_inf_se = load_sim_inf(name)
        print(f"{name}: theory upsilon_Inf = r2_MQ/r2_MM = {theory_inf:.4f} (se {theory_inf_se:.4f}), "
              f"sim k=Inf slope = {sim_inf:.4f} (se {sim_inf_se:.4f})")
        rows.append(dict(scenario=name, theory_upsilon_inf=theory_inf, theory_upsilon_inf_se=theory_inf_se,
                          sim_upsilon_inf=sim_inf, sim_upsilon_inf_se=sim_inf_se))

    out = pd.DataFrame(rows)
    out.to_csv(base.OUTDIR / "inf_reference.csv", index=False)
    print(f"Saved {base.OUTDIR / 'inf_reference.csv'}")


if __name__ == "__main__":
    main()
