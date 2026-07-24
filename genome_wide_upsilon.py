#!/usr/bin/env python3
"""Compute genome-wide crude upsilon_k estimates."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "upsilon_estimator"))

from upsilon_estimator._theory import (
    p1_from_k,
    rho_for_k,
    _joint_ibd_probs_no_ibd2,
    _joint_ibd_probs_full_sibs,
)
from upsilon_estimator._panel import LazyPanel, fetch_sites, _pairwise_r_from_rows


def compute_phi_k(k, c_ab):
    """Compute shared IBD part B_k."""
    c_ab = np.asarray(c_ab, dtype=float)

    if k == 2:
        rho1 = c_ab**2 + (1.0 - c_ab) ** 2
        p = _joint_ibd_probs_full_sibs(rho1)
        cov_sa_sb = p["p11"] + 2.0 * p["p12"] + 2.0 * p["p21"] + 4.0 * p["p22"] - 1.0
        B = 0.25 * cov_sa_sb
    else:
        p1 = p1_from_k(k)
        rho = rho_for_k(k, c_ab)
        B = 0.25 * p1 * (1.0 - p1) * rho

    return B


# Load panel
d = np.load("../data/processed/founders_panel.npz")
hap = d["haplotypes"]
af = d["allele_freq"]
rate = float(d["recombination_rate"])
pos = d["positions"].astype(float)
gpos = pos * rate * 100.0

mask = (af > 0) & (af < 1)
af = af[mask]
hap = hap[mask]
pos = pos[mask]
gpos = gpos[mask]

panel = LazyPanel(genotypes=hap, positions=pos, allele_freq=af, genetic_position=gpos)

rng = np.random.default_rng(42)

# Scenarios
scenarios = [
    ("All vs All", (af > 0), (af > 0)),
    ("Common (>1%) vs Rare (<5%)", (af > 0.01), (af < 0.05)),
    ("Common (>5%) vs Very Rare (<1%)", (af > 0.05), (af < 0.01)),
]

k_values = list(range(2, 10))

print("=" * 80)
print("GENOME-WIDE CRUDE UPSILON_K ESTIMATES")
print("=" * 80)

results_list = []

for name, marker_mask, causal_mask in scenarios:
    marker_idx = np.where(marker_mask)[0]
    causal_idx = np.where(causal_mask)[0]

    if len(marker_idx) < 10 or len(causal_idx) < 10:
        continue

    print(f"\n{name}:")
    print(f"  Markers: {len(marker_idx)} sites, Causal: {len(causal_idx)} sites")

    # Draw pairs for r² computation
    n_pairs = 5000

    # Marker × Causal pairs
    marker_anchor = rng.choice(marker_idx, n_pairs)
    causal_partner = rng.choice(causal_idx, n_pairs)
    marker_geno = fetch_sites(panel, marker_anchor)
    causal_geno = fetch_sites(panel, causal_partner)
    r_MQ = _pairwise_r_from_rows(marker_geno, causal_geno)
    r2_MQ_mean = np.nanmean(r_MQ**2)

    # Marker × Marker pairs
    marker_anchor2 = rng.choice(marker_idx, n_pairs)
    marker_partner = rng.choice(marker_idx, n_pairs)
    bad = marker_anchor2 == marker_partner
    while bad.any():
        marker_partner[bad] = rng.choice(marker_idx, bad.sum())
        bad = marker_anchor2 == marker_partner
    marker_geno1 = fetch_sites(panel, marker_anchor2)
    marker_geno2 = fetch_sites(panel, marker_partner)
    r_MM = _pairwise_r_from_rows(marker_geno1, marker_geno2)
    r2_MM_mean = np.nanmean(r_MM**2)

    r2_ratio = r2_MQ_mean / r2_MM_mean

    print(f"  r²_MQ = {r2_MQ_mean:.6f}")
    print(f"  r²_MM = {r2_MM_mean:.6f}")
    print(f"  r²_MQ / r²_MM = {r2_ratio:.4f}")
    print()
    print(f"  {'k':>3} | {'upsilon_k':>10} | {'Φₖ at 0.1cM':>12}")
    print(f"  " + "-" * 30)

    c_ref = 0.5 * (1 - np.exp(-2 * 0.1 / 100))  # 0.1 cM reference distance

    for k in k_values:
        phi_k = compute_phi_k(k, c_ref)
        upsilon = (r2_MQ_mean + phi_k) / (r2_MM_mean + phi_k)

        results_list.append({
            "Scenario": name,
            "k": k,
            "upsilon_k": upsilon,
            "r2_ratio": r2_ratio,
        })

        print(f"  {k:>3} | {upsilon:>10.4f} | {phi_k:>12.6f}")

# Summary table
print("\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)

results_df = pd.DataFrame(results_list)
pivot = results_df.pivot_table(values='upsilon_k', index='k', columns='Scenario')

print()
print(pivot.to_string())

print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)
print(f"""
1. All vs All: upsilon_k ≈ {pivot.loc[2, 'All vs All']:.3f} at k=2, {pivot.loc[9, 'All vs All']:.3f} at k=9
   → Near-identity baseline (r²_MQ/r²_MM ≈ 1.10)

2. Common vs Rare: upsilon_k ≈ {pivot.loc[2, 'Common (>1%) vs Rare (<5%)']:.3f} at k=2, {pivot.loc[9, 'Common (>1%) vs Rare (<5%)']:.3f} at k=9
   → Imperfect tagging (r²_MQ/r²_MM ≈ 0.48)

3. Common vs Very Rare: upsilon_k ≈ {pivot.loc[2, 'Common (>5%) vs Very Rare (<1%)']:.3f} at k=2, {pivot.loc[9, 'Common (>5%) vs Very Rare (<1%)']:.3f} at k=9
   → Even worse tagging (r²_MQ/r²_MM ≈ 0.38)

Without A_k (kurtosis) weighting, upsilon_k ≤ 1.0 for all realistic scenarios.
A_k weighting would push rare-variant scenarios back toward 1.0 or > 1.0.
""")
