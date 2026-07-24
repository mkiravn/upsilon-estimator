#!/usr/bin/env python3
"""Compare crude upsilon_k vs full (with A_k weighting)."""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent / "upsilon_estimator"))

from upsilon_estimator._theory import (
    p1_from_k,
    rho_for_k,
    lambda_stat,
    cov_k_three_way,
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

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Crude vs Full upsilon_k: Impact of A_k (kurtosis) weighting",
             fontsize=14, fontweight="bold")

for scenario_idx, (name, marker_mask, causal_mask) in enumerate(scenarios):
    ax = axes[scenario_idx]

    marker_idx = np.where(marker_mask)[0]
    causal_idx = np.where(causal_mask)[0]

    if len(marker_idx) < 10 or len(causal_idx) < 10:
        continue

    print(f"\n{name}:")

    # Draw pairs for r² and hk components
    n_pairs = 5000

    rng_local = np.random.default_rng(42 + scenario_idx)  # Different seed per scenario

    # Marker × Causal pairs (numerator)
    marker_anchor = rng_local.choice(marker_idx, n_pairs)
    causal_partner = rng_local.choice(causal_idx, n_pairs)
    marker_geno = fetch_sites(panel, marker_anchor)
    causal_geno = fetch_sites(panel, causal_partner)
    r_MQ = _pairwise_r_from_rows(marker_geno, causal_geno)
    r2_MQ = r_MQ**2
    r2_MQ_mean = np.nanmean(r2_MQ)

    # Marker × Marker pairs (denominator)
    marker_anchor2 = rng_local.choice(marker_idx, n_pairs)
    marker_partner = rng_local.choice(marker_idx, n_pairs)
    bad = marker_anchor2 == marker_partner
    while bad.any():
        marker_partner[bad] = rng_local.choice(marker_idx, bad.sum())
        bad = marker_anchor2 == marker_partner
    marker_geno1 = fetch_sites(panel, marker_anchor2)
    marker_geno2 = fetch_sites(panel, marker_partner)
    r_MM = _pairwise_r_from_rows(marker_geno1, marker_geno2)
    r2_MM = r_MM**2
    r2_MM_mean = np.nanmean(r2_MM)

    print(f"  r²_MQ = {r2_MQ_mean:.6f}, r²_MM = {r2_MM_mean:.6f}")

    crude_vals = []
    full_vals = []

    c_ref = 0.5 * (1 - np.exp(-2 * 0.1 / 100))  # 0.1 cM reference

    for k in k_values:
        # Crude: (r2_MQ + Phi_k) / (r2_MM + Phi_k)
        phi_k = compute_phi_k(k, c_ref)
        crude = (r2_MQ_mean + phi_k) / (r2_MM_mean + phi_k)
        crude_vals.append(crude)

        # Full: mean(h_k_num) / mean(h_k_den)
        # h_k includes A_k * r2 + C_k * r * lambda + B_k (shared IBD)
        # Use correct indexing: marker_anchor are indices into marker_idx, so map back to af indices
        marker_af_num = af[marker_idx[marker_anchor]]
        causal_af_num = af[causal_idx[causal_partner]]
        marker_af_den = af[marker_idx[marker_anchor2]]

        hk_num = cov_k_three_way(k, marker_af_num, causal_af_num,
                                  r_MQ, np.full_like(r_MQ, c_ref), n_haplotypes=hap.shape[1])
        hk_den = cov_k_three_way(k, marker_af_den, marker_af_den,
                                  r_MM, np.full_like(r_MM, c_ref), n_haplotypes=hap.shape[1])
        full = np.nanmean(hk_num["total"]) / np.nanmean(hk_den["total"])
        full_vals.append(full)

    # Plot
    ax.plot(k_values, crude_vals, marker="o", markersize=7, lw=2.5,
            label="Crude (r² pooled)", color="steelblue")
    ax.plot(k_values, full_vals, marker="s", markersize=7, lw=2.5,
            label="Full (with A_k)", color="darkred")
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, lw=1)

    ax.set_xlabel("Kinship class k", fontsize=11)
    ax.set_ylabel("upsilon_k", fontsize=11)
    ax.set_title(name, fontsize=12, fontweight="bold")
    ax.set_xticks(k_values)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="best")
    ax.set_ylim([0.4, 1.15])

plt.tight_layout()
plt.savefig("/tmp/crude_vs_full.png", dpi=150, bbox_inches="tight")
print("\nSaved plot to /tmp/crude_vs_full.png")

# Print comparison
print("\n" + "=" * 80)
print("CRUDE VS FULL COMPARISON (at 0.1 cM)")
print("=" * 80)

for scenario_idx, (name, marker_mask, causal_mask) in enumerate(scenarios):
    marker_idx = np.where(marker_mask)[0]
    causal_idx = np.where(causal_mask)[0]
    if len(marker_idx) < 10 or len(causal_idx) < 10:
        continue

    print(f"\n{name}:")
    print(f"{'k':>3} | {'Crude':>10} | {'Full':>10} | {'Difference':>12} | {'% Change':>10}")
    print("-" * 55)

    # Recompute for display
    n_pairs = 5000
    rng_local = np.random.default_rng(42 + list(scenarios).index((name, marker_mask, causal_mask)))

    marker_anchor = rng_local.choice(marker_idx, n_pairs)
    causal_partner = rng_local.choice(causal_idx, n_pairs)
    marker_geno = fetch_sites(panel, marker_anchor)
    causal_geno = fetch_sites(panel, causal_partner)
    r_MQ = _pairwise_r_from_rows(marker_geno, causal_geno)
    r2_MQ_mean = np.nanmean(r_MQ**2)

    marker_anchor2 = rng_local.choice(marker_idx, n_pairs)
    marker_partner = rng_local.choice(marker_idx, n_pairs)
    bad = marker_anchor2 == marker_partner
    while bad.any():
        marker_partner[bad] = rng_local.choice(marker_idx, bad.sum())
        bad = marker_anchor2 == marker_partner
    marker_geno1 = fetch_sites(panel, marker_anchor2)
    marker_geno2 = fetch_sites(panel, marker_partner)
    r_MM = _pairwise_r_from_rows(marker_geno1, marker_geno2)
    r2_MM_mean = np.nanmean(r_MM**2)

    c_ref = 0.5 * (1 - np.exp(-2 * 0.1 / 100))

    for k in k_values:
        phi_k = compute_phi_k(k, c_ref)
        crude = (r2_MQ_mean + phi_k) / (r2_MM_mean + phi_k)

        marker_af_num = af[marker_idx[marker_anchor]]
        causal_af_num = af[causal_idx[causal_partner]]
        marker_af_den = af[marker_idx[marker_anchor2]]

        hk_num = cov_k_three_way(k, marker_af_num, causal_af_num,
                                  r_MQ, np.full_like(r_MQ, c_ref), n_haplotypes=hap.shape[1])
        hk_den = cov_k_three_way(k, marker_af_den, marker_af_den,
                                  r_MM, np.full_like(r_MM, c_ref), n_haplotypes=hap.shape[1])
        full = np.nanmean(hk_num["total"]) / np.nanmean(hk_den["total"])

        diff = full - crude
        pct_change = (full - crude) / crude * 100 if crude != 0 else 0

        print(f"{k:>3} | {crude:>10.4f} | {full:>10.4f} | {diff:>12.4f} | {pct_change:>9.1f}%")
