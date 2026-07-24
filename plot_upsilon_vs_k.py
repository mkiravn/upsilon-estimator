#!/usr/bin/env python3
"""Plot crude upsilon_k vs k (not vs distance)."""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

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

# Define scenarios
scenarios = [
    ("All vs All (identity)", (af > 0), (af > 0)),
    ("Common (>1%) vs Rare (<5%)", (af > 0.01), (af < 0.05)),
    ("Common (>5%) vs Very Rare (<1%)", (af > 0.05), (af < 0.01)),
]

# Distances to plot
distances_to_show = [0.01, 0.1, 1.0, 10.0]  # cM

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Crude upsilon_k: dependency on kinship class k",
             fontsize=14, fontweight="bold")

k_values = list(range(2, 10))

for scenario_idx, (name, marker_mask, causal_mask) in enumerate(scenarios):

    marker_idx = np.where(marker_mask)[0]
    causal_idx = np.where(causal_mask)[0]

    if len(marker_idx) < 10 or len(causal_idx) < 10:
        continue

    print(f"\n{name}:")

    # Draw actual pairs and compute r²
    n_pairs = min(3000, len(marker_idx) * len(causal_idx) // 100)

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

    print(f"  r²_MQ = {r2_MQ_mean:.6f}, r²_MM = {r2_MM_mean:.6f}")

    # Plot in a 2x2 grid (3 scenarios, use first 3 positions)
    ax = axes[scenario_idx // 2, scenario_idx % 2]

    # For each distance, plot upsilon_k vs k
    for dist_cM in distances_to_show:
        c = 0.5 * (1 - np.exp(-2 * dist_cM / 100))

        upsilon_vals = []
        for k in k_values:
            phi_k = compute_phi_k(k, c)
            upsilon = (r2_MQ_mean + phi_k) / (r2_MM_mean + phi_k)
            upsilon_vals.append(upsilon)

        ax.plot(k_values, upsilon_vals, marker="o", markersize=6,
                label=f"{dist_cM} cM", lw=2)

    ax.axhline(1.0, color="red", linestyle="--", alpha=0.5, lw=1.5, label="upsilon=1")
    ax.set_xlabel("Kinship class k", fontsize=11)
    ax.set_ylabel("upsilon_k (crude)", fontsize=11)
    ax.set_title(name, fontsize=12, fontweight="bold")
    ax.set_xticks(k_values)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="best")
    ax.set_ylim([0.55, 1.15])

# Hide the 4th subplot
axes[1, 1].axis("off")

plt.tight_layout()
plt.savefig("/tmp/upsilon_vs_k.png", dpi=150, bbox_inches="tight")
print("\nSaved plot to /tmp/upsilon_vs_k.png")
