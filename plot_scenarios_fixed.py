#!/usr/bin/env python3
"""Crude upsilon_k for different marker/causal scenarios - FIXED with real r²."""

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

panel = LazyPanel(genotypes=hap, positions=pos[(af > 0) & (af < 1)],
                  allele_freq=af, genetic_position=gpos[(af > 0) & (af < 1)])

rng = np.random.default_rng(42)

# Define scenarios
scenarios = [
    ("All vs All (identity)", (af > 0), (af > 0)),
    ("Common (>1%) vs Rare (<5%)", (af > 0.01), (af < 0.05)),
    ("Common (>5%) vs Very Rare (<1%)", (af > 0.05), (af < 0.01)),
]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Crude upsilon_k: (r²_MQ + Φₖ) / (r²_MM + Φₖ)  [using real r² from data]",
             fontsize=14, fontweight="bold")

k_values = list(range(2, 10))  # k=2 to k=9 only
distances_cM = np.logspace(-2, 1.5, 35)

for idx, (name, marker_mask, causal_mask) in enumerate(scenarios):
    ax = axes[idx]

    marker_idx = np.where(marker_mask)[0]
    causal_idx = np.where(causal_mask)[0]

    if len(marker_idx) < 10 or len(causal_idx) < 10:
        ax.text(0.5, 0.5, f"Insufficient sites",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_title(name, fontsize=12, fontweight="bold")
        continue

    print(f"\n{name}:")
    print(f"  Markers: {len(marker_idx)}, Causal: {len(causal_idx)}")

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

    print(f"  r²_MQ = {r2_MQ_mean:.6f}, r²_MM = {r2_MM_mean:.6f}, ratio = {r2_MQ_mean/r2_MM_mean:.2f}")

    # Plot by k
    for k in k_values:
        upsilon_by_dist = []
        for dist_cM in distances_cM:
            c = 0.5 * (1 - np.exp(-2 * dist_cM / 100))
            phi_k = compute_phi_k(k, c)
            upsilon = (r2_MQ_mean + phi_k) / (r2_MM_mean + phi_k)
            upsilon_by_dist.append(upsilon)

        ax.semilogx(distances_cM, upsilon_by_dist, marker="o" if k in [2, 6, 10] else None,
                    markersize=3, label=f"k={k}", lw=1.5, alpha=0.8)

    ax.axhline(1.0, color="red", linestyle="--", alpha=0.5, lw=1.5, label="upsilon=1")
    ax.set_xlabel("Genetic distance (cM)", fontsize=11)
    ax.set_ylabel("upsilon_k (crude)", fontsize=11)
    ax.set_title(name, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8, ncol=2, loc="best")
    ax.set_ylim([0.6, 1.15])  # Proper axis limits

plt.tight_layout()
plt.savefig("/tmp/upsilon_scenarios_fixed.png", dpi=150, bbox_inches="tight")
print("\nSaved plot to /tmp/upsilon_scenarios_fixed.png")
