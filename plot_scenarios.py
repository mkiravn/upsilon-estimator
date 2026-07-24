#!/usr/bin/env python3
"""Crude upsilon_k for different marker/causal frequency scenarios."""

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
af = d["allele_freq"]
af = af[(af > 0) & (af < 1)]

rng = np.random.default_rng(42)

# Define scenarios
scenarios = [
    ("Identity (all vs all)", (af > 0), (af > 0), 0.01, 0.02),
    ("Standard (>1% vs <5%)", (af > 0.01), (af < 0.05), 0.01, 0.02),
    ("Stringent (>5% vs <1%)", (af > 0.05), (af < 0.01), 0.01, 0.02),
    ("Intermediate (>0.5% vs <2%)", (af > 0.005), (af < 0.02), 0.01, 0.02),
]

fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle("Crude upsilon_k: Impact of marker/causal frequency thresholds",
             fontsize=15, fontweight="bold")

k_values = list(range(2, 11))
distances_cM = np.logspace(-2, 1.5, 40)

for idx, (name, marker_mask, causal_mask, r2_marker_mean, r2_causal_mean) in enumerate(scenarios):
    ax = axes[idx // 2, idx % 2]

    marker_af = af[marker_mask]
    causal_af = af[causal_mask]

    if len(marker_af) < 10 or len(causal_af) < 10:
        ax.text(0.5, 0.5, f"Insufficient sites:\nmarkers={len(marker_af)}, causal={len(causal_af)}",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_title(name, fontsize=12, fontweight="bold")
        continue

    # Simulate r² for this scenario (with appropriate means)
    r2_marker_vals = np.random.beta(1.5, 8, 2000) * (r2_marker_mean * 2)
    r2_causal_vals = np.random.beta(1.5, 8, 2000) * (r2_causal_mean * 2)

    r2_MM_mean = np.mean(r2_marker_vals)
    r2_MQ_mean = np.mean(r2_causal_vals)

    # Plot by k (one line per k)
    for k in k_values:
        upsilon_by_dist = []
        for dist_cM in distances_cM:
            c = 0.5 * (1 - np.exp(-2 * dist_cM / 100))
            phi_k = compute_phi_k(k, c)
            upsilon = (r2_MQ_mean + phi_k) / (r2_MM_mean + phi_k)
            upsilon_by_dist.append(upsilon)

        ax.semilogx(distances_cM, upsilon_by_dist, marker="o" if k in [2, 6, 10] else None,
                    markersize=3, label=f"k={k}", lw=1.5, alpha=0.8)

    ax.axhline(1.0, color="red", linestyle="--", alpha=0.5, lw=1.5)
    ax.set_xlabel("Genetic distance (cM)", fontsize=11)
    ax.set_ylabel("upsilon_k (crude)", fontsize=11)
    ax.set_title(name, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8, ncol=2, loc="best")
    ax.set_ylim([0.6, 1.1])  # Fixed axis limits

plt.tight_layout()
plt.savefig("/tmp/upsilon_scenarios.png", dpi=150, bbox_inches="tight")
print("Saved plot to /tmp/upsilon_scenarios.png")

# Print scenario summaries
print("\n" + "=" * 80)
print("Crude upsilon_k by scenario (at 0.1 cM distance)")
print("=" * 80)

for name, marker_mask, causal_mask, r2_m, r2_c in scenarios:
    marker_af = af[marker_mask]
    causal_af = af[causal_mask]

    if len(marker_af) < 10 or len(causal_af) < 10:
        print(f"\n{name}:")
        print(f"  (insufficient sites: marker={len(marker_af)}, causal={len(causal_af)})")
        continue

    print(f"\n{name}:")
    print(f"  Markers: n={len(marker_af):5d}, freq range={marker_af.min():.4f}-{marker_af.max():.4f}")
    print(f"  Causal:  n={len(causal_af):5d}, freq range={causal_af.min():.4f}-{causal_af.max():.4f}")

    r2_MM_mean = np.mean(np.random.beta(1.5, 8, 2000) * (r2_m * 2))
    r2_MQ_mean = np.mean(np.random.beta(1.5, 8, 2000) * (r2_c * 2))

    c = 0.5 * (1 - np.exp(-2 * 0.1 / 100))
    print(f"  {'k':>3} | {'upsilon_k':>10}")
    print(f"  " + "-" * 15)
    for k in [2, 4, 6, 8, 10]:
        phi_k = compute_phi_k(k, c)
        ups = (r2_MQ_mean + phi_k) / (r2_MM_mean + phi_k)
        print(f"  {k:>3} | {ups:>10.4f}")
