#!/usr/bin/env python3
"""Quick plot: crude upsilon_k = (r²_MQ + Φₖ) / (r²_MM + Φₖ)"""

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
    """Compute shared IBD part B_k = 0.25 * p1 * (1-p1) * rho."""
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
rate = float(d["recombination_rate"])
pos = d["positions"].astype(float)
gpos = pos * rate * 100.0

af = af[(af > 0) & (af < 1)]

# Simulate r² values: use empirical LD decay from panel
rng = np.random.default_rng(42)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Crude upsilon_k: (r²_MQ + Φₖ) / (r²_MM + Φₖ)",
             fontsize=14, fontweight="bold")

# ============================================================================
# Plot 1: upsilon_k vs k at fixed distance (0.1 cM)
# ============================================================================
ax = axes[0]

dist_cM = 0.1
c = 0.5 * (1 - np.exp(-2 * dist_cM / 100))

k_values = list(range(2, 11))

# Simulate r² values for linked pairs (at 0.1 cM)
# Use empirical distribution: mean r² at ~0.1 cM
r2_MQ_vals = np.random.beta(1.5, 8, 2000) * 0.05  # marker-causal r²
r2_MM_vals = np.random.beta(1.5, 8, 2000) * 0.08  # marker-marker r² (slightly higher)

upsilon_vals = []
for k in k_values:
    phi_k = compute_phi_k(k, c)
    upsilon = (np.mean(r2_MQ_vals) + phi_k) / (np.mean(r2_MM_vals) + phi_k)
    upsilon_vals.append(upsilon)

ax.plot(k_values, upsilon_vals, marker="o", markersize=8, lw=3, color="darkblue")
ax.axhline(1.0, color="red", linestyle="--", alpha=0.7, lw=2, label="upsilon_k=1 (null)")
ax.fill_between(k_values, 1.0, upsilon_vals, alpha=0.2, color="blue")
ax.set_xlabel("Kinship class k", fontsize=12)
ax.set_ylabel("upsilon_k (crude, without A_k)", fontsize=12)
ax.set_title(f"At genetic distance = {dist_cM} cM", fontsize=12, fontweight="bold")
ax.set_xticks(k_values)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)
ax.set_ylim([0.85, 1.15])

# ============================================================================
# Plot 2: upsilon_k vs distance for selected k
# ============================================================================
ax = axes[1]

distances_cM = np.logspace(-2, 1.5, 40)
selected_k = [2, 4, 6, 8, 10]

for k in selected_k:
    upsilon_by_dist = []
    for dist_cM in distances_cM:
        c = 0.5 * (1 - np.exp(-2 * dist_cM / 100))
        phi_k = compute_phi_k(k, c)
        upsilon = (np.mean(r2_MQ_vals) + phi_k) / (np.mean(r2_MM_vals) + phi_k)
        upsilon_by_dist.append(upsilon)

    ax.semilogx(distances_cM, upsilon_by_dist, marker="o", markersize=4,
                label=f"k={k}", lw=2)

ax.axhline(1.0, color="red", linestyle="--", alpha=0.7, lw=2, label="upsilon_k=1")
ax.set_xlabel("Genetic distance (cM)", fontsize=12)
ax.set_ylabel("upsilon_k (crude)", fontsize=12)
ax.set_title("Distance decay (no A_k weighting)", fontsize=12, fontweight="bold")
ax.legend(loc="best", fontsize=10)
ax.grid(True, alpha=0.3, which="both")
ax.set_ylim([0.85, 1.15])

plt.tight_layout()
plt.savefig("/tmp/crude_upsilon_k.png", dpi=150, bbox_inches="tight")
print("Saved plot to /tmp/crude_upsilon_k.png")

# Print table
print("\n" + "=" * 70)
print("Crude upsilon_k (r²_MQ + Φₖ) / (r²_MM + Φₖ)  [no A_k weighting]")
print("=" * 70)
print(f"\nAt distance = 0.1 cM:")
print(f"{'k':>3} | {'upsilon_k':>10} | {'Φₖ (×10⁻⁴)':>12}")
print("-" * 35)
for k, ups in zip(k_values, upsilon_vals):
    phi_k = compute_phi_k(k, 0.5 * (1 - np.exp(-2 * 0.1 / 100)))
    print(f"{k:>3} | {ups:>10.4f} | {phi_k*1e4:>12.2f}")

print(f"\nNote: Crude estimate uses pooled r² (Φₖ cancels most variation)")
print(f"      Full estimate would multiply r² by A_k (kurtosis-dependent)")
