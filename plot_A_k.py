#!/usr/bin/env python3
"""Visualize how A_k varies with k, distance, and allele frequency."""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent / "upsilon_estimator"))

from upsilon_estimator._theory import (
    p1_from_k,
    rho_for_k,
    lambda_stat,
    _joint_ibd_probs_no_ibd2,
    _joint_ibd_probs_full_sibs,
)


def compute_A_k(k, f_a, f_b, c_ab):
    """Compute A_k coefficient."""
    f_a = np.asarray(f_a, dtype=float)
    f_b = np.asarray(f_b, dtype=float)
    c_ab = np.asarray(c_ab, dtype=float)

    lambda_a = lambda_stat(f_a)
    lambda_b = lambda_stat(f_b)
    kappa_a = 1.0 + lambda_a**2
    kappa_b = 1.0 + lambda_b**2

    if k == 2:
        rho1 = c_ab**2 + (1.0 - c_ab) ** 2
        p = _joint_ibd_probs_full_sibs(rho1)
        A = (
            p["p00"]
            + p["p10"] * (kappa_a + 3.0) / 4.0
            + p["p01"] * (kappa_b + 3.0) / 4.0
            + p["p20"] * (kappa_a + 3.0) / 2.0
            + p["p02"] * (kappa_b + 3.0) / 2.0
            + p["p11"] * 0.75
            + p["p21"] * (kappa_a + 5.0) / 4.0
            + p["p12"] * (kappa_b + 5.0) / 4.0
            + p["p22"]
        )
    else:
        p1 = p1_from_k(k)
        rho = rho_for_k(k, c_ab)
        p00, p10, p11 = _joint_ibd_probs_no_ibd2(p1, rho)
        A = p00 + p10 * (kappa_a + kappa_b + 6.0) / 4.0 + 0.75 * p11

    return A


# Load panel SFS
d = np.load("../data/processed/founders_panel.npz")
af = d["allele_freq"]
af = af[(af > 0) & (af < 1)]

rng = np.random.default_rng(42)

# Create figure with 4 subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("A_k: r² coefficient in h_k decomposition", fontsize=16, fontweight="bold")

# ============================================================================
# Plot 1: A_k vs distance for different k (with common variants)
# ============================================================================
ax = axes[0, 0]

common_af = af[af > 0.1]
distances_cM = np.logspace(-2, 1, 30)  # 0.01 to 10 cM
k_values = [2, 3, 4, 6, 8, 10]

for k in k_values:
    A_means = []
    for dist_cM in distances_cM:
        c = 0.5 * (1 - np.exp(-2 * dist_cM / 100))
        n_pairs = 1000
        f_a = rng.choice(common_af, n_pairs)
        f_b = rng.choice(common_af, n_pairs)
        A_vals = compute_A_k(k, f_a, f_b, c)
        A_means.append(np.mean(A_vals))

    ax.loglog(distances_cM, A_means, marker="o", markersize=4, label=f"k={k}", lw=2)

ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="A_k=1")
ax.set_xlabel("Genetic distance (cM)", fontsize=11)
ax.set_ylabel("Mean A_k (common variants)", fontsize=11)
ax.set_title("A_k decay with distance (by kinship class)", fontsize=12, fontweight="bold")
ax.legend(loc="best")
ax.grid(True, alpha=0.3)

# ============================================================================
# Plot 2: A_k vs allele frequency at c=0.1 cM
# ============================================================================
ax = axes[0, 1]

freqs = np.logspace(-3.3, -0.01, 50)  # 0.0005 to 0.98
freqs = freqs[freqs < 1]

for k in [2, 4, 6, 10]:
    A_vals = []
    for f in freqs:
        A = compute_A_k(k, f, f, 0.5 * (1 - np.exp(-2 * 0.1 / 100)))
        A_vals.append(A)

    ax.loglog(freqs, A_vals, marker="o", markersize=3, label=f"k={k}", lw=2)

ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
ax.set_xlabel("Allele frequency", fontsize=11)
ax.set_ylabel("A_k (at 0.1 cM distance)", fontsize=11)
ax.set_title("A_k inflation at rare alleles (kurtosis effect)", fontsize=12, fontweight="bold")
ax.legend(loc="best")
ax.grid(True, alpha=0.3, which="both")

# ============================================================================
# Plot 3: Kurtosis (1 + lambda^2) vs allele frequency
# ============================================================================
ax = axes[1, 0]

freqs = np.linspace(0.001, 0.5, 100)
kappa = 1 + lambda_stat(freqs) ** 2

ax.semilogy(freqs, kappa, color="darkred", lw=3, label="κ = 1 + λ²")
ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
ax.fill_between(freqs, 1, kappa, alpha=0.2, color="red")
ax.set_xlabel("Allele frequency", fontsize=11)
ax.set_ylabel("Kurtosis κ", fontsize=11)
ax.set_title("Kurtosis grows at rare/common frequencies", fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.3, which="both")

# Add annotations
ax.annotate("rare variants\nhigh kurtosis", xy=(0.01, 100), xytext=(0.02, 1000),
            arrowprops=dict(arrowstyle="->" , color="red", lw=1.5),
            fontsize=10, color="red", fontweight="bold")
ax.annotate("common variants\nlow kurtosis", xy=(0.35, 1.5), xytext=(0.2, 10),
            arrowprops=dict(arrowstyle="->", color="blue", lw=1.5),
            fontsize=10, color="blue", fontweight="bold")

# ============================================================================
# Plot 4: 2D heatmap of A_k(k, distance)
# ============================================================================
ax = axes[1, 1]

k_values_heat = [2, 3, 4, 5, 6, 7, 8, 9, 10]
distances_heat = np.logspace(-2, 1, 20)  # 0.01 to 10 cM

A_matrix = np.zeros((len(k_values_heat), len(distances_heat)))

for i, k in enumerate(k_values_heat):
    for j, dist_cM in enumerate(distances_heat):
        c = 0.5 * (1 - np.exp(-2 * dist_cM / 100))
        n_pairs = 500
        f_a = rng.choice(common_af, n_pairs)
        f_b = rng.choice(common_af, n_pairs)
        A_vals = compute_A_k(k, f_a, f_b, c)
        A_matrix[i, j] = np.mean(A_vals)

im = ax.imshow(A_matrix, aspect="auto", cmap="RdYlBu_r", origin="lower",
               extent=[np.log10(distances_heat[0]), np.log10(distances_heat[-1]), 2, 10],
               vmin=0.8, vmax=2.5)

ax.set_xlabel("log₁₀(distance in cM)", fontsize=11)
ax.set_ylabel("Kinship class k", fontsize=11)
ax.set_title("A_k heatmap: distance × kinship (common variants)", fontsize=12, fontweight="bold")
ax.set_yticks(k_values_heat)

cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Mean A_k", fontsize=10)

plt.tight_layout()
plt.savefig("/tmp/A_k_analysis.png", dpi=150, bbox_inches="tight")
print("Saved plot to /tmp/A_k_analysis.png")
plt.show()

# Print summary statistics
print("\nA_k Summary Statistics")
print("=" * 70)
print(f"Min A_k across grid: {A_matrix.min():.3f}")
print(f"Max A_k across grid: {A_matrix.max():.3f}")
print(f"Median A_k: {np.median(A_matrix):.3f}")
print(f"\nFor close distances (0.01 cM):")
print(f"  k=2: A_k = {A_matrix[0, 0]:.3f}")
print(f"  k=10: A_k = {A_matrix[8, 0]:.3f}")
print(f"  Ratio (k=2/k=10): {A_matrix[0, 0] / A_matrix[8, 0]:.1f}×")
