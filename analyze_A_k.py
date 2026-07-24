#!/usr/bin/env python3
"""Analyze how A_k (r^2 coefficient) varies with k, distance, and allele frequency.

A_k is the coefficient that multiplies r^2 in h_k. It depends on:
- k (via IBD state probabilities p1, rho)
- genetic distance c (via rho_for_k)
- allele frequencies f_a, f_b (via kurtosis = 1 + lambda^2)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "upsilon_estimator"))

from upsilon_estimator._theory import (
    p1_from_k,
    rho_for_k,
    lambda_stat,
    _joint_ibd_probs_no_ibd2,
    _joint_ibd_probs_full_sibs,
)


def compute_A_k(k, f_a, f_b, c_ab):
    """Compute A_k coefficient (multiplies r^2) for given parameters.

    Args:
        k: kinship class
        f_a, f_b: allele frequencies (scalars or arrays)
        c_ab: recombination fraction (scalar or arrays)

    Returns:
        A_k array
    """
    f_a = np.asarray(f_a, dtype=float)
    f_b = np.asarray(f_b, dtype=float)
    c_ab = np.asarray(c_ab, dtype=float)

    lambda_a = lambda_stat(f_a)
    lambda_b = lambda_stat(f_b)
    kappa_a = 1.0 + lambda_a**2
    kappa_b = 1.0 + lambda_b**2

    if k == 2:
        # Full siblings
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


def main():
    """Analyze A_k across different scenarios."""

    # Load panel to get empirical allele frequencies
    import numpy as np
    d = np.load("../data/processed/founders_panel.npz")
    af = d["allele_freq"]
    af = af[(af > 0) & (af < 1)]  # Remove invariant sites

    print("=" * 70)
    print("A_k Analysis: How r² coefficient varies with k, distance, and SFS")
    print("=" * 70)

    # Scenario 1: Mean A_k across the SFS, by k and distance
    print("\n1. Mean A_k across empirical SFS, varying k and genetic distance:")
    print("-" * 70)

    k_values = [2, 3, 4, 6, 8, 10]
    distances_cM = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]

    results = []
    for k in k_values:
        for dist_cM in distances_cM:
            c = 0.5 * (1 - np.exp(-2 * dist_cM / 100))  # Haldane map

            # Draw random pairs from panel SFS
            n_pairs = 5000
            rng = np.random.default_rng(42)
            f_a = rng.choice(af, n_pairs)
            f_b = rng.choice(af, n_pairs)

            A_vals = compute_A_k(k, f_a, f_b, c)

            results.append(
                {
                    "k": k,
                    "distance_cM": dist_cM,
                    "mean_A_k": np.mean(A_vals),
                    "std_A_k": np.std(A_vals),
                    "min_A_k": np.min(A_vals),
                    "max_A_k": np.max(A_vals),
                }
            )

    df = pd.DataFrame(results)

    # Print by k
    for k in k_values:
        print(f"\nk={k}:")
        subset = df[df["k"] == k][["distance_cM", "mean_A_k", "std_A_k"]]
        for _, row in subset.iterrows():
            print(
                f"  dist={row['distance_cM']:>4.1f} cM: "
                f"A_k = {row['mean_A_k']:6.3f} (±{row['std_A_k']:.3f})"
            )

    # Scenario 2: A_k for different frequency regimes
    print("\n2. A_k by frequency regime (at c=0.1 cM):")
    print("-" * 70)

    c = 0.5 * (1 - np.exp(-2 * 0.1 / 100))

    freq_regimes = [
        ("rare", (af > 0) & (af < 0.01)),
        ("low-freq", (af >= 0.01) & (af < 0.05)),
        ("common", (af >= 0.05) & (af < 0.1)),
        ("frequent", (af >= 0.1)),
    ]

    for k in [2, 4, 6, 10]:
        print(f"\nk={k}:")
        for regime_name, mask in freq_regimes:
            af_subset = af[mask]
            if len(af_subset) < 10:
                print(f"  {regime_name:>12}: (n<10, skipped)")
                continue

            n_pairs = min(2000, len(af_subset) ** 2 // 100)
            f_a = rng.choice(af_subset, n_pairs)
            f_b = rng.choice(af_subset, n_pairs)

            A_vals = compute_A_k(k, f_a, f_b, c)
            print(
                f"  {regime_name:>12}: mean A_k = {np.mean(A_vals):6.3f} "
                f"(range {np.min(A_vals):.3f}-{np.max(A_vals):.3f})"
            )

    # Scenario 3: A_k inflation at rare frequencies
    print("\n3. Kurtosis inflation at rare allele frequencies:")
    print("-" * 70)
    print("(A_k grows with kappa = 1 + lambda^2; lambda explodes at rare/common)")

    test_freqs = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5]
    for f in test_freqs:
        lam = lambda_stat(f)
        kappa = 1 + lam**2
        print(
            f"  f={f:>5.3f}: lambda={lam:>8.2f}, kappa={kappa:>10.2f}, "
            f"1 + lambda^2 = {kappa:.2f}x baseline"
        )

    # Scenario 4: Distance decay of A_k (shows rho effect)
    print("\n4. Distance decay of A_k (IBD correlation effect via rho_for_k):")
    print("-" * 70)

    # Use common markers to isolate the rho effect (suppress kappa variation)
    common_af = af[af > 0.1]
    n_pairs = 1000

    for k in [4, 6, 10]:
        print(f"\nk={k}:")
        for dist_cM in [0.01, 0.1, 0.5, 1.0, 5.0]:
            c = 0.5 * (1 - np.exp(-2 * dist_cM / 100))
            f_a = rng.choice(common_af, n_pairs)
            f_b = rng.choice(common_af, n_pairs)
            A_vals = compute_A_k(k, f_a, f_b, c)
            print(f"  {dist_cM:>5.2f} cM: A_k = {np.mean(A_vals):.4f}")


if __name__ == "__main__":
    main()
