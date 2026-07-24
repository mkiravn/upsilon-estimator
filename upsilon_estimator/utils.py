"""Utility functions for paired locus sampling and component estimation."""

import numpy as np

from . import _theory as theory
from . import _panel as zb


def paired_draws(marker_idx, causal_idx, n, rng, panel, gpos):
    """Draw paired loci with common random numbers (marker + causal partner).

    The anchor (marker) is shared between numerator and denominator to reduce variance.
    Partners are drawn independently for each.

    Args:
        marker_idx: Array of marker site indices.
        causal_idx: Array of causal site indices.
        n: Number of pairs to draw.
        rng: numpy.random.Generator instance.
        panel: LazyPanel object.
        gpos: Genetic positions array.

    Returns:
        tuple (num, den) where each is a dict with keys:
            - f_i, f_j: allele frequencies at the two loci
            - r: correlation coefficient
            - c_ab: recombination fraction (from genetic distance via Haldane map function)
            - anchor, partner: locus indices
    """
    anchor = rng.choice(marker_idx, n)

    def draw_partner(pool, seed_offset=0):
        partner_rng = np.random.default_rng(rng.integers(0, 2**31) + seed_offset)
        j = partner_rng.choice(pool, n)
        bad = j == anchor
        while bad.any():
            j[bad] = partner_rng.choice(pool, int(bad.sum()))
            bad = j == anchor
        return j

    def compute_pair_stats(anchor_idx, partner_idx):
        anchor_geno = zb.fetch_sites(panel, anchor_idx)
        partner_geno = zb.fetch_sites(panel, partner_idx)

        r = zb._pairwise_r_from_rows(anchor_geno, partner_geno)
        gd = np.abs(gpos[anchor_idx] - gpos[partner_idx])
        c_ab = theory.haldane_map_function(gd / 100.0)

        return {
            "f_i": panel.allele_freq[anchor_idx],
            "f_j": panel.allele_freq[partner_idx],
            "r": r,
            "c_ab": c_ab,
            "anchor": anchor_idx,
            "partner": partner_idx,
        }

    # Numerator: marker x causal
    causal_partner = draw_partner(causal_idx, seed_offset=1)
    num = compute_pair_stats(anchor, causal_partner)

    # Denominator: marker x marker
    marker_partner = draw_partner(marker_idx, seed_offset=2)
    den = compute_pair_stats(anchor, marker_partner)

    return num, den


def estimate_components(k, pair_data, n_hap):
    """Estimate h_k components (r2_part, cross_part, shared_ibd_part, total) for a pair dataset.

    Args:
        k: Kinship class.
        pair_data: Dict with keys f_i, f_j, r, c_ab (each an array).
        n_hap: Number of haplotypes (for r2 debiasing).

    Returns:
        Dict with keys:
            - r2_part: r^2 component (A_k * r^2)
            - cross_part: cross component (C_k * r * lambda_a * lambda_b)
            - shared_ibd_part: shared IBD component (B_k)
            - total: sum of all three
    """
    f_i = pair_data["f_i"]
    f_j = pair_data["f_j"]
    r = pair_data["r"]
    c_ab = pair_data["c_ab"]

    # Debiased r^2
    r2_debiased = theory.r2_debiased(r**2, n_hap)

    # Get three-way decomposition
    tw = theory.cov_k_three_way(k, f_i, f_j, r, c_ab)

    return {
        "r2_part": tw["r2_part"],
        "cross_part": tw["cross_part"],
        "shared_ibd_part": tw["shared_ibd_part"],
        "total": tw["total"],
    }
