"""Scale-aware, density-weighted locus-pair sampling and block-jackknife SEs.

The default :func:`~upsilon_estimator.utils.paired_draws` samples the partner
uniformly over the pool, independent of the anchor. On a genome-wide panel almost
every such pair is unlinked (contributing ~0 to both numerator and denominator),
and the short genetic-distance scales that carry the r^2 / cross-term signal are
badly under-sampled. That makes the estimate high-variance and its variance
depends on how much unlinked "filler" the panel contains, so one chromosome and
the whole genome do not behave alike.

This module instead samples the partner *conditional on the anchor*, stratified
by genetic distance on a log grid so every scale is covered, and attaches an
importance weight equal to the number of real pool loci that the sampled pair
represents (post-stratification / Horvitz-Thompson). Self-normalised weighted
means then recover the true density-weighted pair average within the linked
window, which -- because unlinked pairs cancel in the ratio -- equals the
genome-wide upsilon_k. The weights are where marker/causal *density* enters
explicitly (rather than implicitly via uniform sampling).

Derivation of the weight. Draw anchor a ~ Uniform(markers), stratum
s ~ Uniform(non-empty strata of a), partner b ~ Uniform(pool loci of a in s).
Then Pr(a, b in s) = (1/M) * (1/S_a) * (1/N(a,s)), so the Horvitz-Thompson
weight 1/Pr is proportional to S_a * N(a,s). Self-normalising the weighted mean
cancels the constant M and the per-draw 1/n, leaving

    <cov>_pairs  ~  sum_i w_i cov_i / sum_i w_i,   w_i = N(a_i, s_i) * S_{a_i},

an unbiased estimator of the pair average over the linked window.
"""

import numpy as np

from . import _theory as theory
from . import _panel as zb


def _log_strata_edges(d_min, d_max, n_strata):
    """Log-spaced genetic-distance stratum edges (cM), length n_strata + 1."""
    return np.geomspace(d_min, d_max, n_strata + 1)


def _stratified_partners(anchor_gpos, pool_gpos_sorted, edges, rng):
    """Draw one partner per anchor, stratified by genetic distance, with weights.

    Args:
        anchor_gpos: (n,) genetic positions (cM) of the anchors.
        pool_gpos_sorted: (P,) sorted genetic positions of the candidate pool.
        edges: (S+1,) log-spaced distance stratum edges (cM).
        rng: numpy Generator.

    Returns:
        pool_pos: (n,) index into the *sorted pool* of the chosen partner
                  (-1 where the anchor has no pool locus in any stratum).
        weight:   (n,) importance weight N(a,s) * S_a (0 where pool_pos == -1).
    """
    g = np.asarray(anchor_gpos, float)
    n = g.size
    # searchsorted positions of g +/- each edge in the sorted pool
    R = np.searchsorted(pool_gpos_sorted, g[:, None] + edges[None, :])  # (n, S+1)
    L = np.searchsorted(pool_gpos_sorted, g[:, None] - edges[None, :])  # (n, S+1)
    # per-stratum right/left counts, stratum s spans (edges[s-1], edges[s])
    cnt_r = R[:, 1:] - R[:, :-1]      # (n, S)
    cnt_l = L[:, :-1] - L[:, 1:]      # (n, S)
    N = cnt_r + cnt_l                 # (n, S) pool loci per stratum
    nonempty = N > 0
    S_a = nonempty.sum(axis=1)        # non-empty strata per anchor

    # choose a uniformly random non-empty stratum per anchor (argmax of masked noise)
    keys = rng.random((n, N.shape[1]))
    keys[~nonempty] = -1.0
    s = keys.argmax(axis=1)           # (n,)
    rows = np.arange(n)

    N_sel = N[rows, s].astype(float)
    weight = N_sel * S_a
    weight[S_a == 0] = 0.0

    # sample a partner uniformly among the N_sel loci of the chosen stratum
    cr = cnt_r[rows, s]
    Rlo = R[rows, s]                  # right block starts at R[:, s] (== g+edges[s-1])
    Llo = L[rows, s + 1]             # left block starts at L[:, s+1] (== g-edges[s])
    t = np.floor(rng.random(n) * np.maximum(N_sel, 1)).astype(int)
    take_right = t < cr
    pool_pos = np.where(take_right, Rlo + t, Llo + (t - cr))
    pool_pos[S_a == 0] = -1
    return pool_pos, weight


def stratified_draws(marker_idx, causal_idx, n, rng, panel, gpos,
                     d_min=1e-3, d_max=None, n_strata=24):
    """Scale-stratified, density-weighted marker-causal and marker-marker draws.

    Anchors (markers) are shared between numerator and denominator (common random
    numbers). Partners are drawn conditional on the anchor, stratified over
    log-spaced genetic-distance bins, and carry importance weights.

    Returns:
        (num, den): dicts with per-pair arrays f_i, f_j, r, c_ab, weight,
        anchor, partner, anchor_gpos.
    """
    marker_idx = np.sort(np.asarray(marker_idx, int))
    causal_idx = np.sort(np.asarray(causal_idx, int))
    m_gpos = gpos[marker_idx]
    c_gpos = gpos[causal_idx]
    if d_max is None:
        d_max = float(gpos.max() - gpos.min())  # whole map span => all linked pairs
    # Leading [0, d_min] stratum captures ultra-close (high-r^2) pairs that a pure
    # log grid would drop; the rest are log-spaced so every scale is covered.
    edges = np.concatenate([[0.0], _log_strata_edges(d_min, d_max, n_strata)])

    anchor = rng.choice(marker_idx, n)
    anchor_gpos = gpos[anchor]

    def build(pool_idx, pool_gpos):
        pool_pos, w = _stratified_partners(anchor_gpos, pool_gpos, edges, rng)
        ok = pool_pos >= 0
        partner = np.full(n, -1, int)
        partner[ok] = pool_idx[pool_pos[ok]]
        # Drop self-pairs (possible in marker-marker at distance ~0): zero their weight.
        self_pair = partner == anchor
        w = np.where(self_pair, 0.0, w)
        ok = ok & ~self_pair
        ag = zb.fetch_sites(panel, anchor[ok])
        pg = zb.fetch_sites(panel, partner[ok])
        r = np.full(n, np.nan)
        r[ok] = zb._pairwise_r_from_rows(ag, pg)
        gd = np.abs(gpos[anchor] - np.where(ok, gpos[partner], gpos[anchor]))
        return {
            "f_i": panel.allele_freq[anchor],
            "f_j": np.where(ok, panel.allele_freq[partner], np.nan),
            "r": r,
            "c_ab": theory.haldane_map_function(gd / 100.0),
            "weight": w,
            "anchor": anchor,
            "partner": partner,
            "anchor_gpos": anchor_gpos,
        }

    num = build(causal_idx, c_gpos)   # marker x causal
    den = build(marker_idx, m_gpos)   # marker x marker
    return num, den


def _pair_contrib(k, pair, n_hap, include_A_k=False):
    """Per-pair robust contribution r^2 + Phi_k (or A_k r^2 + Phi_k)."""
    r2 = theory.r2_debiased(pair["r"] ** 2, n_hap)
    tw = theory.cov_k_three_way(k, pair["f_i"], pair["f_j"], pair["r"], pair["c_ab"])
    phi = tw["shared_ibd_part"]
    if include_A_k:
        # A_k = r2_part / r2 (recover the coefficient), applied to the debiased r2
        with np.errstate(invalid="ignore", divide="ignore"):
            A_k = np.where(pair["r"] ** 2 > 0, tw["r2_part"] / (pair["r"] ** 2), 1.0)
        return A_k * r2 + phi
    return r2 + phi


def _wmean(x, w):
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not m.any():
        return np.nan
    return np.sum(w[m] * x[m]) / np.sum(w[m])


def _upsilon_from_draws(k, num, den, n_hap, include_A_k=False, idx=None):
    """Self-normalised weighted ratio on an optional subset `idx` (for jackknife)."""
    cn = _pair_contrib(k, num, n_hap, include_A_k)
    cd = _pair_contrib(k, den, n_hap, include_A_k)
    wn, wd = num["weight"], den["weight"]
    if idx is not None:
        cn, cd, wn, wd = cn[idx], cd[idx], wn[idx], wd[idx]
    return _wmean(cn, wn) / _wmean(cd, wd)


def block_jackknife_se(k, num, den, n_hap, n_blocks=200, include_A_k=False):
    """Delete-a-block jackknife SE of upsilon_k over contiguous anchor-gpos blocks.

    Blocking on the anchor position captures the dominant LD dependence between
    pairs (anchors are drawn with replacement and repeat; partners are spread
    across strata). Deletes each block from *both* numerator and denominator (they
    share anchors) and recomputes the full ratio, so the numerator/denominator
    correlation is handled automatically.
    """
    g = num["anchor_gpos"]
    edges = np.quantile(g, np.linspace(0, 1, n_blocks + 1))
    edges[-1] = np.nextafter(edges[-1], np.inf)
    blk = np.clip(np.searchsorted(edges, g, side="right") - 1, 0, n_blocks - 1)

    theta_full = _upsilon_from_draws(k, num, den, n_hap, include_A_k)
    reps = []
    present = np.unique(blk)
    for b in present:
        keep = np.nonzero(blk != b)[0]
        reps.append(_upsilon_from_draws(k, num, den, n_hap, include_A_k, idx=keep))
    reps = np.array(reps, float)
    reps = reps[np.isfinite(reps)]
    B = reps.size
    if B < 2:
        return theta_full, np.nan
    mean = reps.mean()
    se = np.sqrt((B - 1) / B * np.sum((reps - mean) ** 2))
    return theta_full, se
