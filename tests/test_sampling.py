"""Tests for the scale-aware, density-weighted sampler and block jackknife."""
import numpy as np
import pytest

from upsilon_estimator import _panel as zb
from upsilon_estimator import sampling as S


def _norm_ppf(p):
    """Inverse standard-normal CDF (Acklam's rational approximation, vectorised)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    dd = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
          3.754408661907416e+00]
    p = np.asarray(p, float)
    x = np.zeros_like(p)
    lo, hi = p < 0.02425, p > 1 - 0.02425
    mid = ~(lo | hi)
    q = np.sqrt(-2 * np.log(np.where(lo, p, 1e-300)))
    x = np.where(lo, (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])
                 / ((((dd[0]*q+dd[1])*q+dd[2])*q+dd[3])*q+1), x)
    q = np.sqrt(-2 * np.log(np.where(hi, 1 - p, 1e-300)))
    x = np.where(hi, -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])
                 / ((((dd[0]*q+dd[1])*q+dd[2])*q+dd[3])*q+1), x)
    qm = p - 0.5
    r = qm * qm
    x = np.where(mid, (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*qm
                 / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1), x)
    return x


def _synthetic_panel(n_sites=800, n_hap=400, rho=0.85, seed=0):
    """Panel with decaying LD (AR(1) latent field thresholded to target freqs)."""
    rng = np.random.default_rng(seed)
    f = rng.uniform(0.02, 0.5, n_sites)          # frequency spectrum: common + rare
    thr = _norm_ppf(f)[:, None]
    Z = np.empty((n_sites, n_hap))
    Z[0] = rng.standard_normal(n_hap)
    s = np.sqrt(1 - rho ** 2)
    for i in range(1, n_sites):                   # AR(1) along sites => LD decays with distance
        Z[i] = rho * Z[i - 1] + s * rng.standard_normal(n_hap)
    hap = (Z < thr).astype(np.uint8)              # allele 1 with prob ~ f
    af = hap.mean(axis=1)
    keep = (af > 0) & (af < 1)
    hap, af = hap[keep], af[keep]
    pos = np.sort(rng.choice(np.arange(1, 10_000_000), size=hap.shape[0], replace=False)).astype(float)
    gpos = pos * 1e-8 * 100.0                     # ~1 cM/Mb
    panel = zb.LazyPanel(genotypes=hap, positions=pos, allele_freq=af, genetic_position=gpos)
    return panel, gpos, af


def test_weights_valid_and_self_pairs_excluded():
    panel, gpos, af = _synthetic_panel()
    mk = np.nonzero(af > 0.1)[0]
    cs = np.nonzero((af > 0) & (af < 0.1))[0]
    rng = np.random.default_rng(1)
    num, den = S.stratified_draws(mk, cs, 5000, rng, panel, gpos, n_strata=16)
    for pair in (num, den):
        w = pair["weight"]
        assert np.all(w >= 0)
        assert np.all(np.isfinite(w[w > 0]))
    # marker-marker draws must never pair a locus with itself (weight zeroed)
    self_mask = den["partner"] == den["anchor"]
    assert np.all(den["weight"][self_mask] == 0)


def test_jackknife_se_finite_and_positive():
    panel, gpos, af = _synthetic_panel()
    mk = np.nonzero(af > 0.1)[0]
    cs = np.nonzero((af > 0) & (af < 0.1))[0]
    rng = np.random.default_rng(2)
    num, den = S.stratified_draws(mk, cs, 8000, rng, panel, gpos, n_strata=16)
    ups, se = S.block_jackknife_se(4, num, den, panel.n_haplotypes, n_blocks=20)
    assert np.isfinite(ups) and np.isfinite(se)
    assert se > 0
    assert 0.0 < ups < 2.0


def test_density_invariance_under_thinning():
    """Thinning SNPs should leave upsilon_k within a couple of jackknife SEs."""
    panel, gpos, af = _synthetic_panel(n_sites=1600, n_hap=500)
    rng0 = np.random.default_rng(3)

    def run(mask):
        p = zb.LazyPanel(genotypes=panel.genotypes[mask], positions=panel.positions[mask],
                         allele_freq=af[mask], genetic_position=gpos[mask])
        g = gpos[mask]
        mk = np.nonzero(p.allele_freq > 0.1)[0]
        cs = np.nonzero((p.allele_freq > 0) & (p.allele_freq < 0.1))[0]
        r = np.random.default_rng(4)
        a, b = S.stratified_draws(mk, cs, 20000, r, p, g, n_strata=16)
        return S.block_jackknife_se(4, a, b, p.n_haplotypes, n_blocks=25)

    full = np.ones(len(af), bool)
    thin = rng0.random(len(af)) < 0.4
    u_full, se_full = run(full)
    u_thin, se_thin = run(thin)
    assert abs(u_full - u_thin) < 4 * np.hypot(se_full, se_thin)
