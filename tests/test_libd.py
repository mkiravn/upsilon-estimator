"""Gene-drop validation of the linked-IBD (LIBD) split ported into the
upsilon_estimator theory module. Mirrors the ld_sim test suite.

LIBD = linked identity-by-descent (Sved 1971): both loci IBD through the SAME
ancestral haplotype (no recombination between them). The split assigns the
same-lineage (1,1) state the covariance 3/4 r^2 + 1/4 r*lambda_a*lambda_b and
the cross-lineage state r^2/4.
"""
import numpy as np
from upsilon_estimator import _theory as theory


def _halfsib_origins(c, n, rng):
    """Half-sib (k=3, depth 1): both sibs get a gamete from the one shared
    parent (haplotype ids 0/1); origin = which parental haplotype at each locus."""
    def og():
        pick = rng.integers(0, 2, n); rec = rng.random(n) < c; pb = pick ^ rec
        return pick.astype(float), pb.astype(float)
    return og(), og()


def _firstcousin_origins(c, n, rng):
    """First cousin (k=4, depth 2): grandparental haplotype ids 0,1 (GP0) and
    2,3 (GP1). Each parent takes one gamete from each grandparent; each cousin
    takes a gamete mixing its parent's two grandparental haplotypes."""
    def gam(o0, o1):
        pick = rng.integers(0, 2, n); rec = rng.random(n) < c; pb = pick ^ rec
        return np.where(pick == 0, o0[0], o1[0]), np.where(pb == 0, o0[1], o1[1])
    def cousin():
        A = gam((np.zeros(n), np.zeros(n)), (np.ones(n), np.ones(n)))
        B = gam((np.full(n, 2.0), np.full(n, 2.0)), (np.full(n, 3.0), np.full(n, 3.0)))
        return gam(A, B)
    return cousin(), cousin()


def test_p11_libd_split_matches_gene_drop():
    n = 3_000_000
    for k, drop in [(3, _halfsib_origins), (4, _firstcousin_origins)]:
        depth = theory.depth_from_k(k); p1 = float(theory.p1_from_k(k))
        for c in (0.05, 0.2, 0.4):
            rng = np.random.default_rng(k * 100 + int(c * 100))
            (oia, oib), (oja, ojb) = drop(c, n, rng)
            both = (oia == oja) & (oib == ojb)
            same = both & (oia == oib)
            rho = float(theory.rho_for_k(k, np.array([c]))[0])
            p11, p11_same, p11_cross = theory.p11_libd_split(
                np.array([p1]), np.array([rho]), np.array([c]), depth)
            assert abs(both.mean() - float(p11[0])) < 4e-3, (k, c, "p11")
            assert abs(same.mean() - float(p11_same[0])) < 4e-3, (k, c, "p11_same")
            assert float(p11_cross[0]) >= -1e-9, (k, c, "p11_cross>=0")


def test_cross_lineage_11_covariance_is_quarter_r2():
    """IBD at both loci through DIFFERENT ancestral haplotypes -> shared alleles
    independent -> conditional covariance is exactly r^2/4."""
    rng = np.random.default_rng(0)
    f, r = 0.2, 0.4
    D = r * f * (1 - f)
    P = np.array([f * f + D, f * (1 - f) - D, (1 - f) * f - D, (1 - f) * (1 - f) + D])
    s = np.sqrt(2 * f * (1 - f))
    def hap(nn):
        idx = rng.choice(4, size=nn, p=P)
        return (idx < 2).astype(float), ((idx == 0) | (idx == 2)).astype(float)
    def G(x, y): return (x + y - 2 * f) / s
    N = 8_000_000
    Xa = (rng.random(N) < f).astype(float)
    Yb = (rng.random(N) < f).astype(float)
    Hia, Hib = hap(N); Hja, Hjb = hap(N)
    Za = G(Xa, Hia) * G(Xa, Hja); Zb = G(Yb, Hib) * G(Yb, Hjb)
    cov = np.mean(Za * Zb) - np.mean(Za) * np.mean(Zb)
    assert abs(cov - 0.25 * r**2) < 2e-3, (cov, 0.25 * r**2)
