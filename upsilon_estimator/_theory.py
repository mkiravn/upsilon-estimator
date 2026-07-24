"""Vendored theory functions from ld_sim for computing h_k components and upsilon_k."""

import numpy as np

# Polynomial coefficients for rho_k (recombination-decay term per kinship class)
RHO_POLY_COEFFS = {
    3: [4.0, -4.0, 1.0],
    4: [2.6666666666666665, -8.0, 10.0, -5.333333333333333, 1.0],
    5: [2.2857142857142856, -6.857142857142857, 8.0, -4.571428571428571, 1.0],
    6: [2.1333333333333333, -10.666666666666666, 22.933333333333334,
        -26.666666666666668, 17.6, -6.4, 1.0],
    7: [2.064516129032258, -10.32258064516129, 21.677419354838708,
        -24.774193548387096, 16.516129032258064, -6.193548387096774, 1.0],
    8: [2.0317460317460316, -14.222222222222221, 44.19047619047619,
        -79.23809523809524, 89.39682539682539, -65.01587301587301,
        29.96825396825397, -8.126984126984127, 1.0],
    9: [2.015748031496063, -14.11023622047244, 43.338582677165356,
        -76.5984251968504, 85.66929133858268, -62.488188976377955,
        29.228346456692915, -8.062992125984252, 1.0],
    10: [2.007843137254902, -18.070588235294117, 73.78823529411764,
         -179.7019607843137, 288.62745098039215, -319.2470588235294,
         246.46274509803922, -131.51372549019607, 46.68235294117647,
         -10.03921568627451, 1.0],
}


def p1_from_meioses(m):
    """P(S=1 | k) = 2^-m for relationship class at "distance" m meioses."""
    return 2.0 ** (-np.asarray(m, dtype=float))


def p1_from_k(k):
    """P(S=1 | k) using relationship-degree index k (k = m + 2)."""
    return p1_from_meioses(np.asarray(k, dtype=float) - 2)


def rho_for_k(k, c_ab):
    """Exact two-locus recombination-decay term rho(c_ab) for relationship class k.

    Replaces the naive (1-c_ab)**m with exact polynomial for collateral relationships.
    """
    if k not in RHO_POLY_COEFFS:
        raise ValueError(
            f"rho_for_k only supports k in {sorted(RHO_POLY_COEFFS)}; "
            "k=2 is full siblings (handled separately)"
        )
    c_ab = np.asarray(c_ab, dtype=float)
    return np.polyval(RHO_POLY_COEFFS[k], c_ab)


def haldane_map_function(genetic_distance_morgans):
    """Haldane's mapping function: c = 0.5*(1 - exp(-2*d))."""
    d = np.asarray(genetic_distance_morgans, dtype=float)
    return 0.5 * (1.0 - np.exp(-2.0 * d))


def lambda_stat(f):
    """lambda_l = (1 - 2*f_l) / sqrt(f_l * (1 - f_l)) -- skewness statistic."""
    f = np.asarray(f, dtype=float)
    return (1.0 - 2.0 * f) / np.sqrt(f * (1.0 - f))


def r2_debiased(r2, n_haplotypes):
    """Unbiased estimator of population r^2, removing finite-sample 1/(n-2) bias."""
    r2 = np.asarray(r2, dtype=float)
    return r2 - (1.0 - r2) / (n_haplotypes - 2.0)


def _joint_ibd_probs_no_ibd2(p1, rho):
    """P(S_a, S_b | k) for S_a, S_b in {0,1}, given p1 and rho."""
    p11 = p1**2 + p1 * (1 - p1) * rho
    p00 = (1 - p1) ** 2 + p1 * (1 - p1) * rho
    p10 = p1 * (1 - p1) * (1 - rho)
    return p00, p10, p11


def _joint_ibd_probs_full_sibs(rho1):
    """P(S_a, S_b | full sibs) for S_a, S_b in {0,1,2} as function of rho1."""
    rho1 = np.asarray(rho1, dtype=float)
    q11 = rho1 / 2.0
    q10 = (1.0 - rho1) / 2.0

    p00 = q11**2
    p22 = q11**2
    p01 = 2.0 * q10 * q11
    p02 = q10**2
    p11 = 2.0 * q11**2 + 2.0 * q10**2

    return {
        "p00": p00,
        "p01": p01,
        "p10": p01,
        "p02": p02,
        "p20": p02,
        "p11": p11,
        "p12": p01,
        "p21": p01,
        "p22": p22,
    }


def cov_k_three_way(k, f_a, f_b, r_ab, c_ab, n_haplotypes=None):
    """Regroup Cov_k(Z_a, Z_b) by r^2, cross, and shared-IBD components.

    Returns dict with r2_part, cross_part, shared_ibd_part, total, and coefficients A, C, B.
    """
    f_a = np.asarray(f_a, dtype=float)
    f_b = np.asarray(f_b, dtype=float)
    r = np.asarray(r_ab, dtype=float)
    c_ab = np.asarray(c_ab, dtype=float)
    r2 = r**2 if n_haplotypes is None else r2_debiased(r**2, n_haplotypes)

    lambda_a, lambda_b = lambda_stat(f_a), lambda_stat(f_b)
    kappa_a, kappa_b = 1.0 + lambda_a**2, 1.0 + lambda_b**2

    if k == 2:
        # Full siblings: handle IBD2
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
        C = 0.25 * (p["p11"] + p["p21"] + p["p12"]) + 0.5 * p["p22"]
        cov_sa_sb = p["p11"] + 2.0 * p["p12"] + 2.0 * p["p21"] + 4.0 * p["p22"] - 1.0
        B = 0.25 * cov_sa_sb
    else:
        # Non-IBD2 relatives (k >= 3)
        p1 = p1_from_k(k)
        rho = rho_for_k(k, c_ab)
        p00, p10, p11 = _joint_ibd_probs_no_ibd2(p1, rho)
        A = p00 + p10 * (kappa_a + kappa_b + 6.0) / 4.0 + 0.75 * p11
        C = 0.25 * p11
        B = np.broadcast_to(0.25 * p1 * (1.0 - p1) * rho, r.shape)

    r2_part = A * r2
    cross_part = C * r * lambda_a * lambda_b
    shared_ibd_part = np.broadcast_to(B, r.shape)

    return {
        "r2_part": r2_part,
        "cross_part": cross_part,
        "shared_ibd_part": shared_ibd_part,
        "conditional_part": r2_part + cross_part,
        "total": r2_part + cross_part + shared_ibd_part,
        "A": A,
        "C": np.broadcast_to(C, r.shape),
        "B": shared_ibd_part,
    }
