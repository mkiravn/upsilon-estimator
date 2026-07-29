"""Core upsilon_k estimation functions."""

from pathlib import Path

import numpy as np
import pandas as pd

from . import _theory as theory
from . import _panel as zb
from . import sampling as _samp
from .utils import paired_draws, estimate_components


def load_panel(path, chromosome=None):
    """Load a founder panel from zarr (xftsim or standard) or npz format.

    Args:
        path: Path to zarr directory or .npz file.
        chromosome: If specified, load only this chromosome (requires zarr with chrom info).

    Returns:
        dict with keys:
            - panel: LazyPanel object
            - pos: physical positions
            - af: allele frequencies
            - gpos: genetic positions
            - n_hap: number of haplotypes
            - n_sites: number of sites
            - chrom: chromosome array (if available)
    """
    path = Path(path)

    if path.suffix == ".npz":
        # Load from npz
        d = np.load(path)
        hap = d.get("haplotypes", d.get("genotypes"))
        pos = d["positions"].astype(float)
        af = d["allele_freq"]

        # Compute genetic position if not present
        if "genetic_position" in d:
            gpos = d["genetic_position"].astype(float)
        elif "recombination_rate" in d:
            rate = float(d["recombination_rate"])
            gpos = pos * rate * 100.0
        else:
            gpos = pos

        chrom = d.get("chrom", None)
        if chromosome is not None and chrom is not None:
            mask = chrom == chromosome
            hap = hap[mask]
            pos = pos[mask]
            af = af[mask]
            gpos = gpos[mask]
            chrom = chrom[mask]

        panel = zb.LazyPanel(
            genotypes=hap, positions=pos, allele_freq=af, genetic_position=gpos
        )
        return {
            "panel": panel,
            "pos": pos,
            "af": af,
            "gpos": gpos,
            "chrom": chrom,
            "n_hap": hap.shape[1],
            "n_sites": len(pos),
        }

    # Try xftsim format first (has HaplotypeArray, chrom metadata)
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
        from ld_sim.xft_zarr_reader import open_xft_founders

        store = open_xft_founders(str(path))
        chrom_to_load = chromosome if chromosome is not None else store.chromosomes()[0]
        panel = store.panel_for_chromosome(chrom_to_load)

        return {
            "panel": panel,
            "pos": panel.positions,
            "af": panel.allele_freq,
            "gpos": panel.genetic_position,
            "chrom": np.full(len(panel.positions), chrom_to_load),
            "n_hap": panel.n_haplotypes,
            "n_sites": panel.n_sites,
        }
    except (ImportError, FileNotFoundError, AssertionError):
        pass

    # Fall back to standard zarr
    try:
        import zarr
    except ImportError:
        raise ImportError("zarr is required for loading zarr format. Install with: pip install zarr")

    root = zarr.open(str(path), mode="r")
    hap = np.array(root.get("haplotypes", root.get("genotypes")))
    pos = np.array(root["positions"]).astype(float)
    af = np.array(root["allele_freq"])
    gpos = np.array(root.get("genetic_position", pos))
    chrom = np.array(root.get("chrom", None)) if "chrom" in root else None

    if chromosome is not None and chrom is not None:
        mask = chrom == chromosome
        hap = hap[mask]
        pos = pos[mask]
        af = af[mask]
        gpos = gpos[mask]
        chrom = chrom[mask]

    panel = zb.LazyPanel(
        genotypes=hap, positions=pos, allele_freq=af, genetic_position=gpos
    )

    return {
        "panel": panel,
        "pos": pos,
        "af": af,
        "gpos": gpos,
        "chrom": chrom,
        "n_hap": hap.shape[1],
        "n_sites": len(pos),
    }


def estimate_upsilon_k(
    zarr_path,
    marker_indices=None,
    causal_indices=None,
    marker_threshold=None,
    causal_threshold=None,
    n_draws=150_000,
    seed=1,
    k_values=None,
    chromosome=None,
):
    """Estimate upsilon_k: (r²_MQ + Φₖ) / (r²_MM + Φₖ).

    Primary estimator: debiased r² and shared IBD component only.
    Drops the cross term (B_k r·λ_a·λ_b) because:
      - It contributes <0.5% to signal but dominates variance
      - It has kurtosis in the thousands for rare variants
      - It is sign-cancelling (coupling/repulsion pairs mostly cancel)
      - Full model inflates estimates by 13-39% with no signal gain

    Args:
        zarr_path: Path to zarr or npz panel.
        marker_indices: Explicit marker site indices (preferred).
        causal_indices: Explicit causal site indices (preferred).
        marker_threshold: AF threshold for markers (if indices not provided).
        causal_threshold: AF threshold for causal (if indices not provided).
        n_draws: Number of locus pairs to sample (default 150,000).
        seed: Random seed (default 1).
        k_values: List of kinship classes (default [2..9]).
        chromosome: Specific chromosome to load (for xftsim zarr).

    Returns:
        DataFrame with columns: k, upsilon_k, r2_MM, r2_MQ, shared_ibd_part
    """
    if k_values is None:
        k_values = list(range(2, 10))  # k=2 to k=9

    panel_data = load_panel(zarr_path, chromosome=chromosome)
    panel = panel_data["panel"]
    af = panel_data["af"]
    gpos = panel_data["gpos"]
    n_hap = panel_data["n_hap"]

    # Identify marker and causal sites
    if marker_indices is not None and causal_indices is not None:
        marker_idx = np.asarray(marker_indices, dtype=int)
        causal_idx = np.asarray(causal_indices, dtype=int)
    else:
        if marker_threshold is None:
            marker_threshold = 0.01
        if causal_threshold is None:
            causal_threshold = 0.05

        marker_idx = np.nonzero(af > marker_threshold)[0]
        causal_idx = np.nonzero((af > 0) & (af < causal_threshold))[0]

    if len(marker_idx) == 0 or len(causal_idx) == 0:
        raise ValueError(
            f"No marker sites or causal sites found. "
            f"(marker: {len(marker_idx)}, causal: {len(causal_idx)})"
        )

    # Draw paired loci
    rng = np.random.default_rng(seed)
    num, den = paired_draws(
        marker_idx, causal_idx, n_draws, rng=rng, panel=panel, gpos=gpos
    )

    # Compute upsilon_k for each k (drops cross term)
    rows = []
    r2_MM = np.nanmean(theory.r2_debiased(den["r"] ** 2, n_hap))
    r2_MQ = np.nanmean(theory.r2_debiased(num["r"] ** 2, n_hap))

    for k in k_values:
        comp_num = estimate_components(k, num, n_hap)
        shared_ibd = np.nanmean(comp_num["shared_ibd_part"])

        # Primary: (r2_MQ + Phi_k) / (r2_MM + Phi_k), no cross term
        upsilon = (r2_MQ + shared_ibd) / (r2_MM + shared_ibd)

        rows.append(
            {
                "k": k,
                "upsilon_k": upsilon,
                "r2_MM": r2_MM,
                "r2_MQ": r2_MQ,
                "shared_ibd_part": shared_ibd,
            }
        )

    return pd.DataFrame(rows)


def estimate_upsilon_k_scale_aware(
    zarr_path,
    marker_indices=None,
    causal_indices=None,
    marker_threshold=None,
    causal_threshold=None,
    n_draws=150_000,
    seed=1,
    k_values=None,
    chromosome=None,
    d_min=1e-3,
    d_max=None,
    n_strata=24,
    n_blocks=200,
    include_A_k=False,
    alpha=None,
):
    """Scale-aware, density-weighted upsilon_k with block-jackknife standard errors.

    Improves on estimate_upsilon_k() in two ways (see upsilon_estimator.sampling):
      * Partners are drawn conditional on the anchor, stratified over log-spaced
        genetic-distance bins, so every scale is covered; importance weights make
        the estimate density-invariant (thinning SNPs leaves upsilon_k unchanged
        within the SE) and unbiased for the pair average within the linked window.
      * A delete-a-block jackknife over contiguous anchor-position blocks gives a
        ratio-correct, LD-aware standard error.

    Note on scale: upsilon_k depends on d_max (the maximum genetic distance over
    which pairs are counted). The genome-relevant value uses d_max = the
    chromosome's full genetic length; pass d_max explicitly to fix the scale, or
    leave None to use the panel's full map span.

    Args:
        d_min, d_max: genetic-distance stratum range in cM (d_max None => full span).
        n_strata: number of log-spaced distance strata.
        n_blocks: number of jackknife blocks.
        include_A_k: if True use A_k * r^2 (keeps the coefficient); default robust r^2.
        alpha: if not None, also compute the Speed alpha-model weighted estimate,
            gamma_l = [2f(1-f)]^(1+alpha) on the causal locus, normalised over the
            causal pool (alpha = -1 is GCTA/uniform and reproduces the unweighted
            estimate). Adds columns upsilon_k_weighted, upsilon_k_weighted_se, alpha.
        (other args as in estimate_upsilon_k.)

    Returns:
        DataFrame with columns: k, upsilon_k, upsilon_k_se, n_marker, n_causal
        (plus weighted columns when alpha is given).
    """
    if k_values is None:
        k_values = list(range(2, 10))

    panel_data = load_panel(zarr_path, chromosome=chromosome)
    panel = panel_data["panel"]
    af = panel_data["af"]
    gpos = panel_data["gpos"]
    n_hap = panel_data["n_hap"]

    if marker_indices is not None and causal_indices is not None:
        marker_idx = np.asarray(marker_indices, dtype=int)
        causal_idx = np.asarray(causal_indices, dtype=int)
    else:
        if marker_threshold is None:
            marker_threshold = 0.01
        if causal_threshold is None:
            causal_threshold = 0.05
        marker_idx = np.nonzero(af > marker_threshold)[0]
        causal_idx = np.nonzero((af > 0) & (af < causal_threshold))[0]

    if len(marker_idx) == 0 or len(causal_idx) == 0:
        raise ValueError(
            f"No marker/causal sites (marker: {len(marker_idx)}, causal: {len(causal_idx)})"
        )

    rng = np.random.default_rng(seed)
    num, den = _samp.stratified_draws(
        marker_idx, causal_idx, n_draws, rng, panel, gpos,
        d_min=d_min, d_max=d_max, n_strata=n_strata,
    )

    gamma_num = None
    if alpha is not None:
        # weight each numerator pair by its causal locus (the partner, num["f_j"]),
        # normalised to unit mean over the causal pool.
        gamma_num = _samp.effect_size_weights(num["f_j"], af[causal_idx], alpha)

    rows = []
    for k in k_values:
        ups, se = _samp.block_jackknife_se(
            k, num, den, n_hap, n_blocks=n_blocks, include_A_k=include_A_k
        )
        row = {
            "k": k,
            "upsilon_k": ups,
            "upsilon_k_se": se,
            "n_marker": len(marker_idx),
            "n_causal": len(causal_idx),
        }
        if alpha is not None:
            uw, sew = _samp.block_jackknife_se(
                k, num, den, n_hap, n_blocks=n_blocks, include_A_k=include_A_k,
                gamma_num=gamma_num,
            )
            row["upsilon_k_weighted"] = uw
            row["upsilon_k_weighted_se"] = sew
            row["alpha"] = alpha
        rows.append(row)
    return pd.DataFrame(rows)


def estimate_upsilon_k_full(
    zarr_path,
    marker_indices=None,
    causal_indices=None,
    marker_threshold=None,
    causal_threshold=None,
    n_draws=150_000,
    seed=1,
    k_values=None,
    chromosome=None,
):
    """DEPRECATED: Estimate upsilon_k including the cross term.

    WARNING: This estimator includes B_k r·λ_a·λ_b (cross term), which:
      - Dominates variance with kurtosis in the thousands
      - Contributes <0.5% to the signal
      - Inflates estimates by 13-39% compared to the crude model
      - Is sign-cancelling (provides no additional information)

    Use estimate_upsilon_k() instead (the crude model without cross term).

    Args:
        See estimate_upsilon_k() for parameter documentation.

    Returns:
        DataFrame with columns:
            - k: kinship class
            - upsilon_k: estimate without cross term
            - upsilon_k_full: estimate with cross term (UNRELIABLE)
            - r2_MM: mean debiased r2 (marker-marker)
            - r2_MQ: mean debiased r2 (marker-causal)
            - shared_ibd_part: mean shared IBD component (Phi_k)
    """
    import warnings
    warnings.warn(
        "estimate_upsilon_k_full is deprecated. Use estimate_upsilon_k() instead. "
        "The full model (with cross term) has 13-39% higher variance with <0.5% signal gain.",
        DeprecationWarning,
        stacklevel=2
    )

    if k_values is None:
        k_values = list(range(2, 11))

    panel_data = load_panel(zarr_path, chromosome=chromosome)
    panel = panel_data["panel"]
    af = panel_data["af"]
    gpos = panel_data["gpos"]
    n_hap = panel_data["n_hap"]

    # Use provided indices, or fall back to thresholds
    if marker_indices is not None and causal_indices is not None:
        marker_idx = np.asarray(marker_indices, dtype=int)
        causal_idx = np.asarray(causal_indices, dtype=int)
    else:
        if marker_threshold is None:
            marker_threshold = 0.01
        if causal_threshold is None:
            causal_threshold = 0.05

        marker_idx = np.nonzero(af > marker_threshold)[0]
        causal_idx = np.nonzero((af > 0) & (af < causal_threshold))[0]

    if len(marker_idx) == 0 or len(causal_idx) == 0:
        raise ValueError(
            f"No marker sites or causal sites found. "
            f"(marker: {len(marker_idx)}, causal: {len(causal_idx)})"
        )

    # Draw paired loci
    rng = np.random.default_rng(seed)
    num, den = paired_draws(
        marker_idx, causal_idx, n_draws, rng=rng, panel=panel, gpos=gpos
    )

    # Compute results for each k
    rows = []
    for k in k_values:
        comp_num = estimate_components(k, num, n_hap)
        comp_den = estimate_components(k, den, n_hap)

        r2_MM = np.nanmean(theory.r2_debiased(den["r"] ** 2, n_hap))
        r2_MQ = np.nanmean(theory.r2_debiased(num["r"] ** 2, n_hap))

        shared_ibd = np.nanmean(comp_num["shared_ibd_part"])

        # Primary: (r2_MQ + Phi_k) / (r2_MM + Phi_k)
        upsilon = (r2_MQ + shared_ibd) / (r2_MM + shared_ibd)

        # Full with cross term: mean(h_k_num) / mean(h_k_den)
        upsilon_full = np.nanmean(comp_num["total"]) / np.nanmean(comp_den["total"])

        rows.append(
            {
                "k": k,
                "upsilon_k": upsilon,
                "upsilon_k_full": upsilon_full,
                "r2_MM": r2_MM,
                "r2_MQ": r2_MQ,
                "shared_ibd_part": shared_ibd,
            }
        )

    return pd.DataFrame(rows)


def estimate_upsilon_k_crude(
    zarr_path,
    marker_indices=None,
    causal_indices=None,
    marker_threshold=None,
    causal_threshold=None,
    n_draws=150_000,
    seed=1,
    k_values=None,
    chromosome=None,
):
    """DEPRECATED: Alias for estimate_upsilon_k().

    This function is kept for backward compatibility. Use estimate_upsilon_k() instead.
    Returns the same results with column "upsilon_k_crude" for compatibility.
    """
    import warnings
    warnings.warn(
        "estimate_upsilon_k_crude is deprecated. Use estimate_upsilon_k() directly; "
        "it now computes the same robust estimate.",
        DeprecationWarning,
        stacklevel=2
    )
    result = estimate_upsilon_k(
        zarr_path=zarr_path,
        marker_indices=marker_indices,
        causal_indices=causal_indices,
        marker_threshold=marker_threshold,
        causal_threshold=causal_threshold,
        n_draws=n_draws,
        seed=seed,
        k_values=k_values,
        chromosome=chromosome,
    )
    # Rename for backward compatibility
    result = result.rename(columns={"upsilon_k": "upsilon_k_crude"})
    return result


def _add_distance_bins(num, den, marker_idx, causal_idx, gpos, results, k_values, n_hap, n_bins):
    """Add distance-binned columns to results."""
    gd = np.abs(gpos[num["anchor"]] - gpos[num["partner"]])
    bins = np.linspace(0, gd.max(), n_bins + 1)
    bin_labels = [f"{bins[i]:.3f}-{bins[i+1]:.3f}" for i in range(n_bins)]

    # For each distance bin
    binned_rows = []
    for k in k_values:
        bin_results = {"k": k}
        for i, (bin_start, bin_end) in enumerate(zip(bins[:-1], bins[1:])):
            mask = (gd >= bin_start) & (gd < bin_end)
            if mask.sum() < 10:  # Skip bins with too few pairs
                continue

            num_bin = {key: val[mask] for key, val in num.items() if key not in ("anchor", "partner")}
            den_bin = {key: val[mask] for key, val in den.items() if key not in ("anchor", "partner")}

            comp_num = estimate_components(k, num_bin, n_hap)
            comp_den = estimate_components(k, den_bin, n_hap)

            r2_MM = np.nanmean(theory.r2_debiased(den_bin["r"] ** 2, n_hap))
            r2_MQ = np.nanmean(theory.r2_debiased(num_bin["r"] ** 2, n_hap))
            shared_ibd = np.nanmean(comp_num["shared_ibd_part"])

            upsilon = (r2_MQ + shared_ibd) / (r2_MM + shared_ibd)

            bin_key = f"bin_{i}"
            bin_results[f"{bin_key}_upsilon_k"] = upsilon
            bin_results[f"{bin_key}_r2_MM"] = r2_MM
            bin_results[f"{bin_key}_r2_MQ"] = r2_MQ
            bin_results[f"{bin_key}_distance"] = f"{bin_start:.3f}-{bin_end:.3f}"

        binned_rows.append(bin_results)

    # Merge with main results
    binned_df = pd.DataFrame(binned_rows).set_index("k")
    results = results.set_index("k").join(binned_df, how="left").reset_index()

    return results



def estimate_upsilon_k_weighted(
    zarr_path,
    marker_indices=None,
    causal_indices=None,
    marker_threshold=None,
    causal_threshold=None,
    n_draws=150_000,
    seed=1,
    k_values=None,
    chromosome=None,
    alpha=-1.0,
):
    """Estimate upsilon_k with frequency-dependent effect-size weighting (Speed alpha-model).

    The standard unweighted model treats all causal loci symmetrically. Under a random-effects
    model where SNP effect-size variance depends on allele frequency, the covariance is:

        Cov(g_i, g_j) = σ_β² Σ_ℓ γ_ℓ X_i_ℓ X_j_ℓ

    where E[β_ℓ²] = σ_β² γ_ℓ and γ_ℓ = [2f_ℓ(1-f_ℓ)]^(1+alpha).

    For marker-causal pairs {ℓ,m}, the per-pair weighting factor is (γ_ℓ + γ_m) / 2.

    The weighted upsilon_k is:
        upsilon_k^(gamma) = mean(h_k * (γ_marker + γ_causal) / 2) / mean(h_k)

    Args:
        zarr_path: Path to zarr or npz panel.
        marker_indices, causal_indices: Explicit site indices (preferred).
        marker_threshold, causal_threshold: AF thresholds for automatic site selection.
        n_draws: Number of locus pairs to sample (default 150,000).
        seed: Random seed (default 1).
        k_values: List of kinship classes (default [2..9]).
        chromosome: Specific chromosome to load (for xftsim zarr).
        alpha: Speed alpha-model exponent (default -1.0 for GCTA/uniform weighting).
               - alpha=-1: uniform weighting (matches GCTA)
               - alpha<-1: upweight rare variants
               - alpha>-1: upweight common variants

    Returns:
        DataFrame with columns: k, upsilon_k, upsilon_k_weighted, r2_MM, r2_MQ,
                               shared_ibd_part, gamma_marker, gamma_causal
    """
    if k_values is None:
        k_values = list(range(2, 10))

    panel_data = load_panel(zarr_path, chromosome=chromosome)
    panel = panel_data["panel"]
    af = panel_data["af"]
    gpos = panel_data["gpos"]
    n_hap = panel_data["n_hap"]

    # Identify marker and causal sites
    if marker_indices is not None and causal_indices is not None:
        marker_idx = np.asarray(marker_indices, dtype=int)
        causal_idx = np.asarray(causal_indices, dtype=int)
    else:
        if marker_threshold is None:
            marker_threshold = 0.01
        if causal_threshold is None:
            causal_threshold = 0.05

        marker_idx = np.nonzero(af > marker_threshold)[0]
        causal_idx = np.nonzero((af > 0) & (af < causal_threshold))[0]

    if len(marker_idx) == 0 or len(causal_idx) == 0:
        raise ValueError(
            f"No marker sites or causal sites found. "
            f"(marker: {len(marker_idx)}, causal: {len(causal_idx)})"
        )

    # Draw paired loci
    rng = np.random.default_rng(seed)
    num, den = paired_draws(
        marker_idx, causal_idx, n_draws, rng=rng, panel=panel, gpos=gpos
    )

    # Effect-size weight of each numerator pair's CAUSAL locus (the partner, f_j),
    # normalised to unit mean over the causal pool (not all sites). alpha = -1 gives
    # gamma == 1, so the weighted estimate reduces exactly to the unweighted.
    from . import sampling as _samp
    gamma_causal = _samp.effect_size_weights(num["f_j"], af[causal_idx], alpha)

    # Compute upsilon_k for each k
    rows = []
    r2_MM = np.nanmean(theory.r2_debiased(den["r"] ** 2, n_hap))
    r2_MQ = np.nanmean(theory.r2_debiased(num["r"] ** 2, n_hap))

    for k in k_values:
        # Get components for unweighted numerator
        comp_num = estimate_components(k, num, n_hap)
        shared_ibd = np.nanmean(comp_num["shared_ibd_part"])

        # Unweighted: (r2_MQ + Phi_k) / (r2_MM + Phi_k)
        upsilon = (r2_MQ + shared_ibd) / (r2_MM + shared_ibd)

        # Weighted numerator: scale each pair's FULL robust contribution
        # (debiased r^2 + Phi_k) by the causal locus's effect-size weight. The
        # weighted causal GRM multiplies the whole per-locus term by gamma, so
        # Phi_k is scaled too (it cancels only because gamma has unit mean).
        r2_num_pp = theory.r2_debiased(num["r"] ** 2, n_hap)
        contrib_pp = r2_num_pp + comp_num["shared_ibd_part"]
        h_k_weighted_numerator = np.nanmean(gamma_causal * contrib_pp)

        # Upsilon_k^(gamma) = mean(gamma * h_k) / (r2_MM + Phi_k)
        upsilon_weighted = h_k_weighted_numerator / (r2_MM + shared_ibd)

        rows.append(
            {
                "k": k,
                "upsilon_k": upsilon,
                "upsilon_k_weighted": upsilon_weighted,
                "r2_MM": r2_MM,
                "r2_MQ": r2_MQ,
                "shared_ibd_part": shared_ibd,
                "gamma_causal_mean": np.nanmean(gamma_causal),
                "alpha": alpha,
            }
        )

    return pd.DataFrame(rows)