"""Core upsilon_k estimation functions."""

from pathlib import Path

import numpy as np
import pandas as pd

from . import _theory as theory
from . import _panel as zb
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
    """Estimate crude upsilon_k: (r²_MQ + Φₖ) / (r²_MM + Φₖ).

    Simpler, more interpretable variant without A_k (kurtosis) weighting.
    Uses pooled r² and shared IBD component only.

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
        DataFrame with columns: k, upsilon_k_crude, r2_MM, r2_MQ, shared_ibd_part
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

    # Compute crude upsilon_k for each k
    rows = []
    r2_MM = np.nanmean(theory.r2_debiased(den["r"] ** 2, n_hap))
    r2_MQ = np.nanmean(theory.r2_debiased(num["r"] ** 2, n_hap))

    for k in k_values:
        comp_num = estimate_components(k, num, n_hap)
        shared_ibd = np.nanmean(comp_num["shared_ibd_part"])

        # Crude: (r2_MQ + Phi_k) / (r2_MM + Phi_k)
        upsilon_crude = (r2_MQ + shared_ibd) / (r2_MM + shared_ibd)

        rows.append(
            {
                "k": k,
                "upsilon_k_crude": upsilon_crude,
                "r2_MM": r2_MM,
                "r2_MQ": r2_MQ,
                "shared_ibd_part": shared_ibd,
            }
        )

    return pd.DataFrame(rows)


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
    bin_by_distance=False,
    n_distance_bins=10,
):
    """Estimate upsilon_k (marker-causal relatedness ratio) from a founder panel.

    Args:
        zarr_path: Path to zarr directory or .npz file with founder panel.
        marker_indices: Array of site indices to use as markers (preferred over threshold).
        causal_indices: Array of site indices to use as causal sites (preferred over threshold).
        marker_threshold: Allele frequency threshold for markers (only used if marker_indices=None).
        causal_threshold: Allele frequency threshold for causal sites (only used if causal_indices=None).
        n_draws: Number of locus pairs to sample (default 150,000).
        seed: Random seed for reproducibility (default 1).
        k_values: List of kinship classes to compute (default [2,3,4,5,6,7,8,9,10]).
        chromosome: Chromosome to load (if available in data).
        bin_by_distance: If True, return results binned by genetic distance.
        n_distance_bins: Number of distance bins.

    Returns:
        DataFrame with columns:
            - k: kinship class
            - upsilon_k_crude: crude (debiased r2 + Phi_k) estimate
            - upsilon_k_full: full (exact h_k) estimate
            - r2_MM: mean debiased r2 (marker-marker)
            - r2_MQ: mean debiased r2 (marker-causal)
            - shared_ibd_part: mean shared IBD component (Phi_k)
    """
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

        # Crude: (r2_MQ + Phi_k) / (r2_MM + Phi_k)
        upsilon_crude = (r2_MQ + shared_ibd) / (r2_MM + shared_ibd)

        # Full: mean(h_k_num) / mean(h_k_den)
        upsilon_full = np.nanmean(comp_num["total"]) / np.nanmean(comp_den["total"])

        rows.append(
            {
                "k": k,
                "upsilon_k_crude": upsilon_crude,
                "upsilon_k_full": upsilon_full,
                "r2_MM": r2_MM,
                "r2_MQ": r2_MQ,
                "shared_ibd_part": shared_ibd,
            }
        )

    results = pd.DataFrame(rows)

    if bin_by_distance:
        results = _add_distance_bins(
            num, den, marker_idx, causal_idx, gpos, results, k_values, n_hap, n_distance_bins
        )

    return results


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

            upsilon_crude = (r2_MQ + shared_ibd) / (r2_MM + shared_ibd)
            upsilon_full = np.nanmean(comp_num["total"]) / np.nanmean(comp_den["total"])

            bin_key = f"bin_{i}"
            bin_results[f"{bin_key}_upsilon_crude"] = upsilon_crude
            bin_results[f"{bin_key}_upsilon_full"] = upsilon_full
            bin_results[f"{bin_key}_r2_MM"] = r2_MM
            bin_results[f"{bin_key}_r2_MQ"] = r2_MQ
            bin_results[f"{bin_key}_distance"] = f"{bin_start:.3f}-{bin_end:.3f}"

        binned_rows.append(bin_results)

    # Merge with main results
    binned_df = pd.DataFrame(binned_rows).set_index("k")
    results = results.set_index("k").join(binned_df, how="left").reset_index()

    return results
