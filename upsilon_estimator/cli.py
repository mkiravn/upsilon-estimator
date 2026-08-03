"""Command-line interface for upsilon_k estimation.

By default this runs the scale-aware, density-weighted estimator with
block-jackknife standard errors (estimate_upsilon_k_scale_aware), which is the
one to use for genome-scale panels: it covers all genetic-distance scales, is
invariant to SNP density, and reports a standard error per relationship class.
Use --fast for a quick point estimate without the jackknife.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .estimators import estimate_upsilon_k, estimate_upsilon_k_scale_aware


def _load_indices(path):
    """Load site indices from a file (.npy, single-array .npz, or single-column CSV/TXT)."""
    if path is None:
        return None
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".npz":
        data = np.load(path)
        if len(data.files) != 1:
            raise ValueError("NPZ index file must contain exactly one array")
        return data[data.files[0]]
    if path.suffix in (".csv", ".txt"):
        df = pd.read_csv(path)
        if df.shape[1] != 1:
            raise ValueError("CSV/TXT index file must have exactly one column")
        return df.iloc[:, 0].values
    raise ValueError(f"Unsupported index file format: {path.suffix}")


def main():
    p = argparse.ArgumentParser(
        description="Estimate upsilon_k (marker/causal attenuation of h^2) from a "
                    "phased founder panel in zarr or npz format."
    )
    p.add_argument("panel_path", type=str,
                   help="Path to the panel: a zarr directory or an .npz with "
                        "genotypes/haplotypes, positions, allele_freq, genetic_position.")
    p.add_argument("-o", "--output", type=str, required=True, help="Output CSV path.")

    # marker / causal locus sets
    p.add_argument("--marker-indices", type=str, default=None,
                   help="File of marker site indices (.npy, single-array .npz, or 1-col CSV).")
    p.add_argument("--causal-indices", type=str, default=None,
                   help="File of causal site indices (.npy, single-array .npz, or 1-col CSV).")
    p.add_argument("--marker-threshold", type=float, default=None,
                   help="MAF threshold for markers if indices not given (default 0.01).")
    p.add_argument("--causal-threshold", type=float, default=None,
                   help="Upper MAF threshold for causal if indices not given (default 0.05).")
    p.add_argument("--chromosome", type=str, default=None,
                   help="Chromosome to load (for multi-chromosome zarr).")

    # relationship classes and sampling
    p.add_argument("-k", "--k-values", type=int, nargs="+", default=None,
                   help="Relationship classes to compute (default 2..9).")
    p.add_argument("-n", "--n-draws", type=int, default=200_000,
                   help="Number of locus pairs to sample (default 200,000).")
    p.add_argument("-s", "--seed", type=int, default=1, help="Random seed (default 1).")

    # scale-aware estimator options
    p.add_argument("--n-strata", type=int, default=24,
                   help="Log-spaced genetic-distance strata for sampling (default 24).")
    p.add_argument("--n-blocks", type=int, default=200,
                   help="Jackknife blocks for the standard error (default 200).")
    p.add_argument("--d-max", type=float, default=None,
                   help="Max pair genetic distance in cM (default: full map span of the panel).")
    p.add_argument("--alpha", type=float, default=None,
                   help="Speed alpha-model exponent; if set, also report the weighted "
                        "estimate (alpha=-1 is GCTA/uniform).")
    p.add_argument("--include-a-k", action="store_true",
                   help="Use A_k * r^2 instead of the robust r^2 (keeps the coefficient).")

    p.add_argument("--fast", action="store_true",
                   help="Skip the jackknife: quick point estimate via estimate_upsilon_k.")
    p.add_argument("-v", "--verbose", action="store_true", help="Print progress.")

    args = p.parse_args()

    panel_path = Path(args.panel_path)
    if not panel_path.exists():
        print(f"Error: panel not found: {panel_path}", file=sys.stderr)
        sys.exit(1)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    marker_idx = _load_indices(args.marker_indices)
    causal_idx = _load_indices(args.causal_indices)

    if args.verbose:
        print(f"Loading {panel_path}"
              + (f" (chromosome {args.chromosome})" if args.chromosome else ""))
        if marker_idx is not None:
            print(f"  markers: {len(marker_idx)} sites")
        if causal_idx is not None:
            print(f"  causal:  {len(causal_idx)} sites")
        print("  estimator: " + ("fast point estimate" if args.fast
                                  else f"scale-aware + jackknife ({args.n_blocks} blocks)"))

    try:
        if args.fast:
            results = estimate_upsilon_k(
                str(panel_path),
                marker_indices=marker_idx, causal_indices=causal_idx,
                marker_threshold=args.marker_threshold, causal_threshold=args.causal_threshold,
                n_draws=args.n_draws, seed=args.seed,
                k_values=args.k_values, chromosome=args.chromosome,
            )
        else:
            results = estimate_upsilon_k_scale_aware(
                str(panel_path),
                marker_indices=marker_idx, causal_indices=causal_idx,
                marker_threshold=args.marker_threshold, causal_threshold=args.causal_threshold,
                n_draws=args.n_draws, seed=args.seed,
                k_values=args.k_values, chromosome=args.chromosome,
                n_strata=args.n_strata, n_blocks=args.n_blocks, d_max=args.d_max,
                include_A_k=args.include_a_k, alpha=args.alpha,
            )
    except Exception as e:  # noqa: BLE001 - surface the error to the user
        print(f"Error during estimation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    results.to_csv(out, index=False)
    if args.verbose:
        print(f"Wrote {len(results)} rows to {out}")
        print(results.to_string(index=False))


if __name__ == "__main__":
    main()
