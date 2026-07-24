"""Command-line interface for upsilon_k estimation."""

import argparse
import sys
from pathlib import Path

from .estimators import estimate_upsilon_k


def main():
    """Parse arguments and run upsilon_k estimation."""
    parser = argparse.ArgumentParser(
        description="Estimate upsilon_k (marker-causal relatedness ratio) from a founder panel in zarr format."
    )

    parser.add_argument(
        "zarr_path",
        type=str,
        help="Path to zarr directory with founder panel (genotypes, positions, allele_freq, genetic_position).",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output CSV file path.",
    )

    parser.add_argument(
        "--marker-threshold",
        type=float,
        default=0.01,
        help="Allele frequency threshold for marker sites (default 0.01).",
    )

    parser.add_argument(
        "--causal-threshold",
        type=float,
        default=0.05,
        help="Allele frequency threshold for causal sites (default 0.05).",
    )

    parser.add_argument(
        "-n",
        "--n-draws",
        type=int,
        default=150_000,
        help="Number of locus pairs to sample (default 150,000).",
    )

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=1,
        help="Random seed for reproducibility (default 1).",
    )

    parser.add_argument(
        "-k",
        "--k-values",
        type=int,
        nargs="+",
        default=None,
        help="Kinship classes to compute (default 2 3 4 5 6 7 8 9 10).",
    )

    parser.add_argument(
        "--bin-by-distance",
        action="store_true",
        help="Bin results by genetic distance.",
    )

    parser.add_argument(
        "--n-distance-bins",
        type=int,
        default=10,
        help="Number of distance bins (only used with --bin-by-distance, default 10).",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print progress information.",
    )

    args = parser.parse_args()

    # Validate arguments
    zarr_path = Path(args.zarr_path)
    if not zarr_path.exists():
        print(f"Error: zarr directory not found: {zarr_path}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.verbose:
        print(f"Loading panel from {zarr_path}...")

    try:
        results = estimate_upsilon_k(
            str(zarr_path),
            marker_threshold=args.marker_threshold,
            causal_threshold=args.causal_threshold,
            n_draws=args.n_draws,
            seed=args.seed,
            k_values=args.k_values,
            bin_by_distance=args.bin_by_distance,
            n_distance_bins=args.n_distance_bins,
        )
    except Exception as e:
        print(f"Error during estimation: {e}", file=sys.stderr)
        sys.exit(1)

    results.to_csv(output_path, index=False)

    if args.verbose:
        print(f"Results written to {output_path}")
        print(f"Computed {len(results)} rows")


if __name__ == "__main__":
    main()
