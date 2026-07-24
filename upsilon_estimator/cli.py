"""Command-line interface for upsilon_k estimation."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .estimators import estimate_upsilon_k


def _load_indices(path):
    """Load site indices from a file (CSV, NPZ, or .npy)."""
    if path is None:
        return None
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path)
    elif path.suffix == ".npz":
        data = np.load(path)
        if len(data.files) == 1:
            return data[data.files[0]]
        else:
            raise ValueError("NPZ file must have exactly one array")
    elif path.suffix in (".csv", ".txt"):
        df = pd.read_csv(path)
        if len(df.columns) == 1:
            return df.iloc[:, 0].values
        else:
            raise ValueError("CSV must have exactly one column")
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


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
        "--marker-indices",
        type=str,
        default=None,
        help="Path to file with marker site indices (.npy, .npz, or single-column CSV).",
    )

    parser.add_argument(
        "--causal-indices",
        type=str,
        default=None,
        help="Path to file with causal site indices (.npy, .npz, or single-column CSV).",
    )

    parser.add_argument(
        "--marker-threshold",
        type=float,
        default=None,
        help="Allele frequency threshold for markers (default 0.01, ignored if --marker-indices provided).",
    )

    parser.add_argument(
        "--causal-threshold",
        type=float,
        default=None,
        help="Allele frequency threshold for causal (default 0.05, ignored if --causal-indices provided).",
    )

    parser.add_argument(
        "--chromosome",
        type=str,
        default=None,
        help="Chromosome to load (if available in data).",
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
        if args.chromosome:
            print(f"  Chromosome: {args.chromosome}")

    # Load indices if provided
    marker_idx = _load_indices(args.marker_indices) if args.marker_indices else None
    causal_idx = _load_indices(args.causal_indices) if args.causal_indices else None

    if args.verbose and (marker_idx is not None or causal_idx is not None):
        if marker_idx is not None:
            print(f"  Markers: {len(marker_idx)} sites from {args.marker_indices}")
        if causal_idx is not None:
            print(f"  Causal: {len(causal_idx)} sites from {args.causal_indices}")

    try:
        results = estimate_upsilon_k(
            str(zarr_path),
            marker_indices=marker_idx,
            causal_indices=causal_idx,
            marker_threshold=args.marker_threshold,
            causal_threshold=args.causal_threshold,
            n_draws=args.n_draws,
            seed=args.seed,
            k_values=args.k_values,
            chromosome=args.chromosome,
            bin_by_distance=args.bin_by_distance,
            n_distance_bins=args.n_distance_bins,
        )
    except Exception as e:
        print(f"Error during estimation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    results.to_csv(output_path, index=False)

    if args.verbose:
        print(f"Results written to {output_path}")
        print(f"Computed {len(results)} rows")


if __name__ == "__main__":
    main()
