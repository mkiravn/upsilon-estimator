"""Vendored panel and genotype utilities from ld_sim."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LazyPanel:
    """A genome-scale genotype panel where genotypes can be lazy-loaded.

    `genotypes` can be a zarr.Array, numpy.ndarray, or any array supporting
    fancy indexing with integer arrays (e.g., genotypes[indices, :]).
    """

    genotypes: object  # zarr.Array | np.ndarray | row-fancy-indexable 2D array
    positions: np.ndarray
    allele_freq: np.ndarray
    sequence_length: float | None = None
    genetic_position: np.ndarray | None = None

    @property
    def n_sites(self) -> int:
        return len(self.positions)

    @property
    def n_haplotypes(self) -> int:
        return self.genotypes.shape[1]

    def __post_init__(self):
        assert self.genotypes.shape[0] == len(self.positions) == len(self.allele_freq), (
            "genotypes/positions/allele_freq must agree on n_sites"
        )
        assert np.all(np.diff(self.positions) >= 0), "positions must be sorted ascending"
        if self.genetic_position is not None:
            assert len(self.genetic_position) == len(self.positions), (
                "genetic_position must have one entry per site"
            )
            assert np.all(np.diff(self.genetic_position) >= 0), (
                "genetic_position must be non-decreasing"
            )


def fetch_sites(panel: LazyPanel, indices: np.ndarray) -> np.ndarray:
    """Batched on-demand fetch of genotype rows (sites).

    Handles sorting for efficient chunked reads, then restores original order.
    """
    indices = np.asarray(indices)
    order = np.argsort(indices, kind="stable")
    sorted_idx = indices[order]
    rows = np.asarray(panel.genotypes[sorted_idx, :])
    out = np.empty_like(rows)
    out[order] = rows
    return out


def _pairwise_r_from_rows(rows_a: np.ndarray, rows_b: np.ndarray) -> np.ndarray:
    """Signed correlation r between each corresponding pair of rows.

    Computes row-pair-wise correlation (vs full matrix).
    """
    a = rows_a.astype(np.float64)
    b = rows_b.astype(np.float64)
    a_c = a - a.mean(axis=1, keepdims=True)
    b_c = b - b.mean(axis=1, keepdims=True)
    num = (a_c * b_c).sum(axis=1)
    den = np.sqrt((a_c**2).sum(axis=1) * (b_c**2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / den
    return r
