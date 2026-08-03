"""Minimal, dependency-light reader for xftsim's `HaplotypeArray` zarr
store (the format `hapgen2xft.py` in the `glitch-matrix` pipeline writes
founder haplotypes to), adapted into this package's `LazyPanel`s (see
`_panel.py`) so `estimators.load_panel` can run directly against xftsim
founder panels. Vendored from `ld_sim.xft_zarr_reader` so this package has
no dependency on the sibling `ld_sim` repo being importable.

Why this exists instead of just using the `zarr` package
------------------------------------------------------------
`zarr`'s `numcodecs` dependency has no prebuilt wheel in many deployment
environments and is slow to build from source there. The standalone
`blosc` package (distinct from `numcodecs`, just Python bindings for the
same C library) reliably has a prebuilt wheel, and a zarr v2 array is a
simple enough on-disk format (a `.zarray` JSON metadata file plus one
blosc-compressed chunk file per chunk, arranged in a predictable grid)
that a minimal reader needs only `blosc` and `numpy`, not `zarr`/`numcodecs`
at all. This only supports what's needed here: blosc-compressed arrays
with no additional filters, `zarr_format: 2`, C order -- see `ZarrV2BloscArray`.

The xftsim `HaplotypeArray` layout
-------------------------------------
`HaplotypeArray` has shape `(n_individuals, 2 * n_true_sites)`: each true
site occupies *two* adjacent columns (haplotype copy 0, haplotype copy 1 --
see `simulation_utils.make_diploid_shape` in the `glitch-matrix` repo),
so `af`/`pos_bp`/`pos_cM`/`chrom` (each shape `(2*n_true_sites,)`) have
every value duplicated across each such column pair. `XftFounderPanel`
undoes this: it exposes `n_true_sites`-length `positions`/`allele_freq`/
`genetic_position`, and reconstructs *haploid* genotype rows (shape
`(n_sites, 2*n_individuals)`, matching every other `LazyPanel` in this
project) by concatenating the two haplotype copies of each individual as
if they were `2*n_individuals` separate haploid samples -- validated
against the store's own `af` (see `tests/test_xft_zarr_reader.py` for the
synthetic-fixture unit tests, and `notebooks/11_glitch_matrix_upsilon_k.ipynb`
for the validation against the real store: reconstructed empirical allele
frequency matched the stored `af` column exactly on every site checked).

A note on this specific mount's flakiness
--------------------------------------------
Reads against the real store's chunk files intermittently raised
`OSError(EDEADLK, "Resource deadlock avoided")` in this project's sandbox
-- empirically tied to files not yet materialized from wherever this
particular external folder is actually hosted (it cleared reliably after
a host-side read of the same file "primed" it; a single such read anywhere
in a chunk's parent directory was enough to make every file in that
directory subsequently readable). `_read_bytes_with_retry` retries with
backoff as a best-effort safety net, but priming reads (outside this
module -- e.g. a Grep-tool scan over the target directory) may still be
needed the *first* time a given chunk directory is touched in a session.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import numpy as np

try:
    import blosc
except ImportError as e:  # pragma: no cover
    blosc = None
    _BLOSC_IMPORT_ERROR = e

from ._panel import LazyPanel


def _read_bytes_with_retry(path, n_tries=6, base_sleep=0.3):
    last_exc = None
    for attempt in range(n_tries):
        try:
            with open(path, "rb") as f:
                return f.read()
        except OSError as e:
            last_exc = e
            time.sleep(base_sleep * (attempt + 1))
    raise last_exc


class ZarrV2BloscArray:
    """Minimal read-only accessor for one zarr v2 array directory,
    blosc-compressed with no additional filters -- does not depend on
    `zarr`/`numcodecs` at all (see module docstring)."""

    def __init__(self, array_dir):
        if blosc is None:  # pragma: no cover
            raise ImportError(
                "the 'blosc' package is required by xft_zarr_reader but is not installed "
                "(`pip install blosc`; distinct from numcodecs, has a prebuilt wheel)"
            ) from _BLOSC_IMPORT_ERROR
        self.array_dir = array_dir
        raw = _read_bytes_with_retry(os.path.join(array_dir, ".zarray"))
        meta = json.loads(raw)
        assert meta["zarr_format"] == 2
        assert meta["compressor"]["id"] == "blosc", f"only blosc-compressed arrays supported, got {meta['compressor']}"
        assert not meta["filters"], f"filters not supported by this minimal reader, got {meta['filters']}"
        assert meta["order"] == "C"
        self.shape = tuple(meta["shape"])
        self.chunks = tuple(meta["chunks"])
        self.dtype = np.dtype(meta["dtype"])
        self.fill_value = meta["fill_value"]
        self.ndim = len(self.shape)
        assert self.ndim in (1, 2), "only 1D/2D arrays supported"

    def _n_chunks_along(self, axis):
        return -(-self.shape[axis] // self.chunks[axis])  # ceil div

    def _chunk_path(self, chunk_idx_tuple):
        return os.path.join(self.array_dir, ".".join(str(c) for c in chunk_idx_tuple))

    def _read_chunk(self, chunk_idx_tuple):
        """Decompress one chunk at its full nominal chunk shape -- zarr v2
        always stores chunks (including ragged edge chunks) at full
        nominal size, padded with `fill_value` before compression."""
        raw = _read_bytes_with_retry(self._chunk_path(chunk_idx_tuple))
        dec = blosc.decompress(raw)
        arr = np.frombuffer(dec, dtype=self.dtype)
        return arr.reshape(self.chunks)

    def read_full_1d(self):
        assert self.ndim == 1
        n_chunks = self._n_chunks_along(0)
        out = np.empty(self.shape[0], dtype=self.dtype)
        chunk_len = self.chunks[0]
        for c in range(n_chunks):
            chunk = self._read_chunk((c,))
            lo = c * chunk_len
            hi = min(lo + chunk_len, self.shape[0])
            out[lo:hi] = chunk[: hi - lo]
        return out

    def read_columns_2d(self, col_indices):
        """Fetch specific columns (axis=1) across ALL rows (axis=0) --
        the access pattern needed for `HaplotypeArray` (rows=individuals,
        all needed; columns=variant slots, sparse queries). Groups
        requested columns by which variant-chunk they fall in, so each
        touched (row-chunk, col-chunk) pair is decompressed exactly once
        no matter how many of its columns are requested."""
        assert self.ndim == 2
        col_indices = np.asarray(col_indices)
        n_row_chunks = self._n_chunks_along(0)
        row_chunk_len = self.chunks[0]
        col_chunk_len = self.chunks[1]

        col_chunk_idx = col_indices // col_chunk_len
        col_within_chunk = col_indices % col_chunk_len

        out = np.empty((self.shape[0], len(col_indices)), dtype=self.dtype)
        for vc in np.unique(col_chunk_idx):
            local_mask = col_chunk_idx == vc
            local_cols = col_within_chunk[local_mask]
            for rc in range(n_row_chunks):
                chunk = self._read_chunk((rc, int(vc)))
                lo = rc * row_chunk_len
                hi = min(lo + row_chunk_len, self.shape[0])
                out[lo:hi][:, local_mask] = chunk[: hi - lo][:, local_cols]
        return out

    def touched_col_chunks(self, col_indices):
        """Which column-chunk indices a set of (true-site-doubled) column
        indices falls into -- used to prime the right chunk files (e.g.
        via a host-side directory scan) before `read_columns_2d`."""
        col_indices = np.asarray(col_indices)
        return np.unique(col_indices // self.chunks[1]).tolist()


class _XftHaplotypeSiteAccessor:
    """Adapts a `ZarrV2BloscArray` over `HaplotypeArray` (doubled-hcopy,
    diploid-individual columns) into the `genotypes[row_indices, :]`
    protocol every `zarr_backend.py` function expects: `row_indices` are
    *local, chromosome-relative, true-site* indices in; the return value
    is a haploid genotype matrix, shape `(len(row_indices), 2*n_individuals)`
    -- haplotype copy 0 for all individuals, then haplotype copy 1,
    matching every other `LazyPanel.genotypes` in this project (0/1 per
    haploid sample, not 0/1/2 diploid dosage).
    """

    def __init__(self, hap_arr: ZarrV2BloscArray, global_true_site_idx: np.ndarray):
        self._hap_arr = hap_arr
        self._global_true_site_idx = np.asarray(global_true_site_idx)
        self.n_individuals = hap_arr.shape[0]
        self.shape = (len(self._global_true_site_idx), 2 * self.n_individuals)

    def touched_col_chunks(self, local_row_indices):
        global_idx = self._global_true_site_idx[np.asarray(local_row_indices)]
        raw_cols = np.empty(2 * len(global_idx), dtype=np.int64)
        raw_cols[0::2] = 2 * global_idx
        raw_cols[1::2] = 2 * global_idx + 1
        return self._hap_arr.touched_col_chunks(raw_cols)

    def __getitem__(self, key):
        row_indices, col_key = key
        assert col_key == slice(None), "only full-column (all-haplotype) fetches are supported"
        row_indices = np.asarray(row_indices)
        global_idx = self._global_true_site_idx[row_indices]

        raw_cols = np.empty(2 * len(global_idx), dtype=np.int64)
        raw_cols[0::2] = 2 * global_idx
        raw_cols[1::2] = 2 * global_idx + 1

        cols = self._hap_arr.read_columns_2d(raw_cols)  # (n_individuals, 2*k)
        hap0 = cols[:, 0::2].T  # (k, n_individuals)
        hap1 = cols[:, 1::2].T  # (k, n_individuals)
        return np.concatenate([hap0, hap1], axis=1)  # (k, 2*n_individuals)


@dataclass
class XftFounderStore:
    """Handle for one xftsim `HaplotypeArray` zarr store -- holds the
    small (fully in-memory) per-true-site metadata (positions, allele
    frequencies, genetic positions, chromosome labels) plus a lazy handle
    to the (far larger) `HaplotypeArray` itself. Build with `open_xft_founders`.
    """

    zarr_root: str
    hap_arr: ZarrV2BloscArray
    true_pos_bp: np.ndarray
    true_pos_cM: np.ndarray
    true_af: np.ndarray
    true_chrom: np.ndarray

    @property
    def n_true_sites(self) -> int:
        return len(self.true_pos_bp)

    @property
    def n_individuals(self) -> int:
        return self.hap_arr.shape[0]

    @property
    def n_haplotypes(self) -> int:
        return 2 * self.n_individuals

    def chromosomes(self) -> list:
        return sorted(set(self.true_chrom.tolist()))

    def panel_for_chromosome(self, chrom) -> LazyPanel:
        """Build a `zarr_backend.LazyPanel` for one chromosome's true
        sites -- `positions`/`allele_freq`/`genetic_position` fully in
        memory (small even genome-wide), `genotypes` a lazy accessor onto
        the real `HaplotypeArray` (only touched by `zarr_backend.fetch_sites`,
        after sampling decisions are already made)."""
        mask = self.true_chrom == chrom
        global_true_site_idx = np.nonzero(mask)[0]
        assert len(global_true_site_idx) > 0, f"no sites found for chromosome {chrom!r}"

        accessor = _XftHaplotypeSiteAccessor(self.hap_arr, global_true_site_idx)
        return LazyPanel(
            genotypes=accessor,
            positions=self.true_pos_bp[mask].astype(float),
            allele_freq=self.true_af[mask],
            genetic_position=self.true_pos_cM[mask],
        )

    def touched_col_chunks_for_chromosome(self, chrom) -> list:
        """Which `HaplotypeArray` column-chunk indices a chromosome's
        sites fall into -- use this to prime exactly the needed chunk
        files (e.g. via a host-side directory scan with a matching glob)
        before running any sampling against that chromosome's panel."""
        mask = self.true_chrom == chrom
        global_true_site_idx = np.nonzero(mask)[0]
        raw_cols = np.empty(2 * len(global_true_site_idx), dtype=np.int64)
        raw_cols[0::2] = 2 * global_true_site_idx
        raw_cols[1::2] = 2 * global_true_site_idx + 1
        return self.hap_arr.touched_col_chunks(raw_cols)


def open_xft_founders(zarr_root: str) -> XftFounderStore:
    """Open an xftsim `HaplotypeArray` zarr store written by
    `hapgen2xft.py` (or the same layout from any other xftsim
    `save_haplotype_zarr` call). Reads `af`/`chrom`/`pos_bp`/`pos_cM`
    fully into memory (small: one float/int per doubled-hcopy column,
    tens of MB even genome-wide) and de-duplicates them down to one
    entry per true site; leaves `HaplotypeArray` itself lazy.
    """
    af = ZarrV2BloscArray(os.path.join(zarr_root, "af")).read_full_1d()
    chrom = ZarrV2BloscArray(os.path.join(zarr_root, "chrom")).read_full_1d()
    pos_bp = ZarrV2BloscArray(os.path.join(zarr_root, "pos_bp")).read_full_1d()
    pos_cM = ZarrV2BloscArray(os.path.join(zarr_root, "pos_cM")).read_full_1d()

    assert len(af) % 2 == 0, "expected an even number of columns (2 haplotype copies per true site)"
    assert np.array_equal(af[0::2], af[1::2]), "af should be identical across each site's two hcopy columns"
    assert np.array_equal(pos_bp[0::2], pos_bp[1::2]), "pos_bp should be identical across each site's two hcopy columns"
    assert np.array_equal(chrom[0::2], chrom[1::2]), "chrom should be identical across each site's two hcopy columns"

    hap_arr = ZarrV2BloscArray(os.path.join(zarr_root, "HaplotypeArray"))
    assert hap_arr.shape[1] == len(af), "HaplotypeArray's variant axis should match af's length"

    return XftFounderStore(
        zarr_root=zarr_root,
        hap_arr=hap_arr,
        true_pos_bp=pos_bp[0::2].astype(float),
        true_pos_cM=pos_cM[0::2].astype(float),
        true_af=af[0::2],
        true_chrom=chrom[0::2],
    )
