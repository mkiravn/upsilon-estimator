"""Unit tests for upsilon_estimator._xft_zarr_reader, using small synthetic
zarr-v2 blosc-compressed fixtures built by hand (via
`_write_zarr_v2_blosc_array` below) -- exercises the exact on-disk format
`xftsim`'s `save_haplotype_zarr` produces, without depending on a real
external founders store.

`blosc` is an optional dependency (the `xft` extra); skip this whole module
if it isn't installed rather than failing collection.
"""
import json
import os

import numpy as np
import pytest

blosc = pytest.importorskip("blosc")

from upsilon_estimator._xft_zarr_reader import (
    ZarrV2BloscArray,
    _XftHaplotypeSiteAccessor,
    open_xft_founders,
)


def _write_zarr_v2_blosc_array(array_dir, array, chunks, fill_value):
    """Write `array` out as a minimal zarr-v2, blosc-compressed array
    directory -- matching the exact convention `ZarrV2BloscArray` reads:
    edge chunks padded to the full nominal chunk shape (with
    `fill_value`) before compression, C order, chunk files named by
    dot-joined chunk-index tuples."""
    os.makedirs(array_dir, exist_ok=True)
    shape = array.shape
    ndim = len(shape)
    dtype_str = array.dtype.str  # e.g. '<f8', '|i1'

    with open(os.path.join(array_dir, ".zarray"), "w") as f:
        json.dump(
            {
                "zarr_format": 2,
                "shape": list(shape),
                "chunks": list(chunks),
                "dtype": dtype_str,
                "compressor": {"id": "blosc", "cname": "lz4", "clevel": 5, "shuffle": 1, "blocksize": 0},
                "fill_value": fill_value if not (isinstance(fill_value, float) and np.isnan(fill_value)) else "NaN",
                "filters": None,
                "order": "C",
            },
            f,
        )

    n_chunks_per_dim = [-(-shape[d] // chunks[d]) for d in range(ndim)]
    chunk_idx_grid = np.ndindex(*n_chunks_per_dim)
    for chunk_idx in chunk_idx_grid:
        # slice out this chunk's region of the real array (may be ragged at edges)
        slices = tuple(
            slice(chunk_idx[d] * chunks[d], min((chunk_idx[d] + 1) * chunks[d], shape[d]))
            for d in range(ndim)
        )
        region = array[slices]
        # pad up to the full nominal chunk shape, exactly as zarr v2 does on disk
        padded = np.full(chunks, fill_value, dtype=array.dtype)
        pad_slices = tuple(slice(0, region.shape[d]) for d in range(ndim))
        padded[pad_slices] = region

        raw = padded.tobytes(order="C")
        compressed = blosc.compress(raw, typesize=array.dtype.itemsize, cname="lz4", clevel=5, shuffle=blosc.SHUFFLE)
        chunk_path = os.path.join(array_dir, ".".join(str(c) for c in chunk_idx))
        with open(chunk_path, "wb") as f:
            f.write(compressed)


def test_read_full_1d_roundtrip_with_ragged_edge_chunk(tmp_path):
    arr = np.arange(25, dtype="<f8") * 1.5
    _write_zarr_v2_blosc_array(str(tmp_path / "myarr"), arr, chunks=(10,), fill_value=np.nan)

    reader = ZarrV2BloscArray(str(tmp_path / "myarr"))
    assert reader.shape == (25,)
    assert reader.chunks == (10,)
    got = reader.read_full_1d()
    assert np.array_equal(got, arr)


def test_read_columns_2d_matches_direct_indexing_across_chunk_boundaries(tmp_path):
    rng = np.random.default_rng(0)
    # 7 "individuals" (rows), 23 raw columns -- deliberately not a clean
    # multiple of the chunk shape, to force ragged edge chunks in both dims.
    arr = rng.integers(0, 2, size=(7, 23)).astype("|i1")
    _write_zarr_v2_blosc_array(str(tmp_path / "hap"), arr, chunks=(3, 5), fill_value=0)

    reader = ZarrV2BloscArray(str(tmp_path / "hap"))
    assert reader.shape == (7, 23)

    # columns spanning multiple chunk boundaries (chunk width 5: chunks are
    # cols [0-4],[5-9],[10-14],[15-19],[20-22]), unsorted and with repeats
    col_indices = np.array([0, 4, 5, 9, 10, 22, 20, 4, 15])
    got = reader.read_columns_2d(col_indices)
    expected = arr[:, col_indices]
    assert np.array_equal(got, expected)


def test_touched_col_chunks(tmp_path):
    arr = np.zeros((4, 23), dtype="|i1")
    _write_zarr_v2_blosc_array(str(tmp_path / "hap"), arr, chunks=(3, 5), fill_value=0)
    reader = ZarrV2BloscArray(str(tmp_path / "hap"))

    assert reader.touched_col_chunks(np.array([0, 4])) == [0]
    assert reader.touched_col_chunks(np.array([0, 9])) == [0, 1]
    assert reader.touched_col_chunks(np.array([22])) == [4]


def _build_synthetic_founder_store(root, n_individuals=6, true_sites_per_chrom=(4, 5, 3)):
    """Build a full synthetic xftsim-layout store (af/chrom/pos_bp/pos_cM
    + HaplotypeArray, doubled-hcopy convention) across a few fake
    chromosomes, with a known ground-truth haploid genotype matrix to
    check reconstruction against."""
    rng = np.random.default_rng(42)
    n_true_sites = sum(true_sites_per_chrom)
    n_haplotypes = 2 * n_individuals

    # ground truth: haploid genotypes, shape (n_true_sites, n_haplotypes)
    true_haploid = rng.integers(0, 2, size=(n_true_sites, n_haplotypes)).astype("|i1")
    true_af = true_haploid.mean(axis=1)

    true_chrom = np.concatenate([np.full(n, c + 1) for c, n in enumerate(true_sites_per_chrom)])
    true_pos_bp = np.concatenate([np.arange(1, n + 1) * 1000 for n in true_sites_per_chrom]).astype(np.int64)
    true_pos_cM = true_pos_bp / 1e6  # arbitrary but monotonic within each chrom

    # double every per-site metadata array across the two hcopy columns
    af = np.repeat(true_af, 2)
    chrom = np.repeat(true_chrom, 2).astype(np.int64)
    pos_bp = np.repeat(true_pos_bp, 2)
    pos_cM = np.repeat(true_pos_cM, 2)

    # HaplotypeArray: shape (n_individuals, 2*n_true_sites), interleaved
    # hcopy0/hcopy1 columns per site, individual i's hcopy0 = haploid row
    # `i`, hcopy1 = haploid row `n_individuals + i` (arbitrary but fixed
    # convention, matching make_diploid_shape's row-interleaving logic).
    hap_array = np.empty((n_individuals, 2 * n_true_sites), dtype="|i1")
    hap_array[:, 0::2] = true_haploid[:, :n_individuals].T
    hap_array[:, 1::2] = true_haploid[:, n_individuals:].T

    _write_zarr_v2_blosc_array(os.path.join(root, "af"), af, chunks=(6,), fill_value=np.nan)
    _write_zarr_v2_blosc_array(os.path.join(root, "chrom"), chrom, chunks=(6,), fill_value=0)
    _write_zarr_v2_blosc_array(os.path.join(root, "pos_bp"), pos_bp, chunks=(6,), fill_value=0)
    _write_zarr_v2_blosc_array(os.path.join(root, "pos_cM"), pos_cM, chunks=(6,), fill_value=np.nan)
    _write_zarr_v2_blosc_array(os.path.join(root, "HaplotypeArray"), hap_array, chunks=(4, 6), fill_value=0)

    return dict(true_haploid=true_haploid, true_af=true_af, true_chrom=true_chrom,
                true_pos_bp=true_pos_bp, true_pos_cM=true_pos_cM, n_individuals=n_individuals)


def test_open_xft_founders_dedupes_metadata_and_matches_ground_truth(tmp_path):
    root = str(tmp_path / "store.zarr")
    gt = _build_synthetic_founder_store(root)

    store = open_xft_founders(root)
    assert store.n_true_sites == len(gt["true_af"])
    assert store.n_individuals == gt["n_individuals"]
    assert store.n_haplotypes == 2 * gt["n_individuals"]
    assert np.allclose(store.true_af, gt["true_af"])
    assert np.array_equal(store.true_chrom, gt["true_chrom"])
    assert np.array_equal(store.true_pos_bp, gt["true_pos_bp"])
    assert np.allclose(store.true_pos_cM, gt["true_pos_cM"])
    assert store.chromosomes() == [1, 2, 3]


def test_panel_for_chromosome_reconstructs_correct_haploid_genotypes(tmp_path):
    root = str(tmp_path / "store.zarr")
    gt = _build_synthetic_founder_store(root, true_sites_per_chrom=(4, 5, 3))
    store = open_xft_founders(root)

    offset = 0
    for chrom, n in zip([1, 2, 3], [4, 5, 3]):
        panel = store.panel_for_chromosome(chrom)
        assert panel.n_sites == n
        assert panel.n_haplotypes == store.n_haplotypes
        assert np.allclose(panel.positions, gt["true_pos_bp"][offset : offset + n])
        assert np.allclose(panel.allele_freq, gt["true_af"][offset : offset + n])
        assert np.allclose(panel.genetic_position, gt["true_pos_cM"][offset : offset + n])

        # fetch ALL of this chromosome's sites (local indices 0..n-1) and
        # check against ground truth -- also exercises fetch_sites'
        # sort/unsort-by-index logic via an unsorted, duplicated query.
        local_idx = np.array([n - 1] + list(range(n)) + [0])
        got = panel.genotypes[local_idx, :]
        expected = gt["true_haploid"][offset : offset + n][local_idx]
        assert np.array_equal(got, expected)

        offset += n


def test_zarr_v2_blosc_array_rejects_non_blosc_or_filtered(tmp_path):
    d = str(tmp_path / "bad")
    os.makedirs(d)
    with open(os.path.join(d, ".zarray"), "w") as f:
        json.dump(
            {
                "zarr_format": 2, "shape": [4], "chunks": [4], "dtype": "<f8",
                "compressor": {"id": "zstd"}, "fill_value": None, "filters": None, "order": "C",
            }, f,
        )
    with pytest.raises(AssertionError, match="blosc"):
        ZarrV2BloscArray(d)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
