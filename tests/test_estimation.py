"""Test functions in estimation.py"""

from pathlib import Path
from typing import Dict, Union

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import scipy.sparse as sp
import torch
from conftest import sparse_matrix_equal

from cellbender.remove_background.data.io import POSTERIOR_SCHEMA, write_posterior_batch_to_parquet
from cellbender.remove_background.estimation import (
    COUNT_DATATYPE,
    MAP,
    Mean,
    MultipleChoiceKnapsack,
    SingleSample,
    ThresholdCDF,
    _estimation_array_to_csr,
    pandas_grouped_apply,
)
from cellbender.remove_background.posterior import (
    IndexConverter,
    compute_mean_target_removal_as_function,
    dense_to_sparse_op_torch,
    log_prob_sparse_to_dense,
)


@pytest.fixture(scope="module")
def log_prob_coo_base() -> Dict[str, Union[sp.coo_matrix, np.ndarray]]:
    n = -np.inf
    m = np.array(
        [
            [0, n, n, n, n, n, n, n, n, n],  # map 0, mean 0
            [n, 0, n, n, n, n, n, n, n, n],  # map 1, mean 1
            [n, n, 0, n, n, n, n, n, n, n],  # map 2, mean 2
            [-6, -2, np.log(1.0 - 2 * np.exp(-2) - np.exp(-6)), -2, n, n, n, n, n, n],
            [-2.5, -1.5, -0.5, -3, np.log(1.0 - np.exp([-2.5, -1.5, -0.5, -3]).sum()), n, n, n, n, n],
            [-0.74, -1, -2, -4, np.log(1.0 - np.exp([-0.74, -1, -2, -4]).sum()), n, n, n, n, n],
            [-1, -0.74, -2, -4, np.log(1.0 - np.exp([-0.74, -1, -2, -4]).sum()), n, n, n, n, n],
            [-2, -1, -0.74, -4, np.log(1.0 - np.exp([-0.74, -1, -2, -4]).sum()), n, n, n, n, n],
        ]
    )
    # make m sparse, i.e. zero probability entries are absent
    rows, cols, vals = dense_to_sparse_op_torch(torch.tensor(m), tensor_for_nonzeros=torch.tensor(m).exp())
    # make it a bit more difficult by having an empty row at the beginning
    rows = rows + 1
    shape = list(m.shape)
    shape[0] = shape[0] + 1

    # The last dense row (m row 7, COO row 8) has a noise count offset of 1 in the old
    # representation. We encode that offset directly into the COO column values (absolute
    # noise counts), shifting cols for that row by +1.
    last_row_mask = (rows == 8).numpy()
    cols = cols.clone()
    cols[last_row_mask] = cols[last_row_mask] + 1

    # Absolute MAP and CDF values (offset already embedded in COO cols above).
    maps = np.argmax(m, axis=1)
    shifts = np.array([0] * 7 + [1])
    maps = maps + shifts
    cdf_logic = torch.logcumsumexp(torch.tensor(m), dim=-1) > np.log(0.5)
    cdfs_list = [np.where(a)[0][0] for a in cdf_logic]
    cdfs: np.ndarray = np.array(cdfs_list) + shifts

    return {
        "coo": sp.coo_matrix((vals, (rows, cols)), shape=shape),
        "maps": np.array([0] + maps.tolist()),
        "cdfs": np.array([0] + cdfs.tolist()),
    }


@pytest.fixture(scope="module", params=["exact", "filtered", "unsorted"])
def log_prob_coo(request, log_prob_coo_base) -> Dict[str, Union[sp.coo_matrix, np.ndarray]]:
    """When used as an input argument, this offers up a series of dicts that
    can be used for tests"""
    if request.param == "exact":
        return log_prob_coo_base

    elif request.param == "filtered":
        coo = log_prob_coo_base["coo"]
        logic = coo.data >= -6
        new_coo = sp.coo_matrix((coo.data[logic], (coo.row[logic], coo.col[logic])), shape=coo.shape)
        out = {"coo": new_coo}
        out.update({k: v for k, v in log_prob_coo_base.items() if (k != "coo")})
        return out

    elif request.param == "unsorted":
        coo = log_prob_coo_base["coo"]
        order = np.random.permutation(np.arange(len(coo.data)))
        new_coo = sp.coo_matrix((coo.data[order], (coo.row[order], coo.col[order])), shape=coo.shape)
        out = {"coo": new_coo}
        out.update({k: v for k, v in log_prob_coo_base.items() if (k != "coo")})
        return out

    else:
        raise ValueError(f'Test writing error: requested "{request.param}" log_prob_coo')


def test_mean_massive_m(log_prob_coo):
    """Sets up a posterior COO with massive m values that are > max(int32).
    Will trigger github issue #252 if a bug exists, no assertion necessary.
    """

    coo = log_prob_coo["coo"]
    greater_than_max_int32 = 2200000000
    new_row = coo.row.astype(np.int64) + greater_than_max_int32
    new_shape = (coo.shape[0] + greater_than_max_int32, coo.shape[1])
    new_coo = sp.coo_matrix((coo.data, (new_row, coo.col)), shape=new_shape)
    print(f"original COO shape: {coo.shape}")
    print(f"new COO shape: {new_coo.shape}")
    print(f"new row minimum value: {new_coo.row.min()}")
    print(f"new row maximum value: {new_coo.row.max()}")
    # this is just a shim
    converter = IndexConverter(total_n_cells=new_coo.shape[0], total_n_genes=new_coo.shape[1])

    # set up and estimate
    estimator = Mean(index_converter=converter)
    _noise_csr = estimator.estimate_noise(noise_log_prob_coo=new_coo)


@pytest.fixture(scope="module", params=["exact", "filtered", "unsorted"])
def mckp_log_prob_coo(request, log_prob_coo_base) -> Dict[str, Union[sp.coo_matrix, np.ndarray]]:
    """When used as an input argument, this offers up a series of dicts that
    can be used for tests.

    NOTE: separate for MCKP because we cannot include an empty 'm' because it
    throws everything off (which gene is what, etc.)
    """

    def _fix(v):
        if isinstance(v, sp.coo_matrix):
            return _eliminate_row_zero(v)
        else:
            return v

    def _eliminate_row_zero(coo_: sp.coo_matrix) -> sp.coo_matrix:
        row = coo_.row - 1
        shape = list(coo_.shape)
        shape[0] = shape[0] - 1
        return sp.coo_matrix((coo_.data, (row, coo_.col)), shape=shape)

    if request.param == "exact":
        out = log_prob_coo_base

    elif request.param == "filtered":
        coo = log_prob_coo_base["coo"]
        logic = coo.data >= -6
        new_coo = sp.coo_matrix((coo.data[logic], (coo.row[logic], coo.col[logic])), shape=coo.shape)
        out = {"coo": new_coo}
        out.update({k: v for k, v in log_prob_coo_base.items() if (k != "coo")})

    elif request.param == "unsorted":
        coo = log_prob_coo_base["coo"]
        order = np.random.permutation(np.arange(len(coo.data)))
        new_coo = sp.coo_matrix((coo.data[order], (coo.row[order], coo.col[order])), shape=coo.shape)
        out = {"coo": new_coo}
        out.update({k: v for k, v in log_prob_coo_base.items() if (k != "coo")})

    else:
        raise ValueError(f'Test writing error: requested "{request.param}" log_prob_coo')

    return {k: _fix(v) for k, v in out.items()}


@pytest.fixture(scope="module")
def log_prob_parquet(log_prob_coo_base, tmp_path_factory) -> Dict[str, Union[Path, np.ndarray, sp.coo_matrix]]:
    """Write the base COO fixture to a parquet file for SQL-path tests.

    Uses IndexConverter(total_n_cells=1, total_n_genes=n) so m == gene_id,
    matching what the estimator tests use.
    """
    coo = log_prob_coo_base["coo"]
    n_genes = coo.shape[0]
    converter = IndexConverter(total_n_cells=1, total_n_genes=n_genes)

    tmp_dir = tmp_path_factory.mktemp("parquet")
    parquet_path = tmp_dir / "posterior.parquet"

    cell_ids, gene_ids = converter.get_ng_indices(m_inds=coo.row.astype(np.int64))
    writer = pq.ParquetWriter(str(parquet_path), POSTERIOR_SCHEMA)
    write_posterior_batch_to_parquet(
        writer=writer,
        cell_ids=cell_ids.astype(np.int32),
        gene_ids=gene_ids.astype(np.int32),
        c_vals=coo.col.astype(np.int16),
        log_probs=coo.data.astype(np.float32),
        regularized=False,
    )
    writer.close()

    return {
        "path": parquet_path,
        "n_genes": n_genes,
        "coo": coo,
        "maps": log_prob_coo_base["maps"],
        "cdfs": log_prob_coo_base["cdfs"],
    }


def test_single_sample(log_prob_coo):
    """Test the single sample estimator"""

    # the input
    print(log_prob_coo)
    print("input log probs")
    dense = log_prob_sparse_to_dense(log_prob_coo["coo"])
    print(dense)

    # with this shape converter, we get one row, where each value is one m
    converter = IndexConverter(total_n_cells=1, total_n_genes=log_prob_coo["coo"].shape[0])

    # set up and estimate
    estimator = SingleSample(index_converter=converter)
    noise_csr = estimator.estimate_noise(noise_log_prob_coo=log_prob_coo["coo"])

    # output
    print("dense noise count estimate, per m")
    out_per_m = np.array(noise_csr.todense()).squeeze()
    print(out_per_m)

    # test: allowed values are the nonzero column indices in the dense representation
    # (which already encode absolute noise counts since offsets were removed)
    dense = log_prob_sparse_to_dense(log_prob_coo["coo"])
    for i in range(1, log_prob_coo["coo"].shape[0]):
        row = dense[i, :]
        if not np.any(row > -np.inf):
            continue  # empty row
        print(f'testing "m" value {i}')
        allowed_vals = np.arange(dense.shape[1])[row > -np.inf]
        print("allowed values")
        print(allowed_vals)
        print("sample")
        print(out_per_m[i])
        assert out_per_m[i] in allowed_vals, f"sample {out_per_m[i]} is not allowed for {row}"


def test_mean(log_prob_coo):
    """Test the mean estimator"""

    # the input
    print(log_prob_coo)
    print("input log probs")
    dense = log_prob_sparse_to_dense(log_prob_coo["coo"])
    print(dense)

    # with this shape converter, we get one row, where each value is one m
    converter = IndexConverter(total_n_cells=1, total_n_genes=log_prob_coo["coo"].shape[0])

    # set up and estimate
    estimator = Mean(index_converter=converter)
    noise_csr = estimator.estimate_noise(noise_log_prob_coo=log_prob_coo["coo"])

    # output
    print("dense noise count estimate, per m")
    out_per_m = np.array(noise_csr.todense()).squeeze()
    print(out_per_m)

    # truth: column positions in the dense matrix ARE the absolute noise counts
    brute_force = np.matmul(np.arange(dense.shape[1]), np.exp(dense).transpose())
    print("truth")
    print(brute_force)

    # test
    np.testing.assert_allclose(out_per_m, brute_force)


def test_map(log_prob_coo):
    """Test the MAP estimator"""

    # the input
    print(log_prob_coo)
    print("input log probs")
    print(log_prob_sparse_to_dense(log_prob_coo["coo"]))

    # with this shape converter, we get one row, where each value is one m
    converter = IndexConverter(total_n_cells=1, total_n_genes=log_prob_coo["coo"].shape[0])

    # set up and estimate
    estimator = MAP(index_converter=converter)
    noise_csr = estimator.estimate_noise(noise_log_prob_coo=log_prob_coo["coo"])

    # output
    print("dense noise count estimate, per m")
    out_per_m = np.array(noise_csr.todense()).squeeze()
    print(out_per_m)
    print("truth")
    print(log_prob_coo["maps"])

    # test
    np.testing.assert_array_equal(out_per_m, log_prob_coo["maps"])


def test_cdf(log_prob_coo):
    """Test the estimator based on CDF thresholding"""

    # the input
    print(log_prob_coo)
    print("input log probs")
    print(log_prob_sparse_to_dense(log_prob_coo["coo"]))

    # with this shape converter, we get one row, where each value is one m
    converter = IndexConverter(total_n_cells=1, total_n_genes=log_prob_coo["coo"].shape[0])

    # set up and estimate
    estimator = ThresholdCDF(index_converter=converter)
    noise_csr = estimator.estimate_noise(noise_log_prob_coo=log_prob_coo["coo"], q=0.5)

    # output
    print("dense noise count estimate, per m")
    out_per_m = np.array(noise_csr.todense()).squeeze()
    print(out_per_m)
    print("truth")
    print(log_prob_coo["cdfs"])

    # test
    np.testing.assert_array_equal(out_per_m, log_prob_coo["cdfs"])


# ---------------------------------------------------------------------------
# SQL / parquet-path tests for MAP, Mean, ThresholdCDF, SingleSample
# ---------------------------------------------------------------------------


def test_map_from_parquet(log_prob_parquet):
    """MAP from parquet path (SQL argmax) matches known truth."""
    path = log_prob_parquet["path"]
    n_genes = log_prob_parquet["n_genes"]
    converter = IndexConverter(total_n_cells=1, total_n_genes=n_genes)
    estimator = MAP(index_converter=converter)
    noise_csr = estimator.estimate_noise(noise_log_prob_coo=path)
    out_per_m = np.array(noise_csr.todense()).squeeze()
    np.testing.assert_array_equal(out_per_m, log_prob_parquet["maps"])


def test_mean_from_parquet(log_prob_parquet):
    """Mean from parquet path (SQL weighted sum) matches brute-force truth."""
    path = log_prob_parquet["path"]
    n_genes = log_prob_parquet["n_genes"]
    coo = log_prob_parquet["coo"]
    converter = IndexConverter(total_n_cells=1, total_n_genes=n_genes)
    estimator = Mean(index_converter=converter)
    noise_csr = estimator.estimate_noise(noise_log_prob_coo=path)
    out_per_m = np.array(noise_csr.todense()).squeeze()

    dense = log_prob_sparse_to_dense(coo)
    brute_force = np.matmul(np.arange(dense.shape[1]), np.exp(dense).T)
    # rtol=1e-4: log_probs stored as float32 in parquet vs float64 brute force
    np.testing.assert_allclose(out_per_m, brute_force, rtol=1e-4)


def test_cdf_from_parquet(log_prob_parquet):
    """ThresholdCDF from parquet path (SQL window cumsum) matches known truth."""
    path = log_prob_parquet["path"]
    n_genes = log_prob_parquet["n_genes"]
    converter = IndexConverter(total_n_cells=1, total_n_genes=n_genes)
    estimator = ThresholdCDF(index_converter=converter)
    noise_csr = estimator.estimate_noise(noise_log_prob_coo=path, q=0.5)
    out_per_m = np.array(noise_csr.todense()).squeeze()
    np.testing.assert_array_equal(out_per_m, log_prob_parquet["cdfs"])


def test_single_sample_from_parquet(log_prob_parquet):
    """SingleSample from parquet path (Gumbel-max trick) returns valid noise counts."""
    path = log_prob_parquet["path"]
    n_genes = log_prob_parquet["n_genes"]
    coo = log_prob_parquet["coo"]
    converter = IndexConverter(total_n_cells=1, total_n_genes=n_genes)
    estimator = SingleSample(index_converter=converter)
    noise_csr = estimator.estimate_noise(noise_log_prob_coo=path)
    out_per_m = np.array(noise_csr.todense()).squeeze()

    dense = log_prob_sparse_to_dense(coo)
    for i in range(1, coo.shape[0]):
        row = dense[i, :]
        if not np.any(row > -np.inf):
            continue
        allowed_vals = np.arange(dense.shape[1])[row > -np.inf]
        assert out_per_m[i] in allowed_vals, f"sample {out_per_m[i]} not in allowed set for m={i}"


# ---------------------------------------------------------------------------
# Cross-path consistency: COO path == parquet path (deterministic estimators)
# ---------------------------------------------------------------------------


def test_map_coo_vs_parquet_consistency(log_prob_parquet):
    """MAP produces identical results from COO and parquet inputs."""
    coo = log_prob_parquet["coo"]
    path = log_prob_parquet["path"]
    n_genes = log_prob_parquet["n_genes"]
    converter = IndexConverter(total_n_cells=1, total_n_genes=n_genes)
    estimator = MAP(index_converter=converter)
    assert sparse_matrix_equal(
        estimator.estimate_noise(noise_log_prob_coo=coo),
        estimator.estimate_noise(noise_log_prob_coo=path),
    )


def test_mean_coo_vs_parquet_consistency(log_prob_parquet):
    """Mean produces close results from COO and parquet inputs."""
    coo = log_prob_parquet["coo"]
    path = log_prob_parquet["path"]
    n_genes = log_prob_parquet["n_genes"]
    converter = IndexConverter(total_n_cells=1, total_n_genes=n_genes)
    estimator = Mean(index_converter=converter)
    coo_result = np.array(estimator.estimate_noise(noise_log_prob_coo=coo).todense()).squeeze()
    path_result = np.array(estimator.estimate_noise(noise_log_prob_coo=path).todense()).squeeze()
    # float32 quantization in parquet vs float64 COO path
    np.testing.assert_allclose(coo_result, path_result, rtol=1e-4)


def test_cdf_coo_vs_parquet_consistency(log_prob_parquet):
    """ThresholdCDF produces identical results from COO and parquet inputs."""
    coo = log_prob_parquet["coo"]
    path = log_prob_parquet["path"]
    n_genes = log_prob_parquet["n_genes"]
    converter = IndexConverter(total_n_cells=1, total_n_genes=n_genes)
    estimator = ThresholdCDF(index_converter=converter)
    assert sparse_matrix_equal(
        estimator.estimate_noise(noise_log_prob_coo=coo, q=0.5),
        estimator.estimate_noise(noise_log_prob_coo=path, q=0.5),
    )


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------


def test_cdf_parquet_coalesce_fallback(tmp_path_factory):
    """ThresholdCDF returns max(c) when the whole distribution's mass is below q."""
    tmp_dir = tmp_path_factory.mktemp("cdf_edge")
    parquet_path = tmp_dir / "posterior.parquet"

    # Two entries whose total probability (0.6) never exceeds q=0.9.
    log_prob = float(np.log(0.3))
    writer = pq.ParquetWriter(str(parquet_path), POSTERIOR_SCHEMA)
    write_posterior_batch_to_parquet(
        writer=writer,
        cell_ids=np.array([0, 0], dtype=np.int32),
        gene_ids=np.array([0, 0], dtype=np.int32),
        c_vals=np.array([5, 10], dtype=np.int16),
        log_probs=np.array([log_prob, log_prob], dtype=np.float32),
        regularized=False,
    )
    writer.close()

    converter = IndexConverter(total_n_cells=1, total_n_genes=1)
    estimator = ThresholdCDF(index_converter=converter)

    # SQL path: COALESCE to MAX(c) = 10
    result_parquet = estimator.estimate_noise(noise_log_prob_coo=parquet_path, q=0.9)
    assert result_parquet.toarray()[0, 0] == 10

    # COO path: clamp to len(col_values)-1 = 1, col_values[1] = 10
    coo = sp.coo_matrix(
        (np.array([log_prob, log_prob]), (np.array([0, 0]), np.array([5, 10]))),
        shape=(1, 11),
    )
    result_coo = estimator.estimate_noise(noise_log_prob_coo=coo, q=0.9)
    assert result_coo.toarray()[0, 0] == 10


def test_mean_parquet_numerical_stability(tmp_path_factory):
    """Mean SQL path uses log-max shift so extreme log_probs don't cause NaN."""
    tmp_dir = tmp_path_factory.mktemp("mean_stability")
    parquet_path = tmp_dir / "posterior.parquet"

    # Extreme values that would cause EXP underflow without the stability trick.
    # Without shift: EXP(-1000) = 0 in float64 → division by 0 → NaN.
    cs = np.array([3, 5], dtype=np.int16)
    log_probs = np.array([-1000.0, -1001.0], dtype=np.float32)

    writer = pq.ParquetWriter(str(parquet_path), POSTERIOR_SCHEMA)
    write_posterior_batch_to_parquet(
        writer=writer,
        cell_ids=np.zeros(2, dtype=np.int32),
        gene_ids=np.zeros(2, dtype=np.int32),
        c_vals=cs,
        log_probs=log_probs,
        regularized=False,
    )
    writer.close()

    converter = IndexConverter(total_n_cells=1, total_n_genes=1)
    result = Mean(index_converter=converter).estimate_noise(noise_log_prob_coo=parquet_path)

    assert np.isfinite(result.toarray()[0, 0]), "Mean returned NaN/Inf for extreme log_probs"
    # p(c=3) = e^0 / (e^0 + e^{-1}),  p(c=5) = e^{-1} / (e^0 + e^{-1})
    denom = 1.0 + np.exp(-1.0)
    expected = 3.0 / denom + 5.0 * np.exp(-1.0) / denom
    np.testing.assert_allclose(result.toarray()[0, 0], expected, rtol=1e-4)


def test_compute_mean_target_removal_from_parquet(log_prob_parquet):
    """compute_mean_target_removal_as_function accepts a parquet Path and agrees with the COO path."""
    path = log_prob_parquet["path"]
    coo = log_prob_parquet["coo"]
    n_genes = log_prob_parquet["n_genes"]
    converter = IndexConverter(total_n_cells=1, total_n_genes=n_genes)

    # Simple raw count matrix: one count per gene (shape matches the 1-cell converter)
    raw_counts = sp.csr_matrix(np.ones((1, n_genes), dtype=np.float32))

    target_fun_parquet = compute_mean_target_removal_as_function(
        noise_count_posterior_coo=path,
        index_converter=converter,
        raw_count_csr_for_cells=raw_counts,
        n_cells=1,
        device="cpu",
        per_gene=True,
    )
    result_parquet = target_fun_parquet(0.01).numpy()

    target_fun_coo = compute_mean_target_removal_as_function(
        noise_count_posterior_coo=coo,
        index_converter=converter,
        raw_count_csr_for_cells=raw_counts,
        n_cells=1,
        device="cpu",
        per_gene=True,
    )
    result_coo = target_fun_coo(0.01).numpy()

    np.testing.assert_allclose(result_parquet, result_coo, rtol=1e-4)


@pytest.mark.parametrize(
    "n_chunks, parallel_compute",
    ([1, False], [2, False], [2, True]),
    ids=["1chunk", "2chunks_1cpu", "2chunks_parallel"],
)
@pytest.mark.parametrize(
    "n_cells, target, truth, truth_mat",
    (
        [1, np.zeros(8), np.array([0, 1, 2, 0, 0, 0, 0, 1]), None],
        [1, np.ones(8), np.array([0, 1, 2, 1, 1, 1, 1, 1]), None],
        [1, np.ones(8) * 2, np.array([0, 1, 2, 2, 2, 2, 2, 2]), None],
        [4, np.zeros(2), np.array([2, 2]), None],
        [4, np.ones(2) * 4, np.array([4, 4]), np.array([[0, 1], [2, 2], [2, 0], [0, 1]])],
        [4, np.ones(2) * 9, np.array([9, 9]), np.array([[0, 1], [2, 3], [4, 2], [3, 3]])],
    ),
    ids=[
        "1_cell_target_0",
        "1_cell_target_1",
        "1_cell_target_2",
        "4_cell_target_0",
        "4_cell_target_4",
        "4_cell_target_9",
    ],
)
def test_mckp(mckp_log_prob_coo, n_cells, target, truth, truth_mat, n_chunks, parallel_compute):
    """Test the multiple choice knapsack problem estimator"""

    # the input
    print("input log probs ===============================================")
    print(log_prob_sparse_to_dense(mckp_log_prob_coo["coo"]))

    # set up and estimate# with this shape converter, we have 1 cell with 8 genes
    converter = IndexConverter(total_n_cells=n_cells, total_n_genes=mckp_log_prob_coo["coo"].shape[0] // n_cells)
    estimator = MultipleChoiceKnapsack(index_converter=converter)
    noise_csr = estimator.estimate_noise(
        noise_log_prob_coo=mckp_log_prob_coo["coo"],
        noise_targets_per_gene=target,
        verbose=True,
        n_chunks=n_chunks,
        use_multiple_processes=parallel_compute,
    )

    assert noise_csr.shape == (converter.total_n_cells, converter.total_n_genes)

    # output
    print("dense noise count estimate")
    out_mat = np.array(noise_csr.todense())
    print(out_mat)
    print("noise counts per gene")
    out = out_mat.sum(axis=0)
    print(out)
    print("truth")
    print(truth)

    # test
    if truth_mat is not None:
        np.testing.assert_array_equal(out_mat, truth_mat)
    np.testing.assert_array_equal(out, truth)


def _firstval(df):
    return df["log_prob"].iat[0]


def _meanval(df):
    return df["log_prob"].mean()


@pytest.mark.parametrize("fun", (_firstval, _meanval), ids=["first_value", "mean"])
def test_parallel_pandas_grouped_apply(fun):
    """Test that the parallel apply gives the same thing as non-parallel"""

    df = pd.DataFrame(
        data={"m": [0, 0, 0, 1, 1, 1, 2, 2, 2], "c": [0, 1, 2] * 3, "log_prob": [1, 2, 3, 4, 5, 6, 7, 8, 9]}
    )
    print("input data")
    print(df)

    reg = pandas_grouped_apply(
        coo=sp.coo_matrix((df["log_prob"], (df["m"], df["c"])), shape=[3, 3]),
        fun=fun,
        parallel=False,
    )
    print("normal application of groupby apply")
    print(reg)

    parallel = pandas_grouped_apply(
        coo=sp.coo_matrix((df["log_prob"], (df["m"], df["c"])), shape=[3, 3]),
        fun=fun,
        parallel=True,
    )
    print("parallel application of groupby apply")
    print(parallel)

    np.testing.assert_array_equal(reg["m"], parallel["m"])
    np.testing.assert_array_equal(reg["result"], parallel["result"])


def test_estimation_array_to_csr():

    larger_than_uint16 = 2**16 + 1

    converter = IndexConverter(total_n_cells=larger_than_uint16, total_n_genes=larger_than_uint16)
    m = larger_than_uint16 + np.arange(-10, 10)
    data = np.random.rand(len(m)) * -10

    output_csr = _estimation_array_to_csr(index_converter=converter, data=data, m=m, dtype=COUNT_DATATYPE)

    # reimplementation here with totally permissive datatypes
    cell_and_gene_dtype = np.float64
    row, col = converter.get_ng_indices(m_inds=m)
    coo = sp.coo_matrix(
        (data.astype(COUNT_DATATYPE), (row.astype(cell_and_gene_dtype), col.astype(cell_and_gene_dtype))),
        shape=converter.matrix_shape,
        dtype=COUNT_DATATYPE,
    )
    coo.sum_duplicates()
    truth_csr = coo.tocsr()

    assert sparse_matrix_equal(output_csr, truth_csr)
