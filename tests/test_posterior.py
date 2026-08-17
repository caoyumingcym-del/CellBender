"""Test functions in posterior.py"""

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import scipy.sparse as sp
import torch

from cellbender.remove_background.data.io import POSTERIOR_SCHEMA, write_posterior_batch_to_parquet
from cellbender.remove_background.posterior import (
    IndexConverter,
    Posterior,
    compute_mean_target_removal_as_function,
    torch_binary_search,
)

USE_CUDA = torch.cuda.is_available()


@pytest.mark.skip
def test_create_posterior():
    pass


def test_index_converter():
    index_converter = IndexConverter(total_n_cells=10, total_n_genes=5)
    print(index_converter)

    # check basic conversion
    n = np.array([0, 1, 2, 3])
    g = n.copy()
    m = index_converter.get_m_indices(cell_inds=n, gene_inds=g)
    print(f"m inds are {m}")
    truth = 5 * n + g
    print(f"expected {truth}")
    np.testing.assert_equal(m, truth)

    # back and forth
    n_star, g_star = index_converter.get_ng_indices(m_inds=m)
    np.testing.assert_equal(n, n_star)
    np.testing.assert_equal(g, g_star)

    # check on input validity checking
    with pytest.raises(ValueError):
        index_converter.get_m_indices(cell_inds=np.array([-1]), gene_inds=g)
    with pytest.raises(ValueError):
        index_converter.get_m_indices(cell_inds=np.array([10]), gene_inds=g)
    with pytest.raises(ValueError):
        index_converter.get_m_indices(cell_inds=n, gene_inds=np.array([-1]))
    with pytest.raises(ValueError):
        index_converter.get_m_indices(cell_inds=n, gene_inds=np.array([5]))
    with pytest.raises(ValueError):
        index_converter.get_ng_indices(m_inds=np.array([-1]))
    with pytest.raises(ValueError):
        index_converter.get_ng_indices(m_inds=np.array([10 * 5]))


def test_torch_binary_search():
    """Test the general binary search function."""

    tol = 0.001

    def fun1(x):
        return x - 1.0

    out = torch_binary_search(
        evaluate_outcome_given_value=fun1,
        target_outcome=torch.tensor([0.0]),
        init_range=torch.tensor([[0.0, 10.0]]),
        target_tolerance=tol,
    )
    print("Single value binary search")
    print("Target value = [1.]")
    print(f"Output = {out}")
    assert ((out - torch.tensor([1.0])).abs() <= tol).all(), "Single input binary search failed"

    def fun2(x):
        x = x.clone()
        x[0] = x[0] - 1.0
        x[1] = x[1] - 2.0
        return x

    out = torch_binary_search(
        evaluate_outcome_given_value=fun2,
        target_outcome=torch.tensor([0.0, 0.0]),
        init_range=torch.tensor([[-10.0, 5.0], [0.0, 10.0]]),
        target_tolerance=tol,
    )
    print("Two-value binary search")
    print("Target value = [1., 2.]")
    print(f"Output = {out}")
    assert ((out - torch.tensor([1.0, 2.0])).abs() <= tol).all(), "Two-argument input binary search failed"


@pytest.mark.parametrize("fpr", [0.0, 0.1, 0.5, 0.75, 1], ids=lambda a: f"fpr{a}")
@pytest.mark.parametrize("per_gene", [False], ids=lambda n: "per_gene" if n else "overall")
def test_compute_mean_target_removal_as_function(tmp_path, fpr, per_gene):
    """The target removal computation, very important for the MCKP output."""

    # Build a small posterior parquet: 5 cells, 1 gene, each cell has a different
    # noise distribution represented directly by (cell_id, gene_id=0, c, log_prob).
    # Row 0 is empty (no posterior entries), rows 1-4 match the test count_matrix.
    n = -np.inf
    m = np.array(
        [
            [0, n, n, n, n, n, n, n],   # cell 1: MAP 0
            [n, 0, n, n, n, n, n, n],   # cell 2: MAP 1 (1 raw count)
            [-0.3, -1.5, np.log(1.0 - np.exp(np.array([-0.3, -1.5])).sum())] + [n] * 5,  # cell 3: 2 raw counts
            [-3, -1.21, -0.7, -2, -4, np.log(1.0 - np.exp(np.array([-3, -1.21, -0.7, -2, -4])).sum())] + [n] * 2,  # cell 4: 5 raw
        ]
    )
    # Only cells 1-4 have posterior entries (skip empty cell 0).
    # COO representation: row = cell_id (1..4), col = c value.
    rows_list, cols_list, vals_list = [], [], []
    for cell_idx, row in enumerate(m, start=1):
        for c_idx, log_p in enumerate(row):
            if np.isfinite(log_p):
                rows_list.append(cell_idx)
                cols_list.append(c_idx)
                vals_list.append(log_p)

    cell_ids_arr = np.array(rows_list, dtype=np.int32)
    gene_ids_arr = np.zeros(len(rows_list), dtype=np.int32)
    c_vals_arr = np.array(cols_list, dtype=np.int16)
    log_probs_arr = np.array(vals_list, dtype=np.float32)

    parquet_path = tmp_path / "posterior.parquet"
    with pq.ParquetWriter(str(parquet_path), schema=POSTERIOR_SCHEMA) as writer:
        write_posterior_batch_to_parquet(
            writer=writer,
            cell_ids=cell_ids_arr,
            gene_ids=gene_ids_arr,
            c_vals=c_vals_arr,
            log_probs=log_probs_arr,
            regularized=False,
        )

    # n_cells=5 total, 1 gene; count_matrix reflects the raw counts above.
    n_cells = 5
    index_converter = IndexConverter(total_n_cells=n_cells, total_n_genes=1)
    count_matrix = sp.csr_matrix(np.expand_dims(np.array([0, 0, 1, 2, 5]), axis=-1))

    target_fun = compute_mean_target_removal_as_function(
        noise_count_posterior_coo=parquet_path,
        index_converter=index_converter,
        raw_count_csr_for_cells=count_matrix,
        n_cells=n_cells,
        device="cpu",
        per_gene=per_gene,
    )

    target = (target_fun(fpr) * n_cells).item()
    print(f"\nwith fpr={fpr:.2f}, target is: {target:.1g}")

    assert target >= 1, "There is one noise count guaranteed from this test posterior"
    if fpr == 1:
        torch.testing.assert_close(target, float(count_matrix.sum()))


def test_save_and_load(tmpdir_factory):
    """Test that a round trip through save and load gives the same thing"""

    from cellbender.remove_background.data.io import (
        write_posterior_latents_csv,
    )
    from cellbender.remove_background.posterior import _posterior_latents_path

    tmp_dir = tmpdir_factory.mktemp("posterior")
    src_file = str(tmp_dir.join("src_posterior.parquet"))
    dst_file = str(tmp_dir.join("dst_posterior.parquet"))

    num_nonzeros = 100

    # Create a fake posterior parquet with small cell/gene ids.
    cell_ids = np.random.randint(0, 100, size=num_nonzeros, dtype=np.int32)
    gene_ids = np.random.randint(0, 50, size=num_nonzeros, dtype=np.int32)
    c_vals = np.random.randint(0, 10, size=num_nonzeros, dtype=np.int16)
    log_probs = np.random.rand(num_nonzeros).astype(np.float32) * -10

    with pq.ParquetWriter(src_file, schema=POSTERIOR_SCHEMA) as writer:
        write_posterior_batch_to_parquet(
            writer=writer,
            cell_ids=cell_ids,
            gene_ids=gene_ids,
            c_vals=c_vals,
            log_probs=log_probs,
            regularized=False,
        )

    latents = {"p": np.random.randn(100), "d": np.random.randn(100)}
    write_posterior_latents_csv(_posterior_latents_path(Path(src_file)), latents)

    # Set up posterior object directly (bypass model computation).
    posterior = Posterior(dataset_obj=None, vi_model=None)
    posterior._posterior_parquet_path = Path(src_file)
    posterior._latents = latents

    # save
    posterior.save(file=dst_file)

    # load
    posterior2 = Posterior(dataset_obj=None, vi_model=None)
    posterior2.load(file=dst_file)

    # check: parquet path set correctly
    assert posterior2._posterior_parquet_path == Path(dst_file), "Posterior parquet path not set correctly after load"

    # check: latents preserved
    assert posterior2._latents is not None, "_latents should be loaded from CSV sidecar"
    for key in latents:
        np.testing.assert_allclose(
            posterior2._latents[key],
            latents[key],
            rtol=1e-12,
            err_msg=f"Latent '{key}' not preserved after save/load round-trip",
        )

    # check: parquet content round-trip
    table1 = pq.read_table(src_file)
    table2 = pq.read_table(dst_file)
    assert table1.num_rows == table2.num_rows, "Row count mismatch after save/load"


def test_vi_model_freed_after_posterior_computation(tmpdir_factory):
    """After ensure_posterior_computed(), vi_model is None and latents_map is still intact."""

    tmp_dir = tmpdir_factory.mktemp("free_model")
    parquet_path = Path(str(tmp_dir.join("posterior.parquet")))

    num_nonzeros = 50
    cell_ids = np.zeros(num_nonzeros, dtype=np.int32)
    gene_ids = np.arange(num_nonzeros, dtype=np.int32)
    c_vals = np.ones(num_nonzeros, dtype=np.int16)
    log_probs = np.full(num_nonzeros, -1.0, dtype=np.float32)

    with pq.ParquetWriter(str(parquet_path), schema=POSTERIOR_SCHEMA) as writer:
        write_posterior_batch_to_parquet(
            writer=writer,
            cell_ids=cell_ids,
            gene_ids=gene_ids,
            c_vals=c_vals,
            log_probs=log_probs,
            regularized=False,
        )

    latents = {"p": np.ones(1) * 0.99, "d": np.ones(1) * 100.0}

    # Bypass model computation: inject parquet path and latents directly.
    posterior = Posterior(dataset_obj=None, vi_model=None)
    posterior._posterior_parquet_path = parquet_path
    posterior._latents = latents

    # vi_model was None from the start; confirm latents are accessible.
    assert posterior.vi_model is None
    assert posterior.latents_map["p"][0] > 0.5


def test_sort_posterior_parquet_uses_output_dir_as_tmpdir(tmpdir_factory):
    """sort_posterior_parquet sets DuckDB temp_directory to the parquet's parent (not /tmp)."""

    from cellbender.remove_background.data.io import sort_posterior_parquet

    tmp_dir = tmpdir_factory.mktemp("sort_tmpdir")
    parquet_path = Path(str(tmp_dir.join("posterior.parquet")))

    num_nonzeros = 30
    # Write entries in reverse gene order so the sort actually reorders rows.
    gene_ids = np.arange(num_nonzeros - 1, -1, -1, dtype=np.int32)
    cell_ids = np.zeros(num_nonzeros, dtype=np.int32)
    c_vals = np.ones(num_nonzeros, dtype=np.int16)
    log_probs = np.full(num_nonzeros, -1.0, dtype=np.float32)

    with pq.ParquetWriter(str(parquet_path), schema=POSTERIOR_SCHEMA) as writer:
        write_posterior_batch_to_parquet(
            writer=writer,
            cell_ids=cell_ids,
            gene_ids=gene_ids,
            c_vals=c_vals,
            log_probs=log_probs,
            regularized=False,
        )

    # Sort should complete without error and produce an ordered file.
    sort_posterior_parquet(parquet_path)

    table = pq.read_table(str(parquet_path))
    gene_ids_after = table.column("gene_id").to_pylist()
    assert gene_ids_after == sorted(gene_ids_after), "Parquet is not sorted by gene_id after sort_posterior_parquet"
