"""Classes and methods for estimation of noise counts, given a posterior."""

import concurrent.futures
import logging
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp

from cellbender.remove_background.data.io import _make_duckdb_conn

logger = logging.getLogger("cellbender")

_STREAM_BATCH_SIZE = 2_000_000

COUNT_DATATYPE = np.int32


class EstimationMethod(ABC):
    """Base class for estimation of noise counts, given a posterior."""

    def __init__(self, n_cells: int, n_genes: int):
        self.n_cells = n_cells
        self.n_genes = n_genes
        super(EstimationMethod, self).__init__()

    @abstractmethod
    def estimate_noise_to_parquet(
        self,
        noise_log_prob_coo: Path,
        output_path: Path,
        **kwargs,
    ) -> None:
        """Given the posterior parquet, compute noise counts and write to parquet.

        Args:
            noise_log_prob_coo: Path to the posterior parquet file.
            output_path: Destination parquet with (cell_id, gene_id, noise_count).
        """
        pass


class SingleSample(EstimationMethod):
    """A single sample from the noise count posterior"""

    def estimate_noise_to_parquet(
        self,
        noise_log_prob_coo: Path,
        output_path: Path,
        duckdb_memory_limit: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Estimate noise counts and write to parquet without Python materialisation."""
        query = """
            SELECT
                cell_id,
                gene_id,
                CAST(argmax(c, log_prob - LN(-LN(random()))) AS INTEGER) AS noise_count
            FROM posterior
            GROUP BY cell_id, gene_id
        """
        _estimate_via_sql_to_parquet(noise_log_prob_coo, query, output_path, duckdb_memory_limit)


class Mean(EstimationMethod):
    """Posterior mean"""

    def estimate_noise_to_parquet(
        self,
        noise_log_prob_coo: Path,
        output_path: Path,
        duckdb_memory_limit: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Estimate mean noise counts and write to parquet without Python materialisation."""
        query = """
            WITH shifted AS (
                SELECT
                    cell_id,
                    gene_id,
                    c,
                    EXP(log_prob - MAX(log_prob) OVER (PARTITION BY cell_id, gene_id)) AS prob_raw
                FROM posterior
            ),
            normalized AS (
                SELECT
                    cell_id,
                    gene_id,
                    c,
                    prob_raw / SUM(prob_raw) OVER (PARTITION BY cell_id, gene_id) AS prob
                FROM shifted
            )
            SELECT
                cell_id,
                gene_id,
                SUM(CAST(c AS DOUBLE) * prob) AS noise_count
            FROM normalized
            GROUP BY cell_id, gene_id
        """
        _estimate_via_sql_to_parquet(noise_log_prob_coo, query, output_path, duckdb_memory_limit)


class MAP(EstimationMethod):
    """The canonical maximum a posteriori"""

    def estimate_noise_to_parquet(
        self,
        noise_log_prob_coo: Path,
        output_path: Path,
        duckdb_memory_limit: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Estimate MAP noise counts and write to parquet without Python materialisation."""
        query = """
            SELECT
                cell_id,
                gene_id,
                CAST(argmax(c, log_prob) AS INTEGER) AS noise_count
            FROM posterior
            GROUP BY cell_id, gene_id
        """
        _estimate_via_sql_to_parquet(noise_log_prob_coo, query, output_path, duckdb_memory_limit)


class ThresholdCDF(EstimationMethod):
    """Noise estimation via thresholding the noise count CDF"""

    def estimate_noise_to_parquet(
        self,
        noise_log_prob_coo: Path,
        output_path: Path,
        q: float = 0.5,
        duckdb_memory_limit: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Estimate CDF-threshold noise counts and write to parquet."""
        query = f"""
            WITH cumulative AS (
                SELECT
                    cell_id,
                    gene_id,
                    c,
                    SUM(EXP(log_prob)) OVER (
                        PARTITION BY cell_id, gene_id
                        ORDER BY c
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                    ) AS cum_prob
                FROM posterior
            )
            SELECT
                cell_id,
                gene_id,
                CAST(COALESCE(
                    MIN(CASE WHEN cum_prob > {q} THEN c END),
                    MAX(c)
                ) AS INTEGER) AS noise_count
            FROM cumulative
            GROUP BY cell_id, gene_id
        """
        _estimate_via_sql_to_parquet(noise_log_prob_coo, query, output_path, duckdb_memory_limit)


def _register_posterior(
    conn: "duckdb.DuckDBPyConnection",
    source: Path,
) -> None:
    """Register the posterior parquet as a DuckDB view named 'posterior'."""
    conn.execute(f"CREATE OR REPLACE VIEW posterior AS SELECT * FROM read_parquet('{source}')")


def _ng_arrays_to_csr(
    cell_ids: np.ndarray,
    gene_ids: np.ndarray,
    data: np.ndarray,
    shape: Tuple[int, int],
    dtype=COUNT_DATATYPE,
) -> sp.csr_matrix:
    """Build a CSR sparse matrix from flat (cell_id, gene_id, data) arrays."""
    coo = sp.coo_matrix(
        (data.astype(dtype), (cell_ids.astype(np.int32), gene_ids.astype(np.int32))),
        shape=shape,
        dtype=dtype,
    )
    coo.sum_duplicates()
    return coo.tocsr()


def _estimate_via_sql_to_parquet(
    source: Path,
    query: str,
    output_path: Path,
    duckdb_memory_limit: Optional[str] = None,
) -> None:
    """Run a SQL estimation query and write ``(cell_id, gene_id, noise_count)``
    to a parquet file sorted by ``(gene_id, cell_id)``.

    The parquet is written directly by DuckDB via ``COPY TO`` — no Python-side
    DataFrame is materialised, so peak memory is only DuckDB's working set.

    Args:
        source: Path to posterior parquet.
        query: SQL selecting columns ``(cell_id, gene_id, noise_count)``.
        output_path: Destination parquet file.
        duckdb_memory_limit: DuckDB memory cap (e.g. ``'4GB'``).
    """
    tmp_dir = str(source.parent).replace("'", "''")
    conn = _make_duckdb_conn(tmp_dir, memory_limit=duckdb_memory_limit)
    _register_posterior(conn, source)
    out_str = str(output_path).replace("'", "''")
    conn.execute(
        f"COPY (SELECT * FROM ({query}) ORDER BY gene_id, cell_id) "
        f"TO '{out_str}' (FORMAT PARQUET, COMPRESSION 'snappy')"
    )


def _stream_mean_noise_gene_range(
    source: Path,
    start_gene: int,
    end_gene: int,
) -> np.ndarray:
    """Compute sum of per-cell E[noise_count] for gene_ids in [start_gene, end_gene).

    Single streaming pass over the posterior parquet (which must be sorted by
    gene_id, cell_id).  For each (cell_id, gene_id) group, the online
    log-sum-exp recurrence is used to accumulate the weighted mean without
    storing all rows simultaneously.  Groups that span batch boundaries are
    tracked via a 5-scalar pending state.

    Returns an array of shape ``(end_gene - start_gene,)`` with per-gene totals.
    """
    result = np.zeros(end_gene - start_gene, dtype=np.float64)

    pend_gene: int = -1
    pend_cell: int = -1
    pend_max_lp: float = -np.inf
    pend_numer: float = 0.0
    pend_denom: float = 0.0

    pf = pq.ParquetFile(str(source))

    # Prune row groups using column statistics (iter_batches doesn't support filters).
    rg_indices = []
    meta = pf.metadata
    for rg_idx in range(meta.num_row_groups):
        rg = meta.row_group(rg_idx)
        for col_idx in range(rg.num_columns):
            col = rg.column(col_idx)
            if col.path_in_schema == "gene_id":
                stats = col.statistics
                if stats is not None and stats.has_min_max:
                    if stats.max >= start_gene and stats.min < end_gene:
                        rg_indices.append(rg_idx)
                else:
                    rg_indices.append(rg_idx)
                break
        else:
            rg_indices.append(rg_idx)

    for batch in pf.iter_batches(
        batch_size=_STREAM_BATCH_SIZE,
        row_groups=rg_indices if rg_indices else None,
        columns=["gene_id", "cell_id", "c", "log_prob"],
    ):
        if batch.num_rows == 0:
            continue

        gene_arr = batch.column("gene_id").to_numpy()
        # Row-group pruning is coarse; filter rows outside our gene range.
        in_range = (gene_arr >= start_gene) & (gene_arr < end_gene)
        if not np.all(in_range):
            batch = batch.filter(pa.array(in_range))
            if batch.num_rows == 0:
                continue
            gene_arr = batch.column("gene_id").to_numpy()
        cell_arr = batch.column("cell_id").to_numpy()
        c_arr = batch.column("c").to_numpy().astype(np.float64)
        lp_arr = batch.column("log_prob").to_numpy().astype(np.float64)
        n = len(gene_arr)

        # Locate group boundaries: True at the first row of each (gene, cell) group.
        boundary_mask = np.empty(n, dtype=bool)
        boundary_mask[0] = True
        boundary_mask[1:] = (np.diff(gene_arr) != 0) | (np.diff(cell_arr) != 0)

        group_starts = np.where(boundary_mask)[0]
        group_ends = np.empty_like(group_starts)
        group_ends[:-1] = group_starts[1:]
        group_ends[-1] = n
        n_groups = len(group_starts)

        # --- first group: may continue from the pending state ---
        g0_s = int(group_starts[0])
        g0_e = int(group_ends[0])
        g0_gene = int(gene_arr[g0_s])
        g0_cell = int(cell_arr[g0_s])
        g0_lp = lp_arr[g0_s:g0_e]
        g0_c = c_arr[g0_s:g0_e]
        g0_max = float(np.max(g0_lp))

        if pend_gene == g0_gene and pend_cell == g0_cell:
            new_max = max(pend_max_lp, g0_max)
            scale_p = np.exp(pend_max_lp - new_max)
            pend_numer = pend_numer * scale_p + float(np.sum(g0_c * np.exp(g0_lp - new_max)))
            pend_denom = pend_denom * scale_p + float(np.sum(np.exp(g0_lp - new_max)))
            pend_max_lp = new_max
        else:
            if pend_gene >= 0 and pend_denom > 0.0:
                result[pend_gene - start_gene] += pend_numer / pend_denom
            pend_gene = g0_gene
            pend_cell = g0_cell
            pend_max_lp = g0_max
            pend_numer = float(np.sum(g0_c * np.exp(g0_lp - g0_max)))
            pend_denom = float(np.sum(np.exp(g0_lp - g0_max)))

        if n_groups == 1:
            # Entire batch is one group; it may continue in the next batch.
            continue

        # First group is now complete — flush it.
        result[pend_gene - start_gene] += pend_numer / pend_denom
        pend_gene = -1

        # --- middle groups: vectorised via reduceat ---
        if n_groups > 2:
            mid_s = int(group_starts[1])
            mid_e = int(group_ends[-2])

            mg = gene_arr[mid_s:mid_e]
            mc = c_arr[mid_s:mid_e]
            ml = lp_arr[mid_s:mid_e]
            mca = cell_arr[mid_s:mid_e]

            mid_mask = np.empty(mid_e - mid_s, dtype=bool)
            mid_mask[0] = True
            mid_mask[1:] = (np.diff(mg) != 0) | (np.diff(mca) != 0)
            mid_bounds = np.where(mid_mask)[0]

            max_lp_g = np.maximum.reduceat(ml, mid_bounds)
            group_idx = np.cumsum(mid_mask) - 1
            max_row = max_lp_g[group_idx]

            shifted = np.exp(ml - max_row)
            numer_g = np.add.reduceat(mc * shifted, mid_bounds)
            denom_g = np.add.reduceat(shifted, mid_bounds)
            mean_g = numer_g / denom_g

            np.add.at(result, mg[mid_bounds] - start_gene, mean_g)

        # --- last group: save as pending (may continue in next batch) ---
        gL_s = int(group_starts[-1])
        gL_e = int(group_ends[-1])
        gL_lp = lp_arr[gL_s:gL_e]
        gL_c = c_arr[gL_s:gL_e]
        gL_max = float(np.max(gL_lp))
        pend_gene = int(gene_arr[gL_s])
        pend_cell = int(cell_arr[gL_s])
        pend_max_lp = gL_max
        pend_numer = float(np.sum(gL_c * np.exp(gL_lp - gL_max)))
        pend_denom = float(np.sum(np.exp(gL_lp - gL_max)))

    if pend_gene >= 0 and pend_denom > 0.0:
        result[pend_gene - start_gene] += pend_numer / pend_denom

    return result


def estimate_mean_noise_per_gene(
    source: Path,
    n_genes: int,
) -> np.ndarray:
    """Return total E[noise_count] per gene, summed over all cells.

    Streams through the posterior parquet in sorted order using parallel
    readers, each covering a disjoint gene-id range.  The log-sum-exp trick
    is applied per (cell_id, gene_id) pair; no hash join or window function
    is materialised in memory, so peak RAM scales with batch size and the
    number of genes — not the total number of posterior rows.

    Args:
        source: Path to posterior parquet (must be sorted by gene_id, cell_id).
        n_genes: Total number of genes; determines output array length.

    Returns:
        mean_noise_per_gene: 1-D array of shape ``(n_genes,)`` where
            entry *g* is ``sum_over_cells(E[noise_count(cell, g)])``.
    """
    if n_genes == 0:
        return np.zeros(0, dtype=np.float64)

    n_workers = min(os.cpu_count() or 1, 8, n_genes)

    # Divide gene range as evenly as possible across workers.
    base, rem = divmod(n_genes, n_workers)
    ranges: list = []
    start = 0
    for i in range(n_workers):
        end = start + base + (1 if i < rem else 0)
        ranges.append((start, end))
        start = end

    logger.debug(
        "Streaming mean noise per gene: %d genes, %d workers, batch_size=%d",
        n_genes,
        n_workers,
        _STREAM_BATCH_SIZE,
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_stream_mean_noise_gene_range, source, s, e) for s, e in ranges]
        slices = [f.result() for f in futures]

    return np.concatenate(slices)


class MultipleChoiceKnapsack(EstimationMethod):
    """Noise estimation via the multiple-choice knapsack problem, solved with DuckDB SQL.

    DuckDB executes out-of-core by default so this scales to datasets with
    millions of non-zero posterior entries without incurring OOM errors.
    """

    def estimate_noise_to_parquet(
        self,
        noise_log_prob_coo: Path,
        output_path: Path,
        noise_targets_per_gene: Optional[np.ndarray] = None,
        duckdb_memory_limit: Optional[str] = None,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        """Estimate MCKP noise counts and write to parquet without building full CSR matrices.

        Computes MAP counts and knapsack adjustment steps in DuckDB, then uses
        a final DuckDB ``COPY TO`` to write ``(cell_id, gene_id, noise_count)``
        sorted by ``(gene_id, cell_id)``.  No full-matrix Python objects are
        created.

        Args:
            noise_log_prob_coo: Path to posterior parquet.
            output_path: Destination parquet file.
            noise_targets_per_gene: Per-gene noise count targets (required).
            duckdb_memory_limit: DuckDB memory cap.
            verbose: Log intermediate DuckDB results.
        """
        assert noise_targets_per_gene is not None, (
            "noise_targets_per_gene is required for MCKP.estimate_noise_to_parquet"
        )

        t0 = time.time()

        tmp_dir = str(noise_log_prob_coo.parent).replace("'", "''")
        conn = _make_duckdb_conn(tmp_dir, memory_limit=duckdb_memory_limit)
        _register_posterior(conn, noise_log_prob_coo)

        # Step 1: MAP estimate
        map_df = conn.execute("""
            SELECT
                cell_id,
                gene_id,
                CAST(argmax(c, log_prob) AS INTEGER) AS map_c
            FROM posterior
            GROUP BY cell_id, gene_id
        """).df()

        if verbose:
            logger.debug("MAP head:\n%s", map_df.head(10).to_string())

        # Compute per-gene adjustment needed
        map_csr = _ng_arrays_to_csr(
            cell_ids=map_df["cell_id"].values,
            gene_ids=map_df["gene_id"].values,
            data=map_df["map_c"].values.astype(COUNT_DATATYPE),
            shape=(self.n_cells, self.n_genes),
        )
        map_noise_per_gene = np.asarray(map_csr.sum(axis=0)).squeeze()
        del map_csr
        additional = (noise_targets_per_gene - map_noise_per_gene).astype(int)
        step_dir = np.sign(additional).astype(np.int32)
        topk = np.abs(additional).astype(np.int64)

        gene_targets_df = pd.DataFrame(
            {
                "gene_id": np.arange(self.n_genes, dtype=np.int32),
                "step_direction": step_dir,
                "topk": topk,
            }
        )
        gene_targets_df = gene_targets_df[gene_targets_df["step_direction"] != 0].reset_index(drop=True)

        out_str = str(output_path).replace("'", "''")

        if len(gene_targets_df) == 0:
            # MAP already matches targets — write map_df directly to parquet.
            logger.info("MCKP: MAP already matches targets for all genes.")
            conn.register("map_estimates_final", map_df[["cell_id", "gene_id", "map_c"]])
            conn.execute(
                f"COPY (SELECT cell_id, gene_id, map_c AS noise_count "
                f"FROM map_estimates_final ORDER BY gene_id, cell_id) "
                f"TO '{out_str}' (FORMAT PARQUET, COMPRESSION 'snappy')"
            )
            logger.info("Total MCKP estimation time = %.2f sec", time.time() - t0)
            return

        conn.register("gene_targets", gene_targets_df)
        conn.register("map_estimates", map_df[["cell_id", "gene_id", "map_c"]])

        # Step 2: Compute adjustment steps
        steps_df = conn.execute("""
            WITH directed AS (
                SELECT
                    p.cell_id,
                    p.gene_id,
                    p.c,
                    p.log_prob,
                    t.step_direction,
                    t.topk,
                    m.map_c,
                    LAG(p.log_prob)  OVER w AS lag_lp,
                    LEAD(p.log_prob) OVER w AS lead_lp
                FROM posterior p
                JOIN gene_targets  t ON p.gene_id = t.gene_id
                JOIN map_estimates m ON p.cell_id  = m.cell_id
                                    AND p.gene_id  = m.gene_id
                WHERE (
                        (t.step_direction =  1 AND p.c >= m.map_c)
                     OR (t.step_direction = -1 AND p.c <= m.map_c)
                  )
                WINDOW w AS (PARTITION BY p.cell_id, p.gene_id ORDER BY p.c)
            ),
            deltas AS (
                SELECT
                    cell_id,
                    gene_id,
                    step_direction,
                    topk,
                    ABS(
                        CASE WHEN step_direction =  1 THEN log_prob - lag_lp
                             ELSE                          log_prob - lead_lp
                        END
                    ) AS delta
                FROM directed
                WHERE (step_direction =  1 AND lag_lp  IS NOT NULL)
                   OR (step_direction = -1 AND lead_lp IS NOT NULL)
            ),
            ranked AS (
                SELECT
                    cell_id,
                    gene_id,
                    step_direction,
                    topk,
                    ROW_NUMBER() OVER (PARTITION BY gene_id ORDER BY delta) AS rn
                FROM deltas
            )
            SELECT
                cell_id,
                gene_id,
                CAST(COUNT(*) AS INTEGER) * any_value(step_direction) AS step_counts
            FROM ranked
            WHERE rn <= topk
            GROUP BY cell_id, gene_id
        """).df()

        if verbose:
            logger.debug("Steps head:\n%s", steps_df.head(10).to_string())

        logger.info("Total MCKP estimation time = %.2f sec", time.time() - t0)

        # Step 3: Join MAP and adjustments; write final (cell_id, gene_id, noise_count) to parquet.
        conn.register("steps_final", steps_df[["cell_id", "gene_id", "step_counts"]])
        conn.execute(
            f"COPY ("
            f"SELECT m.cell_id, m.gene_id, "
            f"       m.map_c + COALESCE(s.step_counts, 0) AS noise_count "
            f"FROM map_estimates m "
            f"LEFT JOIN steps_final s ON m.cell_id = s.cell_id AND m.gene_id = s.gene_id "
            f"ORDER BY m.gene_id, m.cell_id"
            f") TO '{out_str}' (FORMAT PARQUET, COMPRESSION 'snappy')"
        )


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
