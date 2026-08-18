"""Classes and methods for estimation of noise counts, given a posterior."""

import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
import scipy.sparse as sp

logger = logging.getLogger("cellbender")

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
            WHERE NOT regularized
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
                WHERE NOT regularized
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
            WHERE NOT regularized
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
                WHERE NOT regularized
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
    conn = duckdb.connect()
    tmp_dir = str(source.parent).replace("'", "''")
    conn.execute(f"SET temp_directory='{tmp_dir}'")
    if duckdb_memory_limit is not None:
        conn.execute(f"SET memory_limit='{duckdb_memory_limit}'")
    _register_posterior(conn, source)
    out_str = str(output_path).replace("'", "''")
    conn.execute(
        f"COPY (SELECT * FROM ({query}) ORDER BY gene_id, cell_id) "
        f"TO '{out_str}' (FORMAT PARQUET, COMPRESSION 'snappy')"
    )


def estimate_mean_noise_per_gene(
    source: Path,
    n_genes: int,
    duckdb_memory_limit: Optional[str] = None,
) -> np.ndarray:
    """Return total E[noise_count] per gene, summed over all cells.

    Two-level GROUP BY: first computes per-(cell, gene) mean noise via the
    log-sum-exp trick (reading the parquet twice — once for the per-pair max,
    once for the weighted sum), then aggregates those means by gene.  No window
    functions are used, so every intermediate step can spill to disk with
    predictable, bounded peak RAM.

    Args:
        source: Path to posterior parquet.
        n_genes: Total number of genes; determines output array length.
        duckdb_memory_limit: DuckDB memory cap (e.g. ``'4GB'``).  When
            *None*, DuckDB auto-detects (~80 % of system RAM).

    Returns:
        mean_noise_per_gene: 1-D array of shape ``(n_genes,)`` where
            entry *g* is ``sum_over_cells(E[noise_count(cell, g)])``.
    """
    # Two-pass query: the first scan computes max(log_prob) per (cell, gene),
    # which is joined back to the second scan to shift log-probs before exp().
    # This avoids window functions — all intermediate aggregations can spill to
    # disk, keeping peak RAM proportional to the number of unique (cell, gene)
    # pairs rather than the total number of posterior rows.
    query = """
        WITH max_per_pair AS (
            SELECT cell_id, gene_id, MAX(log_prob) AS max_log_prob
            FROM posterior
            WHERE NOT regularized
            GROUP BY cell_id, gene_id
        ),
        mean_per_pair AS (
            SELECT
                p.gene_id,
                SUM(CAST(p.c AS DOUBLE) * EXP(p.log_prob - m.max_log_prob))
                    / SUM(EXP(p.log_prob - m.max_log_prob)) AS mean_noise_cg
            FROM posterior p
            JOIN max_per_pair m
              ON p.cell_id = m.cell_id AND p.gene_id = m.gene_id
            WHERE NOT p.regularized
            GROUP BY p.cell_id, p.gene_id
        )
        SELECT gene_id, SUM(mean_noise_cg) AS total_mean_noise
        FROM mean_per_pair
        GROUP BY gene_id
    """
    conn = duckdb.connect()
    tmp_dir = str(source.parent).replace("'", "''")
    conn.execute(f"SET temp_directory='{tmp_dir}'")
    if duckdb_memory_limit is not None:
        conn.execute(f"SET memory_limit='{duckdb_memory_limit}'")
    _register_posterior(conn, source)
    df = conn.execute(query).df()
    result = np.zeros(n_genes, dtype=np.float64)
    if len(df) > 0:
        result[df["gene_id"].values.astype(np.int32)] = df["total_mean_noise"].values
    return result


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

        conn = duckdb.connect()
        tmp_dir = str(noise_log_prob_coo.parent).replace("'", "''")
        conn.execute(f"SET temp_directory='{tmp_dir}'")
        if duckdb_memory_limit is not None:
            conn.execute(f"SET memory_limit='{duckdb_memory_limit}'")
        _register_posterior(conn, noise_log_prob_coo)

        # Step 1: MAP estimate
        map_df = conn.execute("""
            SELECT
                cell_id,
                gene_id,
                CAST(argmax(c, log_prob) AS INTEGER) AS map_c
            FROM posterior
            WHERE NOT regularized
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
                WHERE NOT p.regularized
                  AND (
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
