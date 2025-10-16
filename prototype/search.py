"""End-to-end search driver for Phase 0 experiments (pure Python)."""
from __future__ import annotations

import itertools
import statistics
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .data import SpMMProblem, generate_blocky_spmm
from .hybrid_format import build_hybrid_format
from .kernel import KernelResult, ScheduleConfig, execute_schedule


@dataclass
class CandidateConfig:
    tile_size_i: int
    tile_size_k: int
    dense_column_threshold: float
    min_dense_columns: int
    tile_order: Sequence[str]
    tile_j: int
    unroll_j: int


@dataclass
class CandidateEvaluation:
    config: CandidateConfig
    result: KernelResult
    gflops: float
    max_abs_error: float


def _compute_gflops(problem: SpMMProblem, elapsed_s: float) -> float:
    if elapsed_s <= 0:
        return float("inf")
    flops = 2.0 * problem.nnz * problem.k
    return flops / elapsed_s / 1e9


def _enumerate_candidates() -> Iterable[CandidateConfig]:
    tile_sizes = [8, 16]
    dense_thresholds = [0.5, 0.8]
    min_dense_columns_options = [0, 1]
    tile_orders = [("io", "ko"), ("ko", "io")]
    tile_j_options = [8, 16]
    unroll_options = [1, 2]

    for tile_i, tile_k, threshold, min_dense, order, tile_j, unroll in itertools.product(
        tile_sizes,
        tile_sizes,
        dense_thresholds,
        min_dense_columns_options,
        tile_orders,
        tile_j_options,
        unroll_options,
    ):
        yield CandidateConfig(
            tile_size_i=tile_i,
            tile_size_k=tile_k,
            dense_column_threshold=threshold,
            min_dense_columns=min_dense,
            tile_order=order,
            tile_j=tile_j,
            unroll_j=unroll,
        )


def _reference_result(problem: SpMMProblem) -> List[List[float]]:
    n = problem.n
    k = problem.k
    dense_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    for r, c, v in zip(problem.rows, problem.cols, problem.values):
        dense_matrix[r][c] = v
    result = [[0.0 for _ in range(k)] for _ in range(n)]
    for i in range(n):
        row = dense_matrix[i]
        for kk in range(n):
            a_val = row[kk]
            if a_val == 0.0:
                continue
            rhs_row = problem.dense_rhs[kk]
            out_row = result[i]
            for j in range(k):
                out_row[j] += a_val * rhs_row[j]
    return result


def _max_abs_difference(a: List[List[float]], b: List[List[float]]) -> float:
    max_val = 0.0
    for row_a, row_b in zip(a, b):
        for va, vb in zip(row_a, row_b):
            diff = abs(va - vb)
            if diff > max_val:
                max_val = diff
    return max_val


def evaluate_candidate(
    problem: SpMMProblem,
    config: CandidateConfig,
    reference: List[List[float]],
    repeats: int = 2,
) -> CandidateEvaluation:
    fmt = build_hybrid_format(
        n=problem.n,
        rows=problem.rows,
        cols=problem.cols,
        values=problem.values,
        tile_size_i=config.tile_size_i,
        tile_size_k=config.tile_size_k,
        dense_column_threshold=config.dense_column_threshold,
        min_dense_columns=config.min_dense_columns,
    )
    schedule = ScheduleConfig(
        tile_order=tuple(config.tile_order),
        tile_j=config.tile_j,
        unroll_j=config.unroll_j,
    )

    elapsed_times: List[float] = []
    result: KernelResult | None = None
    for _ in range(repeats):
        run = execute_schedule(fmt, problem.dense_rhs, schedule)
        elapsed_times.append(run.elapsed_s)
        result = run
    assert result is not None
    median_time = statistics.median(elapsed_times)
    gflops = _compute_gflops(problem, median_time)
    max_abs_error = _max_abs_difference(reference, result.output)
    return CandidateEvaluation(
        config=config,
        result=result,
        gflops=gflops,
        max_abs_error=max_abs_error,
    )


def search(problem: SpMMProblem, max_candidates: int | None = None) -> CandidateEvaluation:
    best: CandidateEvaluation | None = None
    reference = _reference_result(problem)
    for idx, candidate in enumerate(_enumerate_candidates()):
        if max_candidates is not None and idx >= max_candidates:
            break
        evaluation = evaluate_candidate(problem, candidate, reference)
        if evaluation.max_abs_error > 1e-3:
            continue
        if best is None or evaluation.gflops > best.gflops:
            best = evaluation
    if best is None:
        raise RuntimeError("No valid candidate found")
    return best


def main() -> None:
    problem = generate_blocky_spmm(
        n=96,
        k=48,
        density=0.05,
        block_size=32,
        block_density_variation=0.2,
        seed=0,
    )
    best = search(problem)
    print("Best configuration:")
    print(best.config)
    print(f"GFLOPS: {best.gflops:.3f}")
    print(f"Max abs error: {best.max_abs_error:.2e}")


if __name__ == "__main__":
    main()
