"""Execution kernels for hybrid-format SpMM without NumPy."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Tuple

from .hybrid_format import HybridFormat


@dataclass
class ScheduleConfig:
    tile_order: Tuple[str, str]
    tile_j: int
    unroll_j: int


@dataclass
class KernelResult:
    output: List[List[float]]
    elapsed_s: float


def _zeros(rows: int, cols: int) -> List[List[float]]:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def execute_schedule(
    fmt: HybridFormat,
    rhs: List[List[float]],
    schedule: ScheduleConfig,
) -> KernelResult:
    out_dim = len(rhs[0]) if rhs else 0
    output = _zeros(fmt.n, out_dim)

    start = time.perf_counter()
    for (_, _), tile in fmt.iter_tiles(schedule.tile_order):
        for j_start in range(0, out_dim, schedule.tile_j):
            j_end = min(j_start + schedule.tile_j, out_dim)
            # dense contributions
            if tile.dense_width:
                for row_offset, row_data in enumerate(tile.dense_data):
                    out_row = output[tile.row_start + row_offset]
                    for dense_pos, value in enumerate(row_data):
                        if value == 0.0:
                            continue
                        col_idx = tile.dense_columns[dense_pos]
                        rhs_row = rhs[col_idx]
                        for jj in range(j_start, j_end):
                            out_row[jj] += value * rhs_row[jj]
            # sparse contributions
            if tile.sparse_nnz:
                for rel_row, col_idx, value in zip(
                    tile.sparse_row_offsets, tile.sparse_col_indices, tile.sparse_values
                ):
                    out_row = output[tile.row_start + rel_row]
                    rhs_row = rhs[col_idx]
                    width = j_end - j_start
                    if schedule.unroll_j > 1:
                        limit = width - (width % schedule.unroll_j)
                        for jj in range(0, limit, schedule.unroll_j):
                            for uu in range(schedule.unroll_j):
                                col = j_start + jj + uu
                                out_row[col] += value * rhs_row[col]
                        for col in range(j_start + limit, j_end):
                            out_row[col] += value * rhs_row[col]
                    else:
                        for col in range(j_start, j_end):
                            out_row[col] += value * rhs_row[col]
    elapsed = time.perf_counter() - start
    return KernelResult(output=output, elapsed_s=elapsed)
