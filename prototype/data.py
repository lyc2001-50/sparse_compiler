"""Utilities for constructing synthetic sparse matrix problems without NumPy."""
from __future__ import annotations

import dataclasses
import random
from typing import List


@dataclasses.dataclass
class SpMMProblem:
    """Container for an SpMM instance."""

    n: int
    k: int
    rows: List[int]
    cols: List[int]
    values: List[float]
    dense_rhs: List[List[float]]

    @property
    def nnz(self) -> int:
        return len(self.values)


def generate_blocky_spmm(
    n: int,
    k: int,
    density: float,
    block_size: int,
    block_density_variation: float,
    seed: int | None = None,
) -> SpMMProblem:
    """Generate an SpMM problem that exhibits block structure."""
    if not 0.0 < density < 1.0:
        raise ValueError("density must be in (0, 1)")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    rng = random.Random(seed)
    num_blocks = (n + block_size - 1) // block_size

    rows: List[int] = []
    cols: List[int] = []
    values: List[float] = []

    for bi in range(num_blocks):
        i_start = bi * block_size
        i_end = min((bi + 1) * block_size, n)
        for bj in range(num_blocks):
            j_start = bj * block_size
            j_end = min((bj + 1) * block_size, n)
            area = (i_end - i_start) * (j_end - j_start)
            block_bias = rng.uniform(-block_density_variation, block_density_variation)
            block_density = min(max(density + block_bias, 0.0), 1.0)
            nnz = sum(1 for _ in range(area) if rng.random() < block_density)
            for _ in range(nnz):
                rows.append(rng.randrange(i_start, i_end))
                cols.append(rng.randrange(j_start, j_end))
                values.append(rng.uniform(-1.0, 1.0))

    order = sorted(range(len(rows)), key=lambda idx: (rows[idx], cols[idx]))
    rows = [rows[i] for i in order]
    cols = [cols[i] for i in order]
    values = [values[i] for i in order]

    dense_rhs = [
        [rng.uniform(-1.0, 1.0) for _ in range(k)]
        for _ in range(n)
    ]

    return SpMMProblem(n=n, k=k, rows=rows, cols=cols, values=values, dense_rhs=dense_rhs)
