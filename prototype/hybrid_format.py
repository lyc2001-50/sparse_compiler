"""Hybrid sparse/dense block representation without NumPy."""
from __future__ import annotations

import dataclasses
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Dict, List, Tuple


@dataclasses.dataclass
class HybridTile:
    """Representation of a single hybrid tile."""

    row_start: int
    row_end: int
    col_start: int
    col_end: int
    dense_columns: List[int]
    dense_data: List[List[float]]
    sparse_row_offsets: List[int]
    sparse_col_indices: List[int]
    sparse_values: List[float]

    @property
    def rows_in_tile(self) -> int:
        return self.row_end - self.row_start

    @property
    def dense_width(self) -> int:
        return len(self.dense_columns)

    @property
    def sparse_nnz(self) -> int:
        return len(self.sparse_values)


class HybridFormat:
    """Hybrid block sparse/dense container."""

    def __init__(self, tiles: Dict[Tuple[int, int], HybridTile], tile_shape: Tuple[int, int], n: int):
        self.tiles = tiles
        self.tile_shape = tile_shape
        self.n = n

    def iter_tiles(self, order: Sequence[str]) -> Iterable[Tuple[Tuple[int, int], HybridTile]]:
        assert set(order) == {"io", "ko"}
        items = list(self.tiles.items())
        if tuple(order) == ("io", "ko"):
            key_fn = lambda item: item[0]
        elif tuple(order) == ("ko", "io"):
            key_fn = lambda item: (item[0][1], item[0][0])
        else:
            raise ValueError(f"Unsupported order: {order}")
        for coord, tile in sorted(items, key=key_fn):
            yield coord, tile


def build_hybrid_format(
    n: int,
    rows: List[int],
    cols: List[int],
    values: List[float],
    tile_size_i: int,
    tile_size_k: int,
    dense_column_threshold: float,
    min_dense_columns: int,
) -> HybridFormat:
    if len(rows) != len(cols) or len(rows) != len(values):
        raise ValueError("rows, cols and values must have the same length")
    if not 0.0 <= dense_column_threshold <= 1.0:
        raise ValueError("dense_column_threshold must be in [0, 1]")
    tile_rows = max(1, int(tile_size_i))
    tile_cols = max(1, int(tile_size_k))
    min_dense_columns = max(0, int(min_dense_columns))

    tiles: Dict[Tuple[int, int], HybridTile] = {}
    tile_entries: Dict[Tuple[int, int], List[Tuple[int, int, float]]] = defaultdict(list)

    for r, c, v in zip(rows, cols, values):
        ti = r // tile_rows
        tk = c // tile_cols
        tile_entries[(ti, tk)].append((r, c, v))

    num_tiles_i = (n + tile_rows - 1) // tile_rows
    num_tiles_k = (n + tile_cols - 1) // tile_cols

    for ti in range(num_tiles_i):
        row_start = ti * tile_rows
        row_end = min((ti + 1) * tile_rows, n)
        for tk in range(num_tiles_k):
            col_start = tk * tile_cols
            col_end = min((tk + 1) * tile_cols, n)
            raw_entries = tile_entries.get((ti, tk))
            if not raw_entries:
                continue
            entry_map: Dict[Tuple[int, int], float] = {}
            for r, c, v in raw_entries:
                entry_map[(r, c)] = v
            entries = [(r, c, entry_map[(r, c)]) for (r, c) in entry_map]
            rows_in_tile = row_end - row_start
            cols_in_tile = col_end - col_start

            column_counts = [0] * cols_in_tile
            for _, c, _ in entries:
                column_counts[c - col_start] += 1
            column_density = [count / rows_in_tile for count in column_counts]
            dense_column_indices = [idx for idx, dens in enumerate(column_density) if dens >= dense_column_threshold]
            if len(dense_column_indices) < min_dense_columns and column_density:
                sorted_indices = sorted(range(len(column_density)), key=lambda i: column_density[i], reverse=True)
                dense_column_indices = sorted(sorted_indices[: min_dense_columns])
            dense_columns = [col_start + idx for idx in dense_column_indices]

            dense_data = [
                [0.0 for _ in range(len(dense_columns))]
                for _ in range(rows_in_tile)
            ]
            sparse_rows: List[int] = []
            sparse_cols: List[int] = []
            sparse_vals: List[float] = []

            dense_map = {col: pos for pos, col in enumerate(dense_columns)}

            for r, c, v in entries:
                rel_r = r - row_start
                if c in dense_map:
                    dense_data[rel_r][dense_map[c]] = v
                else:
                    sparse_rows.append(rel_r)
                    sparse_cols.append(c)
                    sparse_vals.append(v)

            tiles[(ti, tk)] = HybridTile(
                row_start=row_start,
                row_end=row_end,
                col_start=col_start,
                col_end=col_end,
                dense_columns=dense_columns,
                dense_data=dense_data,
                sparse_row_offsets=sparse_rows,
                sparse_col_indices=sparse_cols,
                sparse_values=sparse_vals,
            )

    return HybridFormat(tiles=tiles, tile_shape=(tile_rows, tile_cols), n=n)
