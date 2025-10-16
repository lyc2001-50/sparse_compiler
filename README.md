# sparse_compiler

We want to make a sparse compiler that jointly searches data formats and schedules on CPUs and GPUs. WACO performs a unified search, but its formats inherit from TACO (compressed vs. uncompressed per dimension). Hybrid formats such as those used in Adaptive Sparse Tiling (ASP-T) have become widely adopted, so supporting richer combinations can further improve performance. SparseTIR provides a hybrid data format, yet the format and scheduling choices are still handcrafted. Our goal is to combine these directions.

## Prototype search (Phase 0)

The `prototype` package contains a pure-Python search driver that explores hybrid sparse formats and loop schedules for an SpMM kernel (`C = A @ B`). The current experiment targets a matrix `A` generated with block-structured sparsity and a dense right-hand side `B`.

Run the search with:

```bash
python -m prototype.search
```

The script enumerates a lightweight search space, validates correctness against a dense reference implementation, and reports the highest-performing configuration within the sampled candidates.
