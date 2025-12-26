# Why Alignment Matters More Than Stride for GEMM

Run a contiguity benchmark on matrix multiplication versus layer normalization and you'll see something
revealing: transposed views slow GEMM by 10–15% on modern GPUs, while a single-element offset doubles
execution time. LayerNorm shows under 10% variance for both.

The difference is library compensation versus hardware limitation. Modern GEMM libraries detect transposed
inputs with aligned base pointers and switch to optimized TN/NT kernel variants. Benchmarks on
Blackwell-class consumer GPUs show transposed GEMM matching explicitly contiguous copies—the library is
compensating, not the hardware tolerating poor layout.

Misalignment breaks this. A one-element offset forces split cache line transactions on every tile load
across thousands of tiles. Libraries cannot switch algorithms to recover this performance. The result:
consistent 2× slowdown across matrix sizes and data types.

LayerNorm reads each element 2–3 times across reduction passes—mean, variance, normalize. No inner loop
reuse. No tiling. The kernel is already memory-bound at optimal coalescing. Degrading memory bandwidth adds
constant overhead, not multiplicative slowdown. The compute units were already idle.

The arithmetic intensity gap explains resilience, but library heuristics explain why stride alone no longer
predicts GEMM performance. Modern GEMM performance is determined by whether base pointer alignment allows
vectorized tile loads, not by stride pattern.

**Takeaway**: Stride patterns are often survivable through library optimization. Alignment violations are
fundamentally hostile. For streaming elementwise ops, neither matters much—the kernel was already
memory-bound.
