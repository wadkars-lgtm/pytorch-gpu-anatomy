# Memory Layout Penalties: Why Alignment Matters More Than Stride

## The Real Story

Transpose a matrix before feeding it to GEMM and modern libraries compensate—performance stays within 15%.
Misalign the base pointer by a single element and performance drops 2×. Apply either change to layer
normalization and you lose under 10%. This asymmetry reveals how modern GPU software stacks handle memory
layout: stride patterns are often survivable through library heuristics, but alignment violations are
fundamentally hostile to vectorized tile loads.

## Tiled GEMM and the Inner Loop Reuse Pattern

Matrix multiplication kernels partition the output into tiles—typically 32×32 or 64×64 blocks. Each tile is
computed by loading corresponding slices of the input matrices into shared memory, then executing a
triple-nested loop: outer loops over tiles, inner loop over the reduction dimension.

The critical property: each element loaded into a tile participates in dozens of multiply-accumulate
operations before being discarded. A 32×32 tile from matrix A gets combined with 32 different tiles from
matrix B. Each of those 1024 elements is read once from DRAM, placed in shared memory, then used 32–64
times in dot products.

When the input is contiguous, loading a tile row means reading 32 consecutive elements. Thirty-two threads
issue reads in lockstep, and the memory controller coalesces them into a single transaction. Loading the
full 32×32 tile requires roughly 32 transactions—one per row.

When the input is transposed, those same 32 threads now request elements separated by the original row
stride—often thousands of elements apart. The memory controller cannot coalesce these requests. Each thread
triggers a separate cache line fetch. Loading the tile now requires closer to 1024 transactions. The tile
load, previously a minor cost amortized over heavy compute, becomes the bottleneck. Compute units stall
waiting for data.

## Modern Libraries Mask Transpose Penalties

When GEMM libraries detect a transposed input with an aligned base pointer and common stride pattern, they
respond by switching to TN or NT kernel variants optimized for that layout, internally repacking into
contiguous workspace when beneficial, or treating the transpose as the logical layout rather than fighting
it.

Benchmarks on Blackwell-class consumer GPUs show transposed GEMM running at 1.02–1.14× baseline time—not
the 5–10× slowdown predicted by naive stride analysis. The library compensates. This is not hardware
tolerance; it is software adaptation.

The evidence: transposed views perform identically to explicitly contiguous copies of transposed data. The
kernel sees the same effective layout in both cases.

## Why a Single Element Offset Matters

Alignment interacts with vectorized loads. Modern memory controllers fetch data in 128-byte cache lines. A
warp reading 32 consecutive float16 values ideally pulls from two cache lines. If the starting address is
misaligned by one element, the read now spans three cache lines instead of two. The memory subsystem issues
an extra transaction, and effective bandwidth drops by 30–50%.

For GEMM, this penalty applies to every tile load across thousands of tiles. A one-element offset in the
storage does not change the algorithm, but it changes how many memory transactions the hardware must issue.
The result: consistent 2× slowdown across matrix sizes and data types.

Libraries cannot compensate for misalignment the way they compensate for stride. Misaligned base pointers
break vectorized load assumptions, disrupt shared memory tiling alignment, and force split transactions at
the hardware level. No algorithm switch can recover this performance.

## LayerNorm: Streaming Without Reuse

Layer normalization computes row-wise statistics—mean and variance—then normalizes each element. The access
pattern is fundamentally different: each row is read in two or three passes, and each element is touched a
small constant number of times.

There is no inner loop that reuses a loaded tile. There is no shared memory staging area. The kernel reads
a row, reduces it to scalars, reads it again to compute variance, reads it a third time to write normalized
outputs. Each element participates in roughly one floating-point operation per byte loaded—arithmetic
intensity near unity.

When the tensor is transposed, the kernel now reads columns instead of rows. Coalescing degrades: instead
of 32 threads reading consecutive elements, they read elements separated by the row stride. Memory
transactions increase substantially.

But the kernel was already memory-bound. At optimal coalescing, the compute units were idle waiting for the
next row to arrive from DRAM. Degrading memory bandwidth means the data arrives slower, but the compute
units were already waiting. Wall time increases by the ratio of new memory time to old total time—not by
the raw memory slowdown factor.

Additionally, modern GPUs have large L2 caches. For moderate tensor sizes, the working set of a single row
fits in L2 after the first pass. Subsequent passes hit cache, not DRAM. The effective memory slowdown is
much smaller than the raw coalescing penalty would suggest.

Benchmarks confirm this: transposed LayerNorm runs at 0.99–1.32× baseline time across matrix sizes. Even
misaligned offsets show minimal impact (1.03× slower). The kernel has no compute headroom to lose.

## Arithmetic Intensity as the Dividing Line

GEMM achieves thousands of FLOPs per byte loaded from DRAM. Each tile element is reused across many dot
products. The kernel is compute-bound when memory is fast. Performance is sensitive to memory efficiency
because memory efficiency determines whether compute units stay fed.

But modern libraries have learned to detect and compensate for common stride patterns. What they cannot
compensate for is misalignment, which breaks the hardware's ability to issue vectorized loads efficiently.

LayerNorm achieves one FLOP per byte. Each element is read a few times, reduced, and discarded. The kernel
is memory-bound even with perfect coalescing. Degrading memory bandwidth makes it more memory-bound, but it
was already bottlenecked there. The compute units were already idle. Layout penalties add constant
overhead, not multiplicative slowdown.

## Practical Implications

Alignment matters more than stride for tiled algorithms. Modern GEMM libraries detect and optimize common
transpose patterns, but a single-element offset in the base pointer can double execution time. This is not
library behavior—it is hardware limitation.

For streaming elementwise operations, neither stride nor alignment matters much. These kernels are already
memory-bound. Layout penalties increase memory time, but total runtime is dominated by memory time
regardless. The L2 cache absorbs much of the inefficiency for small-to-medium working sets.

The lesson: profile your kernels under layout variations. If performance is stable, your kernel is
memory-bound and already bottlenecked. If performance collapses on misalignment but survives transpose, you
have a tiled algorithm where the library is compensating for stride but cannot compensate for alignment. If
performance collapses on both, you are running on older libraries or non-standard layouts that defeat
heuristics.

Modern GEMM performance is no longer determined solely by tensor strides, but by whether the base pointer
alignment allows the library to preserve vectorized tile loads.
