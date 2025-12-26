# Chapter 02: CPU↔GPU Synchronization Points

## Overview

GPU operations in PyTorch are **asynchronous by default**. When you call `y = x @ w` on CUDA tensors, PyTorch
queues the operation on the GPU and immediately returns control to Python. The CPU doesn't wait for the GPU
to finish—it continues executing the next line of code while the GPU works in parallel.

This asynchrony is critical for performance: the CPU can prepare the next batch of data, queue more
operations, or handle I/O while the GPU crunches numbers. But certain operations **force synchronization**—
they make the CPU wait for the GPU to finish all pending work. These synchronization points are performance
killers in training loops.

This benchmark identifies explicit and implicit synchronization points and measures their cost.

## Why Synchronization Matters

**Asynchronous execution** allows overlapping:
- CPU prepares batch N+1 while GPU processes batch N
- Multiple GPU kernels execute concurrently (when dependencies allow)
- Data transfers overlap with computation

**Synchronization** destroys this parallelism:
- CPU stalls waiting for GPU
- GPU sits idle waiting for CPU
- Pipeline bubbles waste compute cycles

A single `.item()` call in a training loop can cut throughput by 30–50% on small models where kernel launch
overhead dominates.

## What This Benchmark Tests

1. **Async operation queuing** - Baseline performance with no forced synchronization
2. **Explicit synchronization** - Cost of `torch.cuda.synchronize()` after every operation
3. **Implicit sync from `.item()`** - Transferring a single scalar to CPU
4. **Implicit sync from `print()`** - Printing a CUDA tensor element
5. **Other implicit syncs** - `.cpu()`, `.numpy()`, indexing with CPU tensors

## Running the Benchmark

### Basic Usage

```shell
pytorch-gpu-anatomy c01 sync-points \
  --device cuda:0 \
  --dtype fp16 \
  --iters 50
```

### Parameters

| Parameter | Description | Default | Recommended Values |
|-----------|-------------|---------|-------------------|
| `--device` | CUDA device to use | `cuda:0` | `cuda:0`, `cuda:1`, etc. |
| `--dtype` | Data type for tensors | `fp16` | `fp16`, `bf16`, `fp32` |
| `--iters` | Number of timing iterations | `50` | `20-100` |

### Example Commands

**Quick test (20 iterations)**:
```shell
pytorch-gpu-anatomy c01 sync-points --device cuda:0 --dtype fp16 --iters 20
```

**Thorough benchmark (100 iterations)**:
```shell
pytorch-gpu-anatomy c01 sync-points --device cuda:0 --dtype fp32 --iters 100
```

**Compare data types**:
```shell
# FP16 (fastest on modern GPUs with Tensor Cores)
pytorch-gpu-anatomy c01 sync-points --device cuda:0 --dtype fp16 --iters 50

# BF16 (better numerical stability, similar speed)
pytorch-gpu-anatomy c01 sync-points --device cuda:0 --dtype bf16 --iters 50

# FP32 (baseline, slower but highest precision)
pytorch-gpu-anatomy c01 sync-points --device cuda:0 --dtype fp32 --iters 50
```

## Expected Output

The benchmark performs a heavy matrix multiplication (8192×8192 @ 8192×8192) and measures timing under
different synchronization scenarios.

### Actual Output (RTX 5070, FP16)

```shell
# FP16 (fastest on modern GPUs with Tensor Cores)
pytorch-gpu-anatomy c01 sync-points --device cuda:0 --dtype fp16 --iters 50

# Output
=== BIG GEMM (8192x8192 @ 8192x8192) ===
Async (no per-iter sync): 25.0359 ms
Forced sync each iter:    25.2747 ms
Delta:                   0.2389 ms  (238.9 µs)  (0.95%)

=== SMALL GEMM (512x512 @ 512x512) ===
Async (no per-iter sync): 0.0174 ms  (17.4 µs)
Forced sync each iter:    0.0318 ms  (31.8 µs)
Delta:                   0.0144 ms  (14.4 µs)  (83.26%)

=== Implicit sync demos (stall when work is pending) ===
.item() (includes stall): 26.908 ms  (value=135.12500)
print triggers: 135.125
print (includes stall):   28.499 ms

=== Scalar read after explicit synchronize (no pending GPU work) ===
.item() (after sync):     0.131 ms  (value=135.12500)

Takeaway:
  - CUDA ops are queued asynchronously.
  - Per-iter torch.cuda.synchronize() hurts most when kernels are small/fast.
  - item()/print/cpu()/numpy() force a sync if they touch results from pending GPU work.```

``` shell
# BF16 (better numerical stability, similar speed)
pytorch-gpu-anatomy c01 sync-points --device cuda:0 --dtype bf16 --iters 50

# Output

=== BIG GEMM (8192x8192 @ 8192x8192) ===
Async (no per-iter sync): 21.4733 ms
Forced sync each iter:    21.4827 ms
Delta:                   0.0094 ms  (9.4 µs)  (0.04%)

=== SMALL GEMM (512x512 @ 512x512) ===
Async (no per-iter sync): 0.0213 ms  (21.3 µs)
Forced sync each iter:    0.0353 ms  (35.3 µs)
Delta:                   0.0140 ms  (14.0 µs)  (66.01%)

=== Implicit sync demos (stall when work is pending) ===
.item() (includes stall): 18.719 ms  (value=134.00000)
print triggers: 134.0
print (includes stall):   20.658 ms

=== Scalar read after explicit synchronize (no pending GPU work) ===
.item() (after sync):     0.103 ms  (value=134.00000)

Takeaway:
  - CUDA ops are queued asynchronously.
  - Per-iter torch.cuda.synchronize() hurts most when kernels are small/fast.
  - item()/print/cpu()/numpy() force a sync if they touch results from pending GPU work.
```

```shell
# FP32 (baseline, slower but highest precision)
pytorch-gpu-anatomy c01 sync-points --device cuda:0 --dtype fp32 --iters 50

# Output

=== BIG GEMM (8192x8192 @ 8192x8192) ===
Async (no per-iter sync): 72.4066 ms
Forced sync each iter:    72.2672 ms
Delta:                   -0.1394 ms  (-139.4 µs)  (-0.19%)

=== SMALL GEMM (512x512 @ 512x512) ===
Async (no per-iter sync): 0.0410 ms  (41.0 µs)
Forced sync each iter:    0.0556 ms  (55.6 µs)
Delta:                   0.0147 ms  (14.7 µs)  (35.84%)

=== Implicit sync demos (stall when work is pending) ===
.item() (includes stall): 71.855 ms  (value=141.34351)
print triggers: 141.343505859375
print (includes stall):   75.765 ms

=== Scalar read after explicit synchronize (no pending GPU work) ===
.item() (after sync):     0.169 ms  (value=141.34351)

Takeaway:
  - CUDA ops are queued asynchronously.
  - Per-iter torch.cuda.synchronize() hurts most when kernels are small/fast.
  - item()/print/cpu()/numpy() force a sync if they touch results from pending GPU work.
```
### Interpreting the Results

**Async timing (BIG GEMM, FP16: ~25.0 ms/iter)**:
- GPU work is enqueued asynchronously on the CUDA stream
- The CPU continues execution without waiting for kernel completion
- Timing reflects steady-state GPU throughput (measured with CUDA events + final sync)
- This is the **desired execution mode** for high-throughput workloads

**Forced sync timing (BIG GEMM, FP16: ~25.2 ms/iter, +0.98%)**:
- `torch.cuda.synchronize()` called after every operation
- The CPU blocks until the GPU finishes before queuing the next kernel
- The slowdown is minimal because the kernel is extremely heavy (~25 ms)
- Key point: synchronization overhead is amortized when compute dominates

----
**Forced sync timing (SMALL GEMM, FP16: +52% slowdown)**

- When kernels are short (~18 µs), per-iteration synchronization becomes dominant
- `torch.cuda.synchronize()` prevents queueing and eliminates overlap
-Throughput collapses even though the actual GPU work is tiny
- This is the real danger case in production systems

----

**`.item()` sync cost (FP16: ~26 ms, includes stall)**:
- `.item()` requests a CPU scalar from a GPU tensor
- If the producing kernel is still running, `.item()` blocks until completion
- The measured time closely matches the full GEMM runtime
- `.item()` is an implicit synchronization barrier

`.item()` **after explicit sync (~0.13 ms)**
- Once GPU work is complete, .item() measures only device→host transfer and Python overhead
- Even this “cheap” scalar read is expensive relative to microsecond-scale kernels

----

**`print()` sync cost (FP16: ~28 ms, includes stall)**:
- Printing a CUDA tensor element triggers device-to-host transfer
- Like `.item()`, it forces synchronization if GPU work is pending
- Slightly slower than `.item()` due to formatting overhead

----
## Key Observations

### Observation 1: Synchronization Overhead Is Constant; Its Impact Depends on Kernel Size

The benchmark exposes a clear asymmetry between large and small GPU kernels.

**BIG GEMM (8192×8192, FP16)**
- Async: ~25.0 ms
- Forced sync: ~25.2 ms
- Overhead: ~0.24 ms (~1%)

**SMALL GEMM (512×512, FP16)**
- Async: ~18 µs
- Forced sync: ~28 µs
- Overhead: ~9–14 µs (≈50–80%)

**Key insight**:
Synchronization overhead on this RTX 5070 system is approximately **constant (~10–15 µs)** per synchronization point.
For large kernels, this cost is negligible. For small kernels, it dominates execution time.

**Why this matters**:
Real training and inference loops contain many small kernels (activations, normalizations, elementwise ops).
A forced synchronization after each such operation can reduce throughput by **50–100%**, even though the GPU work itself is tiny.

---

### Observation 2: `.item()` Includes Full Kernel Runtime When GPU Work Is Pending

The cost of `.item()` depends entirely on whether the GPU has pending work.

**`.item()` with pending GPU work (FP16)**
- Time: ~26 ms
- Includes: full GEMM execution (~25 ms) + scalar transfer overhead

**`.item()` after explicit synchronization**
- Time: ~0.13 ms
- Pure device→host transfer + Python overhead

**Key insight**:
`.item()` is not “cheap.”
If GPU work is pending, it **forces the CPU to wait for all queued kernels to complete**.

In a training loop, calling `.item()` once per iteration effectively serializes CPU and GPU execution.

---

### Observation 3: `print()` Is an Implicit Synchronization Point

**`print()` with pending GPU work (FP16)**
- Time: ~28 ms

This includes:
- Full kernel completion
- Device→host transfer
- String formatting overhead

**Key insight**:
Both `.item()` and `print()` are implicit synchronization barriers.
`print()` is slightly slower due to formatting, but the dominant cost is the forced stall.

---

### Observation 4: Synchronization Cost Accumulates Linearly Over Iterations

A common pattern in training code:

```python
for batch in dataloader:
    loss = model(batch)
    print(loss.item())  # implicit sync
```

If:
- forward pass ≈ 25 ms
- `.item()` stall ≈ 26 ms

Then per iteration:
- Compute: 25 ms
- Synchronization stall: 26 ms

Over 1,000 iterations:
- Compute time: ~25 s
- Synchronization time: ~26 s

**Total runtime nearly doubles** due to unnecessary synchronization.

**Better pattern**:

```python
losses = []
for batch in dataloader:
    loss = model(batch)
    losses.append(loss.detach())  # stays on GPU

print(torch.stack(losses).mean().item())  # single sync
```

---

### Observation 5: Data Type Changes Kernel Time, Not Sync Overhead

Across dtypes:

- FP16, BF16, FP32 show **similar synchronization overhead (~10–15 µs)**
- Kernel execution time varies significantly:
  - FP16 / BF16: ~20–25 ms
  - FP32: ~70 ms

**Key insight**:
Synchronization overhead is largely **dtype-independent**.
Faster kernels simply make the same fixed overhead more visible as a percentage of total time.

---

### Observation 6: Other Common Implicit Synchronization Points

Beyond `.item()` and `print()`, synchronization also occurs with:

```python
x.cpu()
x.numpy()
```

```python
if (x > 0).any():  # scalar result → sync
    ...
```

```python
indices = torch.tensor([0, 1, 2])  # CPU tensor
y = x[indices]  # sync to transfer indices
```

Any operation that requires a **CPU-visible value** from a CUDA tensor can force synchronization.

---

### For Inference

**Batch processing**:
```python
# BAD: Synchronize for every sample
results = []
for sample in samples:
    pred = model(sample.cuda())
    results.append(pred.cpu())  # Sync!

# GOOD: Batch samples, sync once
batch = torch.stack(samples).cuda()
preds = model(batch)
results = preds.cpu()  # Single sync
```

### For Profiling

When profiling, synchronization can hide true GPU utilization:
```python
# BAD: Measures CPU time, not GPU time
import time
t0 = time.time()
y = x @ w
t1 = time.time()
print(f"Time: {t1 - t0}")  # Wrong! Doesn't wait for GPU

# GOOD: Use CUDA events
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
y = x @ w
end.record()

torch.cuda.synchronize()
print(f"Time: {start.elapsed_time(end)} ms")  # Correct GPU time
```

### Benchmark Implementation Details

The benchmark uses a large matrix multiplication (8192×8192) to create a heavy GPU workload:

```python
x = torch.randn(8192, 8192, device='cuda', dtype=torch.float16)
w = torch.randn(8192, 8192, device='cuda', dtype=torch.float16)

def heavy_op():
    y = x @ w
    return y
```

**Why 8192×8192**:
- Large enough to saturate GPU (several milliseconds per operation)
- Small enough to fit in GPU memory on most devices
- Makes synchronization overhead clearly visible

**Timing methodology**:
- Uses `torch.cuda.Event` for accurate GPU timing
- Warmup iterations to stabilize GPU clocks
- Multiple iterations to reduce measurement noise

### Hardware-Specific Considerations

#### Consumer GPUs (RTX 3090, 4090, 5070)

- Fast kernel execution (0.5–2 ms for 8192×8192 FP16 GEMM)
- Synchronization overhead is 5–20% of kernel time
- Async queuing critical for small batch sizes

#### Datacenter GPUs (A100, H100)

- Even faster kernels (0.2–0.8 ms for same operation)
- Synchronization overhead becomes 20–50% of kernel time
- Multi-stream execution hides more latency

#### Older GPUs (GTX 1080, RTX 2080)

- Slower kernels (2–5 ms)
- Synchronization overhead is smaller percentage
- Still avoid unnecessary syncs, but less critical

### Advanced: Multi-Stream Execution

CUDA streams allow concurrent kernel execution. Synchronization on one stream doesn't block others:

```python
stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()

with torch.cuda.stream(stream1):
    y1 = x1 @ w1

with torch.cuda.stream(stream2):
    y2 = x2 @ w2  # Can execute concurrently with stream1

# Synchronize both streams
torch.cuda.synchronize()
```
This is useful for:
- Overlapping data transfer with computation
- Running multiple models concurrently
- Pipelining batch processing

## Summary

**What the benchmark demonstrates**:

- CUDA kernels are launched **asynchronously**
- Synchronization cost is small but **constant**
- Small kernels suffer the most from forced sync
- `.item()` and `print()` are **full execution barriers**, not cheap reads

**Practical guidance**:
- Avoid per-iteration `.item()` and `print()`
- Accumulate metrics on GPU and synchronize sparingly
- Expect sync overhead to dominate microsecond-scale kernels
- Use CUDA events, not wall-clock timers, for GPU profiling

This is not a micro-optimization—it is a **fundamental execution model constraint**.


Modern GPU training is all about keeping the pipeline full. Every synchronization point is a bubble in that
pipeline—a moment where expensive hardware sits idle. Eliminate unnecessary syncs and batch the necessary
ones.
