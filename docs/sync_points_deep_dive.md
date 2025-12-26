# Synchronization Points: The Hidden Cost That Doubles Your Training Time

## The Asymmetry

Run a heavy matrix multiplication (8192×8192 FP16) on an RTX 5070 and it completes in 25.0 ms. Force
synchronization after every operation and it takes 25.2 ms—barely 1% slower. But shrink the operation to
512×512 and the story changes: async takes 18 µs, forced sync takes 28 µs—a 50% slowdown.

Call `.item()` to extract a scalar and you pay 26 ms when GPU work is pending, but only 0.13 ms after
explicit synchronization—a 200× difference. Over 1000 training iterations, this single mistake doubles your
total runtime from 25 seconds to 51 seconds.

This reveals the critical insight: **synchronization overhead is constant (~10–15 µs), but kernel execution
time scales with problem size**. For heavy operations, sync overhead is negligible. For small, frequent
operations—the kind that dominate real training loops—it destroys throughput.

## CUDA's Asynchronous Execution Model

When you write `y = x @ w` in PyTorch with CUDA tensors, the operation doesn't execute immediately. Instead:

1. PyTorch validates tensor shapes and dtypes on the CPU
2. PyTorch enqueues a cuBLAS GEMM kernel on the GPU's command queue
3. Python immediately returns control to the next line of code
4. The GPU processes the queued kernel asynchronously

The CPU and GPU work in parallel. While the GPU multiplies matrices, the CPU can prepare the next batch,
queue more operations, or handle I/O. This overlap is critical for performance—modern GPUs are so fast that
kernel launch overhead (5–10 μs) becomes significant for small operations.

## What Synchronization Does

Synchronization forces the CPU to wait for the GPU to finish all pending work. When you call
`torch.cuda.synchronize()`, the CPU blocks until every queued kernel completes. When you call `.item()` on a
CUDA tensor, PyTorch implicitly synchronizes, transfers the scalar to CPU memory, and returns the value.

This destroys pipeline parallelism. The CPU can't prepare the next batch while waiting. The GPU can't start
the next kernel until the CPU queues it. Both devices sit idle during the handoff.

## Measuring the Cost

A benchmark using both large and small matrix multiplications on an RTX 5070 reveals the asymmetry:

**BIG GEMM (8192×8192 FP16)**:
- Async queuing: 25.0 ms/iter
- Forced sync each iteration: 25.2 ms/iter
- Overhead: 0.24 ms (~1%)
- Operations queue without waiting; GPU and CPU work in parallel
- Sync overhead is negligible because kernel is so heavy

**SMALL GEMM (512×512 FP16)**:
- Async queuing: 18 µs/iter
- Forced sync each iteration: 28 µs/iter
- Overhead: 10 µs (~50%)
- Same sync overhead, but now it dominates execution time
- This is the real danger case in production

**`.item()` with pending GPU work**: 26 ms
- Includes full GEMM execution (~25 ms) + transfer overhead
- Forces CPU to wait for all queued kernels to complete
- Not "cheap"—it's an implicit synchronization barrier

**`.item()` after explicit sync**: 0.13 ms
- Pure device→host transfer + Python overhead
- 200× faster when no GPU work is pending
- Even this "cheap" case is expensive relative to µs-scale kernels

**`print()` with pending GPU work**: 28 ms
- Similar to `.item()` but with formatting overhead
- Slightly slower (~2 ms) due to string conversion
- Both should be avoided in tight loops

## Why Synchronization Overhead is Constant

The benchmark reveals that synchronization overhead is approximately **10–15 µs per sync point** on RTX
5070, regardless of operation size or data type.

**For the 8192×8192 FP16 GEMM**:
- Kernel execution: ~25 ms
- Synchronization overhead: ~0.24 ms (240 µs)
- Overhead percentage: ~1%

**For the 512×512 FP16 GEMM**:
- Kernel execution: ~18 µs
- Synchronization overhead: ~10 µs (same order of magnitude!)
- Overhead percentage: ~50%

**Key insight**: Synchronization overhead doesn't scale with problem size. It's the cost of CPU↔GPU
coordination: kernel launch overhead, driver calls, and pipeline stalls. A 25 ms kernel amortizes this cost.
An 18 µs kernel doesn't.

**Why this matters**: Real training loops contain many small operations—activations (ReLU, GELU),
normalizations (LayerNorm, BatchNorm), elementwise ops (add, multiply). Each forced sync adds the same
10–15 µs overhead. For a model with 100 small ops per iteration, forced synchronization adds 1–1.5 ms per
iteration—potentially 50–100% slowdown for small models.

**Pipeline bubbles compound the problem**: Even when sync overhead is small per operation, it prevents the
CPU from preparing the next batch while the GPU executes. For small models with fast kernels, this
serialization can cut throughput by 30–50%.

## Common Implicit Synchronization Points

Beyond explicit `torch.cuda.synchronize()`, several operations force synchronization:

**Scalar extraction**:
```python
loss_value = loss.item()  # Synchronizes
```

**Printing tensor values**:
```python
print(f"Loss: {loss}")  # Synchronizes if loss is on CUDA
```

**CPU conversion**:
```python
x_cpu = x.cpu()      # Synchronizes, transfers to CPU
x_np = x.numpy()     # Synchronizes, transfers to NumPy
```

**Indexing with CPU tensors**:
```python
indices = torch.tensor([0, 1, 2])  # CPU tensor
y = x[indices]  # Synchronizes to transfer indices to GPU
```

**Conditional logic on CUDA tensors**:
```python
if (x > 0).any():  # .any() returns scalar, forces sync
    ...
```

**Memory allocation failures**:
```python
# If GPU runs out of memory, PyTorch synchronizes to free cached memory
x = torch.randn(100000, 100000, device='cuda')  # May trigger sync if OOM
```

## Practical Implications

### For Training Loops

The most common mistake is logging loss every iteration:

```python
# BAD: Synchronizes 1000 times per epoch
for i, batch in enumerate(dataloader):
    loss = model(batch)  # Forward pass: ~25 ms
    loss.backward()
    optimizer.step()
    print(f"Step {i}: loss = {loss.item()}")  # Forces sync: adds ~26 ms stall!
```

If the forward pass takes 25 ms and `.item()` forces a 26 ms stall (waiting for GPU work to complete), each
iteration takes 51 ms instead of 25 ms. Over 1000 steps:
- Compute time: 25,000 ms (25 seconds)
- Sync overhead: 26,000 ms (26 seconds)
- **Total runtime doubles** from 25s to 51s

**Better approach**: Accumulate on GPU, sync once per epoch:

```python
losses = []
for i, batch in enumerate(dataloader):
    loss = model(batch)
    loss.backward()
    optimizer.step()
    losses.append(loss.detach())  # No sync, stays on GPU

# Synchronize once at the end
avg_loss = torch.stack(losses).mean().item()
print(f"Epoch loss: {avg_loss}")
```

**Even better**: Log every N steps:

```python
for i, batch in enumerate(dataloader):
    loss = model(batch)
    loss.backward()
    optimizer.step()
    if i % 100 == 0:
        print(f"Step {i}: loss = {loss.item()}")  # Sync only every 100 steps
```

### For Validation

Validation metrics often require CPU-side accumulation, but you can minimize syncs:

```python
# BAD: Synchronizes for every batch
model.eval()
correct = 0
for batch in val_loader:
    pred = model(batch)
    correct += (pred.argmax(1) == labels).sum().item()  # Sync!

# GOOD: Accumulate on GPU, sync once at end
model.eval()
correct = []
for batch in val_loader:
    pred = model(batch)
    correct.append((pred.argmax(1) == labels).sum())  # Stays on GPU

total_correct = torch.stack(correct).sum().item()  # Single sync
accuracy = total_correct / len(val_loader.dataset)
```

### For Inference

Batch processing minimizes synchronization:

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

Using `time.time()` to measure GPU operations is incorrect because it measures CPU time, not GPU time:

```python
# BAD: Measures CPU time, not GPU time
import time
t0 = time.time()
y = x @ w
t1 = time.time()
print(f"Time: {t1 - t0}")  # Wrong! Doesn't wait for GPU
```

The operation queues asynchronously, so `t1 - t0` measures only the time to enqueue the kernel, not execute
it.

**Correct approach**: Use CUDA events:

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
y = x @ w
end.record()

torch.cuda.synchronize()
print(f"Time: {start.elapsed_time(end)} ms")  # Correct GPU time
```

CUDA events are recorded on the GPU timeline, so they measure actual kernel execution time.

## Hardware-Specific Considerations

### Consumer GPUs (RTX 3090, 4090, 5070)

**RTX 5070 (measured)**:
- Big GEMM (8192×8192 FP16): 25 ms
- Small GEMM (512×512 FP16): 18 µs
- Sync overhead: ~10–15 µs per sync point
- `.item()` with pending work: ~26 ms (includes kernel execution)
- `.item()` after sync: ~0.13 ms (pure transfer)
- `print()` with pending work: ~28 ms

**Key insight**: Sync overhead is 1% for big ops, 50% for small ops. Critical for small batch sizes.

### Datacenter GPUs (A100, H100)

**Expected characteristics**:
- Big GEMM (8192×8192 FP16): 5–15 ms
- Small GEMM (512×512 FP16): 5–10 µs
- Sync overhead: ~5–10 µs per sync point
- `.item()` after sync: 0.05–0.1 ms
- Multi-stream execution hides more latency

**Key insight**: Faster kernels make sync overhead a larger percentage. Even more critical to avoid syncs.

### Older GPUs (GTX 1080, RTX 2080)

**Expected characteristics**:
- Big GEMM (8192×8192 FP16): 40–80 ms
- Small GEMM (512×512 FP16): 30–50 µs
- Sync overhead: ~10–20 µs per sync point
- `.item()` after sync: 0.2–0.5 ms

**Key insight**: Slower kernels amortize sync overhead better. Still avoid unnecessary syncs, but less
critical than on modern GPUs.

## Advanced: Multi-Stream Execution

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

This enables:
- Overlapping data transfer with computation
- Running multiple models concurrently
- Pipelining batch processing

**Example: Overlapping H2D transfer with computation**:

```python
stream_compute = torch.cuda.Stream()
stream_transfer = torch.cuda.Stream()

# Transfer batch N+1 while computing batch N
with torch.cuda.stream(stream_transfer):
    batch_next = batch_next_cpu.cuda(non_blocking=True)

with torch.cuda.stream(stream_compute):
    output = model(batch_current)
```

## Why This Matters for Modern Training

The impact of synchronization depends on kernel execution time and operation granularity:

**Small models** (ResNet-18, BERT-base) on RTX 5070:
- Many small kernels: 10–100 µs each
- Synchronization overhead: ~10–15 µs per sync point
- Forced sync after each op: 50–100% slowdown
- `.item()` per iteration: adds ~26 ms stall (doubles runtime if forward pass is ~25 ms)
- **Critical**: Sync overhead dominates for small ops, catastrophic for `.item()` in loops

**Large models** (GPT-3, LLaMA-70B) on RTX 5070:
- Heavy kernels: 10–100 ms each
- Synchronization overhead: ~10–15 µs per sync point
- Forced sync after each op: <1% slowdown
- `.item()` per iteration: still adds ~26 ms stall per call
- **Less critical for forced sync**: Sync overhead is small percentage of kernel time
- **Still critical for `.item()`**: 26 ms stall is significant even for large models

**This benchmark** (8192×8192 GEMM):
- Big kernel: 25 ms
- Small kernel: 18 µs
- Demonstrates both extremes: sync is negligible for big ops, dominant for small ops
- `.item()` with pending work: 26 ms (includes full kernel execution)
- **Key lesson**: Sync overhead is constant; impact depends on kernel size

**Real-world training loops**:
- Mix of large (matmuls) and small (activations, norms) operations
- Small ops are where forced sync hurts most
- `.item()` in loops hurts regardless of model size
- Batch your syncs: log every 100 steps, not every step

## Summary

**Key insights**:

1. **Sync overhead is constant (~10–15 µs), kernel time scales** - For big GEMM (25 ms), sync is 1%. For
   small GEMM (18 µs), sync is 50%. Synchronization matters most for small, frequent operations.

2. **`.item()` includes full kernel execution when work is pending** - With pending work: ~26 ms (includes
   25 ms kernel). After sync: ~0.13 ms (pure transfer). 200× difference. In a training loop, `.item()` per
   iteration doubles runtime from 25s to 51s over 1000 steps.

3. **Small operations are the real danger** - Activations, norms, elementwise ops take 10–100 µs. Forced
   sync adds 10–15 µs overhead—a 50–100% penalty. Real training loops have hundreds of these operations.

4. **Batch your syncs** - Log every 100 steps, not every step. Accumulate metrics on GPU. One sync at the
   end instead of 1000 syncs in the loop saves 26 seconds over 1000 iterations.

5. **Use CUDA events for profiling** - `time.time()` measures CPU time, not GPU time. CUDA events measure
   actual kernel execution time.

6. **Multi-stream execution hides latency** - Overlap data transfer with computation. Use separate streams
   for independent operations.

**Common mistakes**:
- Logging loss every iteration
- Printing tensor values in tight loops
- Using `.cpu()` or `.numpy()` unnecessarily
- Profiling with `time.time()` instead of CUDA events

**Best practices**:
- Accumulate metrics on GPU, sync once per epoch
- Use TensorBoard or async logging frameworks
- Batch inference requests
- Use CUDA events for accurate profiling
- Leverage multi-stream execution for pipelining

Modern GPU training is about keeping the pipeline full. Every synchronization point is a bubble where
expensive hardware sits idle. Eliminate unnecessary syncs and batch the necessary ones.
