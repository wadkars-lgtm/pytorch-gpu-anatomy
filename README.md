# pytorch-gpu-anatomy

A comprehensive benchmarking suite for understanding PyTorch GPU internals, tensor semantics, and performance
characteristics.

## Installation

```sh
pip install pytorch-gpu-anatomy
```

## Development

* Clone this repository
* Requirements:
  * `angreal`
* `pip install angreal && angreal setup`

This project was generated using the [angreal python template](https://github.com/angreal/python) template.

## PyTorch CUDA Setup

To use the latest PyTorch with CUDA 12.8 support:

```shell
pip uninstall -y torch torchvision torchaudio
pip cache purge
pip install -U pip
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
```

---

## Benchmarks

📖 **[Complete Benchmark Suite Overview](docs/c01_tensor_semantics.md)** - Comprehensive guide covering all
experiments with detailed parameters, examples, and troubleshooting.

---

### Experiment 1: Memory Layout and Contiguity

Understanding how PyTorch stores tensors in memory and why contiguity matters for performance.

**Quick Start**:
```shell
pytorch-gpu-anatomy c01 contiguity-bench --device cuda:0 --dtype fp16 --m 4096 --n 4096 --k 4096 --iters 200
```

**Documentation**:
- 📖 **[Main Guide](docs/c01_tensor_semantics.md#1-contiguity-benchmark)** - Complete runbook with
  parameters and examples
- 🔬 **[Deep Dive](docs/contiguity_deep_dive.md)** - Technical details on memory layout and performance
- 📝 **[Summary](docs/contiguity_summary.md)** - Quick reference and key takeaways

**Key Insight**: Non-contiguous tensors can be 2-10× slower depending on operation. Transposed views are
metadata-only but force slower kernels. Always check `.is_contiguous()` in performance-critical code.

---

### Experiment 2: CPU↔GPU Synchronization Points

Identifying and measuring the cost of synchronization between CPU and GPU operations.

**Quick Start**:
```shell
pytorch-gpu-anatomy c01 sync-points --device cuda:0 --dtype fp16 --iters 50
```

**Documentation**:
- 📖 **[Main Guide](docs/c02_sync_points.md)** - Complete runbook with parameters, examples, and interpretation
- 🔬 **[Deep Dive](docs/sync_points_deep_dive.md)** - Technical details on async execution and sync overhead
- 📝 **[Summary](docs/sync_points_summary.md)** - Quick reference and key takeaways

**Key Insight**: `.item()` with pending GPU work includes full kernel execution time—can double training
runtime over 1000 iterations. Sync overhead is constant (~10–15 µs), but impact depends on kernel size. For
big ops (25 ms), sync is 1%. For small ops (18 µs), sync is 50%.

---

### Experiment 3: Autograd Graph Introspection

Inspecting PyTorch's computation graph to understand backward pass structure and kernel fusion.

**Quick Start**:
```shell
pytorch-gpu-anatomy c01 autograd-inspect --device cuda:0 --dtype fp16 --batch 2 --seq 128 --dmodel 768 \
  --nhead 12
```

**Documentation**:
- 📖 **[Main Guide](docs/c03_autograd_inspection.md)** - Complete runbook with parameters, examples, and interpretation
- 🔬 **[Deep Dive](docs/autograd_inspection_deep_dive.md)** - Technical details on graph structure and kernel fusion
- 📝 **[Summary](docs/autograd_inspection_summary.md)** - Quick reference and key takeaways

**Key Insight**: Kernel fusion reduces attention from 20+ backward nodes to 1 node (2-5× speedup). Shape
operations dominate node count (~25 out of 50) but cost nothing. Graph size is independent of batch/sequence
length. Verify fusion by looking for `ScaledDotProductEfficientAttentionBackward0`.
