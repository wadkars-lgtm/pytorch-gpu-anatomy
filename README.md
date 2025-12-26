# pytorch-gpu-anatomy

A comprehensive benchmarking suite for understanding PyTorch GPU internals, tensor semantics, and performance characteristics.

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

### Chapter 01: Tensor Semantics

Comprehensive benchmarks for understanding PyTorch tensor storage, memory layout, contiguity, synchronization points, and autograd graph structure.

**Quick Start**:
```shell
# Contiguity benchmark
pytorch-gpu-anatomy c01 contiguity-bench --device cuda:0 --dtype fp16 --m 4096 --n 4096 --k 4096 --iters 200

# Sync points benchmark
pytorch-gpu-anatomy c01 sync-points --device cuda:0 --dtype fp16 --iters 50

# Autograd inspection
pytorch-gpu-anatomy c01 autograd-inspect --device cuda:0 --dtype fp16 --batch 2 --seq 128 --dmodel 768 --nhead 12
```

**📖 [Full Documentation](docs/c01_tensor_semantics.md)** - Complete runbook with detailed parameters, examples, troubleshooting, and interpretation guides.
