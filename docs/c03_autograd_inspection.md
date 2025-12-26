# Chapter 03: Autograd Graph Introspection

## Overview

PyTorch's autograd system builds a **computation graph** during the forward pass that records every
operation performed on tensors with `requires_grad=True`. This graph is a directed acyclic graph (DAG) where
nodes represent operations and edges represent data dependencies. During `loss.backward()`, PyTorch
traverses this graph in reverse to compute gradients.

Understanding this graph is critical for:
- Diagnosing performance issues in backward passes
- Verifying kernel fusion and optimization paths
- Debugging gradient flow problems
- Estimating memory usage during training

This benchmark inspects the autograd graph for a Transformer block—one of the most common building blocks in
modern deep learning.

## Why Autograd Graph Structure Matters

**Autograd records operations, not modules**:
- Each tensor operation creates a node in the graph
- High-level `nn.Module` boundaries are invisible to autograd
- The graph reflects primitive operations, not Python abstractions

**Graph complexity affects performance**:
- More nodes = more kernel launches during backward
- Fused operations (like `ScaledDotProductAttention`) reduce graph size
- Shape operations (view, transpose) add nodes but minimal compute cost

**Memory scales with graph size**:
- Each node stores metadata for backward computation
- Activation tensors are kept alive until backward completes
- Graph size is independent of batch/sequence length (but memory usage isn't)

## What This Benchmark Tests

1. **Graph structure** - Complete `.grad_fn` tree from loss to inputs
2. **Operation types** - Which backward kernels PyTorch uses
3. **Kernel fusion** - Evidence of fused operations vs separate kernels
4. **Gradient accumulation** - Where gradients flow to parameters
5. **Shape operations** - Metadata-only transformations in the graph

## The Model Under Inspection

```python
class TinyTransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, mlp_mult=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, mlp_mult * d_model)
        self.fc2 = nn.Linear(mlp_mult * d_model, d_model)

    def forward(self, x):
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        h2 = self.ln2(x)
        mlp = self.fc2(F.gelu(self.fc1(h2)))
        x = x + mlp
        return x
```

This is a standard pre-norm Transformer block:
- **LayerNorm → Attention → Residual**
- **LayerNorm → MLP → Residual**

The MLP uses GELU activation and 4× hidden dimension expansion (default for most Transformers).

## Running the Benchmark

### Basic Usage

```shell
pytorch-gpu-anatomy c01 autograd-inspect \
  --device cuda:0 \
  --dtype fp16 \
  --batch 2 \
  --seq 128 \
  --dmodel 768 \
  --nhead 12
```

### Parameters

| Parameter | Description | Default | Recommended Values |
|-----------|-------------|---------|-------------------|
| `--device` | CUDA device to use | `cuda:0` | `cuda:0`, `cuda:1`, etc. |
| `--dtype` | Data type for tensors | `fp16` | `fp16`, `bf16`, `fp32` |
| `--batch` | Batch size | `2` | `1-8` |
| `--seq` | Sequence length | `128` | `64-512` |
| `--dmodel` | Model dimension | `768` | `512-1024` |
| `--nhead` | Number of attention heads | `12` | `8-16` |

### Example Commands

**Default configuration (BERT-base-like)**:
```shell
pytorch-gpu-anatomy c01 autograd-inspect --device cuda:0 --dtype fp16 --batch 2 --seq 128 --dmodel 768 \
  --nhead 12
```

**Smaller model**:
```shell
pytorch-gpu-anatomy c01 autograd-inspect --device cuda:0 --dtype fp16 --batch 4 --seq 64 --dmodel 512 \
  --nhead 8
```

**Compare data types**:
```shell
# FP16 (fastest on modern GPUs with Tensor Cores)
pytorch-gpu-anatomy c01 autograd-inspect --device cuda:0 --dtype fp16 --batch 2 --seq 128 --dmodel 768 \
  --nhead 12

# BF16 (better numerical stability, similar speed)
pytorch-gpu-anatomy c01 autograd-inspect --device cuda:0 --dtype bf16 --batch 2 --seq 128 --dmodel 768 \
  --nhead 12

# FP32 (baseline, slower but highest precision)
pytorch-gpu-anatomy c01 autograd-inspect --device cuda:0 --dtype fp32 --batch 2 --seq 128 --dmodel 768 \
  --nhead 12
```

## Expected Output

The benchmark builds the forward pass, computes a simple loss, and then inspects the autograd graph without
running backward. This shows the graph structure PyTorch has prepared for gradient computation.

### Actual Output (RTX 5070, FP16)

```shell
pytorch-gpu-anatomy c01 autograd-inspect --device cuda:0 --dtype fp16 --batch 2 --seq 128 --dmodel 768 \
  --nhead 12

# Output
=== grad_fn tree (loss.grad_fn) ===
00: MeanBackward0
01: PowBackward0
02: ToCopyBackward0
03: AddBackward0
04: AddBackward0
05: ViewBackward0
06: AccumulateGrad
07: TransposeBackward0
08: AddmmBackward0
09: ViewBackward0
10: AccumulateGrad
11: ViewBackward0
12: TBackward0
13: AddmmBackward0
14: GeluBackward0
15: AccumulateGrad
16: AccumulateGrad
17: ViewBackward0
18: TBackward0
19: ViewBackward0
20: CloneBackward0
21: AccumulateGrad
22: AddmmBackward0
23: PermuteBackward0
24: AccumulateGrad
25: ViewBackward0
26: TBackward0
27: ScaledDotProductEfficientAttentionBackward0
28: NativeLayerNormBackward0
29: AccumulateGrad
30: ViewBackward0
31: ViewBackward0
32: ViewBackward0
33: AccumulateGrad
34: AccumulateGrad
35: TransposeBackward0
36: TransposeBackward0
37: TransposeBackward0
38: ViewBackward0
39: ViewBackward0
40: ViewBackward0
41: SelectBackward0
42: SelectBackward0
43: SelectBackward0
44: CloneBackward0
45: SqueezeBackward1
46: TransposeBackward0
47: UnsqueezeBackward0
48: ViewBackward0
49: AddBackward0

Also useful:
  - y.grad_fn: AddBackward0
  - x.grad shape: (2, 128, 768)
  - First few grad stats: 2.427502110435853e-08 1.0624093192745931e-05
```

### Interpreting the Results

**Total graph size: 50 nodes**:
- This is remarkably compact for a Transformer block
- Kernel fusion (especially in attention) keeps the graph small
- Each node represents a backward operation that will execute during `.backward()`

**Loss computation (nodes 00-02)**:
```
00: MeanBackward0      # loss = y.pow(2).mean()
01: PowBackward0       # y.pow(2)
02: ToCopyBackward0    # y.float() - dtype conversion
```
The loss head creates 3 nodes. `ToCopyBackward0` handles the FP16→FP32 conversion for numerical stability.

**Residual connections (nodes 03-04, 49)**:
```
03: AddBackward0       # Second residual: x = x + mlp
04: AddBackward0       # First residual: x = x + attn_out
49: AddBackward0       # (appears at end due to BFS traversal)
```
Each `x = x + something` creates an `AddBackward0` node. Residuals are graph junctions where gradients split
and flow to multiple paths.

**MLP path (nodes 08, 13-14)**:
```
08: AddmmBackward0     # fc2: Linear(3072 → 768)
13: AddmmBackward0     # fc1: Linear(768 → 3072)
14: GeluBackward0      # GELU activation
```
The MLP is three operations: `fc2(gelu(fc1(h2)))`. `AddmmBackward0` is the backward kernel for matrix
multiplication with bias (`X @ W^T + b`).

**Attention core (node 27)**:
```
27: ScaledDotProductEfficientAttentionBackward0
```
This single node is a **fused backward kernel** that handles:
- Q/K/V projections
- Scaling by `1/sqrt(d_k)`
- Softmax over attention scores
- Attention-weighted sum of values

Without fusion, this would be 20+ separate nodes. The "Efficient" variant uses memory-efficient attention
(Flash Attention or similar).

**LayerNorm (node 28)**:
```
28: NativeLayerNormBackward0
```
One fused kernel per LayerNorm layer. These are bandwidth-bound operations that normalize across the feature
dimension. Critical for training stability but often a performance bottleneck.

**Shape operations (nodes 05, 07, 09, 11-12, 17-19, 23, 25-26, 30-32, 35-40, 45-48)**:
```
ViewBackward0          # Reshape operations
TransposeBackward0     # Transpose dimensions
PermuteBackward0       # General permutation
SelectBackward0        # Indexing/slicing
SqueezeBackward1       # Remove dimensions
UnsqueezeBackward0     # Add dimensions
CloneBackward0         # Explicit copy
TBackward0             # Transpose (alias for TransposeBackward0)
```
These nodes represent metadata transformations—they don't move data, just change how it's interpreted.
During backward, they invert the shape transformation. For example, if forward does `x.view(B, S, H)`,
backward does `grad.view(original_shape)`.

**AccumulateGrad nodes (06, 10, 15-16, 21, 24, 29, 33-34)**:
```
06: AccumulateGrad     # Parameter gradient (likely fc2.bias)
10: AccumulateGrad     # Parameter gradient (likely fc2.weight)
15: AccumulateGrad     # Parameter gradient (likely fc1.bias)
16: AccumulateGrad     # Parameter gradient (likely fc1.weight)
21: AccumulateGrad     # Parameter gradient (attention projection)
24: AccumulateGrad     # Parameter gradient (attention projection)
29: AccumulateGrad     # Parameter gradient (LayerNorm)
33: AccumulateGrad     # Parameter gradient (attention Q/K/V)
34: AccumulateGrad     # Parameter gradient (attention Q/K/V)
```
These are **gradient sinks**—where gradients accumulate into `.grad` attributes of parameters. Each
trainable parameter has one `AccumulateGrad` node.

**Gradient statistics**:
```
x.grad shape: (2, 128, 768)
First few grad stats: 2.427502110435853e-08 1.0624093192745931e-05
```
The input gradient has the same shape as the input. The small magnitude (1e-8 to 1e-5) is typical for FP16
after a single backward pass with random initialization.

## Key Observations

### Observation 1: Kernel Fusion Dramatically Reduces Graph Size

The attention mechanism is a single node (`ScaledDotProductEfficientAttentionBackward0`) instead of 20+
separate operations. This fusion:
- Reduces kernel launch overhead
- Enables memory-efficient attention (recomputes attention scores instead of storing them)
- Improves backward pass performance by 2-5× compared to unfused implementation

**Why this matters**: If you see dozens of small backward nodes for attention (e.g., `SoftmaxBackward`,
`MulBackward`, `DivBackward`), you're not using fused attention and leaving performance on the table.

### Observation 2: Shape Operations Dominate Node Count

Out of 50 nodes, ~25 are shape operations (`ViewBackward`, `TransposeBackward`, etc.). These nodes:
- Add minimal compute cost (just metadata manipulation)
- Are necessary to invert forward transformations
- Don't indicate a performance problem

**Why this matters**: A large graph isn't necessarily slow if most nodes are shape operations. Focus on
compute-heavy nodes like `AddmmBackward`, `NativeLayerNormBackward`, and attention kernels.

### Observation 3: Graph Size is Independent of Batch/Sequence Length

The graph has 50 nodes regardless of whether you use:
- Batch size 1 or 128
- Sequence length 64 or 2048

**Why this matters**: Graph construction overhead is constant. Memory usage scales with batch/sequence
(activations), but graph metadata is negligible. Don't worry about graph size for large batches.

### Observation 4: AccumulateGrad Nodes Reveal Parameter Count

Counting `AccumulateGrad` nodes gives you the number of trainable parameter tensors:
- 2 for each Linear layer (weight + bias)
- 2 for each LayerNorm (weight + bias)
- Multiple for MultiheadAttention (Q/K/V projections + output projection)

This Transformer block has ~9 `AccumulateGrad` nodes, corresponding to:
- fc1: weight + bias (2)
- fc2: weight + bias (2)
- ln1: weight + bias (2)
- ln2: weight + bias (2)
- attn: Q/K/V + output projection weights/biases (~4-6)

### Observation 5: Residual Connections Create Graph Junctions

The two `AddBackward0` nodes for residuals (nodes 03-04) are where gradients split:
- Gradient flows backward through the MLP path
- Gradient flows backward through the attention path
- Both gradients sum at the residual connection

**Why this matters**: Residuals enable deep networks by providing gradient highways. In the autograd graph,
they appear as simple addition nodes, but they're critical for training stability.

## Common Patterns to Recognize

### Pattern 1: Linear Layer
```
AddmmBackward0 → AccumulateGrad (weight) → AccumulateGrad (bias)
```
Every `nn.Linear` creates this pattern. `Addmm` = "add matrix-matrix product" = `X @ W^T + b`.

### Pattern 2: Activation Function
```
GeluBackward0 / ReluBackward0 / SiluBackward0
```
Single node per activation. These are elementwise operations—cheap and memory-efficient.

### Pattern 3: Normalization
```
NativeLayerNormBackward0 → AccumulateGrad (weight) → AccumulateGrad (bias)
```
Fused kernel for LayerNorm. Bandwidth-bound and often a bottleneck in small models.

### Pattern 4: Fused Attention
```
ScaledDotProductEfficientAttentionBackward0 → [multiple AccumulateGrad nodes]
```
Single backward kernel for the entire attention mechanism. If you see this, you're using optimized
attention. If you see `SoftmaxBackward`, `MatmulBackward`, etc., you're not.

## Practical Implications

### For Performance Optimization

1. **Verify kernel fusion**: Look for fused kernels like `ScaledDotProductEfficientAttentionBackward0`
2. **Count compute-heavy nodes**: `AddmmBackward`, `NativeLayerNormBackward`, attention kernels
3. **Ignore shape operations**: They're cheap and unavoidable
4. **Check for unexpected nodes**: Accidental `.cpu()`, `.numpy()`, or dtype conversions add overhead

### For Memory Debugging

1. **Each node stores metadata**: ~100 bytes per node (negligible)
2. **Activations dominate memory**: Not visible in graph, but stored for backward
3. **Gradient accumulation**: Each `AccumulateGrad` allocates a `.grad` tensor
4. **Fused kernels save memory**: Memory-efficient attention recomputes instead of storing

### For Gradient Flow Debugging

1. **Trace gradient path**: Follow edges from loss to input
2. **Check for dead branches**: Nodes with no path to loss won't receive gradients
3. **Verify parameter updates**: Each trainable parameter should have an `AccumulateGrad` node
4. **Inspect gradient magnitudes**: Use the printed stats to check for vanishing/exploding gradients

## Summary

The autograd graph for a Transformer block is compact (50 nodes) and efficient:
- **Kernel fusion** reduces attention to a single backward node
- **Shape operations** dominate node count but add minimal overhead
- **Graph size** is independent of batch/sequence length
- **AccumulateGrad nodes** reveal parameter structure

Understanding this graph helps you:
- Verify that PyTorch is using optimized kernels
- Diagnose performance issues in backward passes
- Debug gradient flow problems
- Estimate memory usage during training

Being able to read autograd graphs is essential for serious deep learning systems work. This benchmark gives
you the tools to inspect and understand what PyTorch is doing under the hood.
