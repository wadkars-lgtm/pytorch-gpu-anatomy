# Autograd Graph Introspection: Reading PyTorch's Hidden Computation Graph

## The Invisible Graph

Build a Transformer block in PyTorch—LayerNorm, MultiheadAttention, GELU, residual connections—and run a
forward pass. Behind the scenes, PyTorch constructs a computation graph with 50 nodes representing every
operation needed for backward propagation. But here's the remarkable part: the attention mechanism, which
could be 20+ separate operations (Q/K/V projections, scaling, softmax, weighted sum), appears as a single
fused kernel. Half the nodes are shape operations that cost nothing. The graph size is the same whether you
use batch size 1 or 128.

This reveals how PyTorch's autograd system works: it records primitive operations, not Python modules. It
aggressively fuses kernels to reduce overhead. It tracks metadata transformations separately from compute.
Understanding this graph is the difference between "my model trains" and "I know exactly what my model is
doing during backward passes."

## What the Graph Reveals

Inspect the autograd graph for a Transformer block (batch=2, seq=128, d_model=768, nhead=12) and you see:

**Total nodes: 50**
- Remarkably compact for a complex architecture
- Kernel fusion reduces what could be 100+ nodes
- Each node = one backward kernel launch

**Loss computation: 3 nodes**
- `MeanBackward0` - average over all elements
- `PowBackward0` - square each element
- `ToCopyBackward0` - FP16→FP32 conversion for numerical stability

**Residual connections: 3 nodes**
- `AddBackward0` for `x = x + attn_out`
- `AddBackward0` for `x = x + mlp`
- Graph junctions where gradients split and recombine

**MLP path: 3 nodes**
- `AddmmBackward0` - fc1: Linear(768 → 3072)
- `GeluBackward0` - GELU activation
- `AddmmBackward0` - fc2: Linear(3072 → 768)

**Attention core: 1 node**
- `ScaledDotProductEfficientAttentionBackward0`
- Fused kernel covering Q/K/V projections, scaling, softmax, weighted sum
- Without fusion: 20+ separate nodes
- "Efficient" = memory-efficient attention (Flash Attention or similar)

**LayerNorm: 1 node per layer**
- `NativeLayerNormBackward0`
- Fused kernel for normalization across feature dimension
- Bandwidth-bound, often a performance bottleneck

**Shape operations: ~25 nodes**
- `ViewBackward0`, `TransposeBackward0`, `PermuteBackward0`, etc.
- Metadata transformations—no data movement
- Necessary to invert forward shape changes
- Cheap but numerous

**AccumulateGrad: 9 nodes**
- Gradient sinks for trainable parameters
- fc1 weight + bias (2)
- fc2 weight + bias (2)
- ln1 weight + bias (2)
- ln2 weight + bias (2)
- attn Q/K/V + output projection (~4-6)

## Why Kernel Fusion Matters

The attention mechanism demonstrates the power of kernel fusion. Without fusion, computing attention
requires:

1. Q/K/V projections: 3 matrix multiplications (3 `AddmmBackward0` nodes)
2. Reshape for multi-head: 3 view operations (3 `ViewBackward0` nodes)
3. Transpose for attention: 3 transposes (3 `TransposeBackward0` nodes)
4. Q @ K^T: 1 batch matrix multiply (`BmmBackward0`)
5. Scale by 1/sqrt(d_k): 1 multiply (`MulBackward0`)
6. Softmax: 1 softmax (`SoftmaxBackward0`)
7. Attention @ V: 1 batch matrix multiply (`BmmBackward0`)
8. Transpose back: 1 transpose (`TransposeBackward0`)
9. Reshape: 1 view (`ViewBackward0`)
10. Output projection: 1 matrix multiply (`AddmmBackward0`)

**Total: 20+ nodes** for unfused attention.

With `ScaledDotProductEfficientAttentionBackward0`: **1 node**.

This fusion:
- Reduces kernel launch overhead (20 launches → 1 launch)
- Enables memory-efficient attention (recomputes attention scores instead of storing them)
- Improves backward pass performance by 2-5×
- Reduces memory usage by not storing intermediate activations

**How to detect unfused attention**: If you see `SoftmaxBackward`, `MulBackward`, `DivBackward`, `BmmBackward`
in your graph, you're not using fused attention. Switch to `F.scaled_dot_product_attention()` or
`nn.MultiheadAttention` with PyTorch 2.0+.

## Shape Operations: Numerous but Cheap

Out of 50 nodes, ~25 are shape operations:
- `ViewBackward0` (8 instances)
- `TransposeBackward0` (4 instances)
- `SelectBackward0` (3 instances)
- `PermuteBackward0`, `SqueezeBackward1`, `UnsqueezeBackward0`, `CloneBackward0`, `TBackward0`

These nodes:
- **Don't move data**: They change tensor metadata (shape, stride) but not the underlying storage
- **Are necessary**: Backward must invert every forward transformation
- **Cost almost nothing**: A few CPU cycles to update metadata
- **Don't indicate a problem**: A graph with 100 nodes where 80 are shape ops is fine

**Why they exist**: If forward does `x.view(B, S, H)`, backward must do `grad.view(original_shape)` to
restore the gradient's shape. PyTorch tracks this transformation as a `ViewBackward0` node.

**When to worry**: If you see hundreds of shape operations in a simple model, you might have accidental
reshaping in loops. But for Transformers, 20-30 shape ops is normal.

## Graph Size vs Memory Usage

**Graph size is independent of batch/sequence length**:
- Batch size 1: 50 nodes
- Batch size 128: 50 nodes
- Sequence length 64: 50 nodes
- Sequence length 2048: 50 nodes

**Why**: The graph records operations, not data. `x @ w` creates one `AddmmBackward0` node whether `x` is
(1, 768) or (128, 2048, 768).

**Memory usage scales with batch/sequence**:
- Each node stores ~100 bytes of metadata (negligible)
- Activations stored for backward scale with batch × sequence × hidden_dim
- For batch=2, seq=128, d_model=768: ~200 KB of graph metadata, ~50 MB of activations

**Practical implication**: Don't worry about graph size when increasing batch size. Worry about activation
memory.

## AccumulateGrad: Where Gradients Land

Every trainable parameter has an `AccumulateGrad` node—the sink where gradients accumulate during backward.
Counting these nodes reveals parameter structure:

**This Transformer block has 9 AccumulateGrad nodes**:
- fc1.weight, fc1.bias (2)
- fc2.weight, fc2.bias (2)
- ln1.weight, ln1.bias (2)
- ln2.weight, ln2.bias (2)
- attn Q/K/V projections + output projection (~4-6)

**Why this matters**:
- Each `AccumulateGrad` allocates a `.grad` tensor matching the parameter shape
- Missing `AccumulateGrad` = parameter not receiving gradients (frozen or detached)
- Extra `AccumulateGrad` = unexpected trainable parameter

**Debugging gradient flow**: If a parameter isn't updating during training, check if it has an
`AccumulateGrad` node in the graph. If not, trace backward from the loss—there's a break in the gradient
path (likely a `.detach()`, `.data`, or operation that doesn't support gradients).

## Residual Connections: Gradient Highways

The two `AddBackward0` nodes for residuals (nodes 03-04) are critical for training deep networks:

```
x = x + attn_out  # AddBackward0 at node 04
x = x + mlp       # AddBackward0 at node 03
```

During backward:
- Gradient flows from node 03 to both the MLP path and node 04
- Gradient flows from node 04 to both the attention path and the input
- Gradients sum at each residual connection

**Why residuals enable deep networks**:
- Provide gradient highways: gradients can flow directly from output to input
- Prevent vanishing gradients: even if MLP/attention gradients are small, residual gradient remains
- Create graph junctions: gradients split and recombine, distributing information

**In the autograd graph**: Residuals appear as simple `AddBackward0` nodes, but they're architecturally
critical. Without them, gradients would have to flow through every layer sequentially, leading to vanishing
gradients in deep networks.

## Common Patterns to Recognize

### Pattern 1: Linear Layer
```
AddmmBackward0 → AccumulateGrad (weight) → AccumulateGrad (bias)
```
Every `nn.Linear(in_features, out_features)` creates this pattern. `Addmm` = "add matrix-matrix product" =
`X @ W^T + b`. The two `AccumulateGrad` nodes are for weight and bias gradients.

### Pattern 2: Activation Function
```
GeluBackward0 / ReluBackward0 / SiluBackward0
```
Single node per activation. These are elementwise operations—cheap, memory-efficient, and fast. No
parameters, so no `AccumulateGrad` nodes.

### Pattern 3: Normalization
```
NativeLayerNormBackward0 → AccumulateGrad (weight) → AccumulateGrad (bias)
```
Fused kernel for LayerNorm. Bandwidth-bound (reads entire tensor, computes mean/variance, normalizes).
Often a bottleneck in small models where compute is fast but memory bandwidth is limited.

### Pattern 4: Fused Attention
```
ScaledDotProductEfficientAttentionBackward0 → [multiple AccumulateGrad nodes]
```
Single backward kernel for the entire attention mechanism. If you see this, you're using optimized
attention. If you see `SoftmaxBackward`, `MatmulBackward`, `DivBackward`, you're not—and you're leaving
2-5× performance on the table.

## Practical Applications

### Performance Optimization

**Verify kernel fusion**:
- Look for `ScaledDotProductEfficientAttentionBackward0` (fused attention)
- Look for `NativeLayerNormBackward0` (fused LayerNorm)
- Absence of these = unfused operations = slower backward pass

**Count compute-heavy nodes**:
- `AddmmBackward0` (matrix multiply): O(n³) compute
- `NativeLayerNormBackward0` (normalization): O(n) but bandwidth-bound
- Attention kernels: O(n²) for sequence length n
- Shape operations: O(1), ignore for performance analysis

**Check for unexpected nodes**:
- `ToCopyBackward0` in tight loops = unnecessary dtype conversions
- `CpuBackward0` = accidental CPU operations in CUDA graph
- Dozens of small ops where one fused kernel should exist = missed optimization

### Memory Debugging

**Graph metadata is negligible**:
- 50 nodes × 100 bytes/node = 5 KB
- Even 1000 nodes = 100 KB
- Don't worry about graph size for memory

**Activations dominate memory**:
- Not visible in graph, but stored for backward
- Each intermediate tensor kept alive until backward completes
- For batch=2, seq=128, d_model=768: ~50 MB of activations
- Gradient checkpointing trades compute for memory by recomputing activations

**Fused kernels save memory**:
- Memory-efficient attention recomputes attention scores instead of storing them
- Saves O(batch × nhead × seq²) memory
- For seq=2048: saves ~32 MB per layer

### Gradient Flow Debugging

**Trace gradient path**:
- Start from loss (`MeanBackward0`)
- Follow edges backward to input
- Every parameter should have a path from loss to its `AccumulateGrad` node

**Check for dead branches**:
- Nodes with no path to loss won't receive gradients
- Common causes: `.detach()`, `.data`, operations that don't support gradients
- Symptom: parameter `.grad` is None after backward

**Verify parameter updates**:
- Each trainable parameter should have an `AccumulateGrad` node
- Count `AccumulateGrad` nodes = count parameter tensors
- Missing node = parameter not in graph = won't update

**Inspect gradient magnitudes**:
- Use printed stats: `First few grad stats: 2.4e-08 1.1e-05`
- Very small (< 1e-10): potential vanishing gradients
- Very large (> 1e2): potential exploding gradients
- Check at different layers to diagnose gradient flow issues

## Summary

**Key insights**:

1. **Kernel fusion is critical** - Attention is 1 node instead of 20+. Fused kernels reduce overhead by
   2-5× and enable memory-efficient implementations.

2. **Shape operations dominate node count but not cost** - 25 out of 50 nodes are shape ops. They're
   metadata transformations that cost almost nothing. Don't worry about them.

3. **Graph size is independent of batch/sequence length** - 50 nodes whether batch=1 or batch=128. Graph
   metadata is negligible (~5 KB). Activations dominate memory (~50 MB).

4. **AccumulateGrad nodes reveal parameter structure** - Count them to find trainable parameters. Missing
   nodes indicate gradient flow problems.

5. **Residual connections create gradient highways** - Simple `AddBackward0` nodes, but architecturally
   critical for deep networks. Enable gradient flow from output to input.

6. **Recognize common patterns** - Linear (Addmm + 2 AccumulateGrad), Activation (single node),
   Normalization (NativeLayerNorm + 2 AccumulateGrad), Fused Attention (single ScaledDotProduct node).

**Practical applications**:
- Verify kernel fusion to ensure optimal performance
- Count compute-heavy nodes to estimate backward cost
- Check for unexpected nodes that indicate missed optimizations
- Trace gradient paths to debug training issues
- Inspect gradient magnitudes to detect vanishing/exploding gradients

Understanding autograd graphs is essential for serious deep learning systems work. This benchmark gives you
the tools to inspect what PyTorch is doing under the hood and optimize accordingly.
