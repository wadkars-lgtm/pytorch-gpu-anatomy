import argparse
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utilities
# -----------------------------
def require_cuda(device: str):
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise SystemExit('CUDA requested but torch.cuda.is_available() is False.')


def set_determinism(seed: int = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pretty_bool(x: bool) -> str:
    return 'yes' if x else 'no'


@dataclass
class Timing:
    ms: float
    iters: int
    per_iter_ms: float


def time_cuda(
    fn: Callable[[], None],
    warmup: int = 20,
    iters: int = 200,
    sync_each_iter: bool = False,
) -> Timing:
    """
    CUDA timing using cudaEvents (accurate, avoids implicit sync from time.time()).
    If sync_each_iter=True you can intentionally force worst-case stalls.
    """
    if not torch.cuda.is_available():
        raise SystemExit('CUDA timing requested but CUDA is not available.')

    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
        if sync_each_iter:
            torch.cuda.synchronize()
    end.record()

    torch.cuda.synchronize()
    ms = start.elapsed_time(end)
    return Timing(ms=ms, iters=iters, per_iter_ms=ms / iters)


def time_cpu(fn: Callable[[], None], warmup: int = 5, iters: int = 50) -> Timing:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    ms = (t1 - t0) * 1000.0
    return Timing(ms=ms, iters=iters, per_iter_ms=ms / iters)


def tensor_layout_report(x: torch.Tensor, name: str):
    print(f'\n[{name}]')
    print(f'  shape          : {tuple(x.shape)}')
    print(f'  dtype          : {x.dtype}')
    print(f'  device         : {x.device}')
    print(f'  is_contiguous  : {pretty_bool(x.is_contiguous())}')
    print(f'  strides        : {tuple(x.stride())}')
    print(f'  storage_offset : {x.storage_offset()}')
    # "alignment-ish": crude proxy: offset in bytes mod 128 (not perfect, but useful intuition)
    offset_bytes = x.storage_offset() * x.element_size()
    print(f'  offset_bytes   : {offset_bytes}  (mod 128 = {offset_bytes % 128})')


# -----------------------------
# Experiment 1: Make strided / misaligned tensors
# -----------------------------
def make_variants(x: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    Returns variants derived from the SAME base tensor x.
    Some variants change shape (by design in your current version).

    NOTE: If you want apples-to-apples, you must keep shapes identical across variants.
    """
    # Classic: transpose breaks contiguity (just metadata)
    x_t = x.t()

    # Slice with offset: changes storage_offset and shape
    x_off = x[:, 1:]  # storage_offset != 0

    # Manual stride: take every other column (changes shape)
    m, n = x.shape
    n2 = n // 2
    x_as = torch.as_strided(
        x,
        size=(m, n2),
        stride=(x.stride(0), x.stride(1) * 2),
        storage_offset=0,
    )

    # Force a contiguous copy of the transpose
    x_t_contig = x_t.contiguous()

    return {
        'contiguous': x,
        'transposed_view': x_t,
        'sliced_offset_view': x_off,
        'as_strided_view': x_as,
        'transposed_contig_copy': x_t_contig,
    }


# -----------------------------
# Experiment 2: Benchmark contig vs strided on GPU
# -----------------------------
def bench_matmul(a: torch.Tensor, b: torch.Tensor) -> Callable[[], None]:
    # matmul tends to strongly prefer contiguous / expected layouts
    def _fn():
        _ = a @ b

    return _fn


def bench_layernormish(x: torch.Tensor) -> Callable[[], None]:
    # A common “memoryy” pattern: normalize across last dim, plus a pointwise op.
    # Strides can make reads less friendly.
    def _fn():
        y = x - x.mean(dim=-1, keepdim=True)
        y = y * torch.rsqrt(y.var(dim=-1, keepdim=True, unbiased=False) + 1e-5)
        _ = y * 0.5 + 0.1

    return _fn


def run_contiguity_bench(
    device_str: str,
    dtype_str: str,
    m: int,
    n: int,
    k: int,
    iters: int,
):
    device = torch.device(device_str)
    require_cuda(device_str)

    dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}[dtype_str]
    set_determinism(0)

    # Matmul test: (m,n) x (n,k)
    a = torch.randn(m, n, device=device, dtype=dtype)
    b = torch.randn(n, k, device=device, dtype=dtype)

    # Create strided variants of A
    variants = make_variants(a)
    for name, x in variants.items():
        tensor_layout_report(x, f'A variant: {name}')

    print('\n=== Matmul timing (A @ B) ===')
    torch.backends.cuda.matmul.allow_tf32 = True  # safe for fp32; no effect on fp16/bf16
    for name, x in variants.items():
        fn = bench_matmul(x, b if x.shape[1] == b.shape[0] else b[: x.shape[1], :])
        t = time_cuda(fn, warmup=20, iters=iters, sync_each_iter=False)
        print(f'{name:24s}  {t.per_iter_ms:8.4f} ms/iter')

    print('\n=== LayerNorm-ish timing ===')
    # For layernorm-ish, keep last dim fairly large
    x = torch.randn(m, n, device=device, dtype=dtype)
    v2 = make_variants(x)
    for name, xv in v2.items():
        fn = bench_layernormish(xv)
        t = time_cuda(fn, warmup=30, iters=iters, sync_each_iter=False)
        print(f'{name:24s}  {t.per_iter_ms:8.4f} ms/iter')

    print('\nNotes:')
    print('  - transposed_view / as_strided_view are *views* (no copy) but often slower.')
    print('  - transposed_contig_copy pays a one-time copy cost, then runs faster kernels.')


# -----------------------------
# Experiment 3: Sync points (where PyTorch forces CPU↔GPU sync)
# -----------------------------
def run_sync_points(device_str: str, dtype_str: str, iters: int):
    device = torch.device(device_str)
    require_cuda(device_str)
    dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}[dtype_str]

    x = torch.randn(8192, 8192, device=device, dtype=dtype)
    w = torch.randn(8192, 8192, device=device, dtype=dtype)

    def heavy_op():
        y = x @ w
        # keep it alive
        return y

    # 1) Async enqueue timing (no explicit sync each iter)
    def fn_async():
        _ = heavy_op()

    t_async = time_cuda(fn_async, warmup=5, iters=iters, sync_each_iter=False)
    print(f'\nAsync (no per-iter sync): {t_async.per_iter_ms:.4f} ms/iter')

    # 2) Force worst-case: sync every iteration
    t_sync = time_cuda(fn_async, warmup=5, iters=iters, sync_each_iter=True)
    print(f'Forced sync each iter:    {t_sync.per_iter_ms:.4f} ms/iter')

    # 3) Common implicit syncs
    y = heavy_op()
    torch.cuda.synchronize()

    # .item() forces sync because CPU needs the scalar
    t0 = time.perf_counter()
    s = y[0, 0].item()
    t1 = time.perf_counter()
    print(f'\n.item() sync cost: {(t1 - t0) * 1000:.3f} ms  (value={s:.5f})')

    # Printing a CUDA tensor also tends to sync (or trigger device->host copy)
    # We'll show the cost by timing a minimal print of a single element.
    y2 = heavy_op()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    print(f'print triggers: {y2[0, 0]}')
    t1 = time.perf_counter()
    print(f'print cost: {(t1 - t0) * 1000:.3f} ms')

    print('\nTakeaway:')
    print('  - CUDA ops are queued asynchronously.')
    print('  - Anything that needs a CPU value (item/print/cpu()/numpy()) forces a sync.')


# -----------------------------
# Experiment 4: Autograd graph inspection for a transformer-ish block
# -----------------------------
class TinyTransformerBlock(nn.Module):
    """
    Not a full LLM block, but close enough to show .grad_fn structure:
      - LayerNorm
      - MultiheadAttention (batch_first)
      - Residual
      - MLP
      - Residual
    """

    def __init__(self, d_model: int, nhead: int, mlp_mult: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, mlp_mult * d_model)
        self.fc2 = nn.Linear(mlp_mult * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        h2 = self.ln2(x)
        mlp = self.fc2(F.gelu(self.fc1(h2)))
        x = x + mlp
        return x


def walk_grad_fn(fn, max_nodes: int = 50):
    """
    BFS over .grad_fn -> .next_functions
    """

    q: deque[int] = deque()
    seen = set()
    q.append(fn)
    out: list[int] = []
    while q and len(out) < max_nodes:
        cur = q.popleft()
        if cur is None:
            continue
        if id(cur) in seen:
            continue
        seen.add(id(cur))
        out.append(cur)
        for nxt, _idx in getattr(cur, 'next_functions', []):
            if nxt is not None and id(nxt) not in seen:
                q.append(nxt)
    return out


def run_autograd_inspect(device_str: str, dtype_str: str, b: int, t: int, d: int, nhead: int):
    device = torch.device(device_str)
    require_cuda(device_str)
    dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}[dtype_str]

    block = TinyTransformerBlock(d_model=d, nhead=nhead).to(device=device, dtype=dtype)
    x = torch.randn(b, t, d, device=device, dtype=dtype, requires_grad=True)

    y = block(x)
    loss = y.float().pow(2).mean()  # float() to keep loss stable
    loss.backward()

    print('\n=== grad_fn tree (loss.grad_fn) ===')
    nodes = walk_grad_fn(loss.grad_fn, max_nodes=60)
    for i, n in enumerate(nodes):
        name = n.__class__.__name__
        print(f'{i:02d}: {name}')

    print('\nAlso useful:')
    print('  - y.grad_fn:', y.grad_fn.__class__.__name__ if y.grad_fn else None)
    print('  - x.grad shape:', tuple(x.grad.shape) if x.grad is not None else None)
    print('  - First few grad stats:', x.grad.float().mean().item(), x.grad.float().std().item())


# -----------------------------
# Entry point
# -----------------------------
def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)

    p_bench = sub.add_parser('contiguity-bench')
    p_bench.add_argument('--device', default='cuda:0')
    p_bench.add_argument('--dtype', choices=['fp16', 'bf16', 'fp32'], default='fp16')
    p_bench.add_argument('--m', type=int, default=4096)
    p_bench.add_argument('--n', type=int, default=4096)
    p_bench.add_argument('--k', type=int, default=4096)
    p_bench.add_argument('--iters', type=int, default=200)

    p_sync = sub.add_parser('sync-points')
    p_sync.add_argument('--device', default='cuda:0')
    p_sync.add_argument('--dtype', choices=['fp16', 'bf16', 'fp32'], default='fp16')
    p_sync.add_argument('--iters', type=int, default=50)

    p_auto = sub.add_parser('autograd-inspect')
    p_auto.add_argument('--device', default='cuda:0')
    p_auto.add_argument('--dtype', choices=['fp16', 'bf16', 'fp32'], default='fp16')
    p_auto.add_argument('--batch', type=int, default=2)
    p_auto.add_argument('--seq', type=int, default=128)
    p_auto.add_argument('--dmodel', type=int, default=768)
    p_auto.add_argument('--nhead', type=int, default=12)

    args = p.parse_args()

    if args.cmd == 'contiguity-bench':
        run_contiguity_bench(args.device, args.dtype, args.m, args.n, args.k, args.iters)
    elif args.cmd == 'sync-points':
        run_sync_points(args.device, args.dtype, args.iters)
    elif args.cmd == 'autograd-inspect':
        run_autograd_inspect(args.device, args.dtype, args.batch, args.seq, args.dmodel, args.nhead)
    else:
        raise SystemExit(f'Unknown cmd: {args.cmd}')


if __name__ == '__main__':
    main()
