"""benchmarks's registered entry point.

Install (-e .) creates the CLI:
  pytorch-gpu-anatomy ...

You can also run without installing (if src is on PYTHONPATH):
  python -m benchmarks ...

Examples:
  pytorch-gpu-anatomy c01 contiguity-bench --device cuda:0 --dtype fp16 --m 4096 --n 4096 --k 4096 --iters 200
  pytorch-gpu-anatomy c01 sync-points       --device cuda:0 --dtype fp16 --iters 50
  pytorch-gpu-anatomy c01 autograd-inspect  --device cuda:0 --dtype fp16 --batch 2 --seq 128 --dmodel 768 --nhead 12
"""

from __future__ import annotations

import argparse
import sys


def _add_c01_subcommands(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """
    Chapter 01 CLI wiring.

    We import lazily so `--help` stays fast and doesn't require torch installed.
    """
    from benchmarks.c01_tensor_semantics import (
        run_autograd_inspect,
        run_contiguity_bench,
        run_sync_points,
    )

    c01 = subparsers.add_parser(
        'c01',
        help='Chapter 01: tensor storage/strides, contiguity, sync points, autograd graph',
    )
    c01_sub = c01.add_subparsers(dest='c01_cmd', required=True)

    # -------- contiguity-bench --------
    p_bench = c01_sub.add_parser(
        'contiguity-bench',
        help='Benchmark contiguous vs strided tensors on GPU (matmul + layernorm-ish)',
    )
    p_bench.add_argument('--device', default='cuda:0')
    p_bench.add_argument('--dtype', choices=['fp16', 'bf16', 'fp32'], default='fp16')
    p_bench.add_argument('--m', type=int, default=4096)
    p_bench.add_argument('--n', type=int, default=4096)
    p_bench.add_argument('--k', type=int, default=4096)
    p_bench.add_argument('--iters', type=int, default=200)
    p_bench.set_defaults(_run=lambda a: run_contiguity_bench(a.device, a.dtype, a.m, a.n, a.k, a.iters))

    # -------- sync-points --------
    p_sync = c01_sub.add_parser(
        'sync-points',
        help='Demonstrate explicit & implicit CPU↔GPU synchronization points and stalls',
    )
    p_sync.add_argument('--device', default='cuda:0')
    p_sync.add_argument('--dtype', choices=['fp16', 'bf16', 'fp32'], default='fp16')
    p_sync.add_argument('--iters', type=int, default=50)
    p_sync.set_defaults(_run=lambda a: run_sync_points(a.device, a.dtype, a.iters))

    # -------- autograd-inspect --------
    p_auto = c01_sub.add_parser(
        'autograd-inspect',
        help='Print grad_fn tree for a transformer-ish block (LayerNorm + MHA + MLP)',
    )
    p_auto.add_argument('--device', default='cuda:0')
    p_auto.add_argument('--dtype', choices=['fp16', 'bf16', 'fp32'], default='fp16')
    p_auto.add_argument('--batch', type=int, default=2)
    p_auto.add_argument('--seq', type=int, default=128)
    p_auto.add_argument('--dmodel', type=int, default=768)
    p_auto.add_argument('--nhead', type=int, default=12)
    p_auto.set_defaults(
        _run=lambda a: run_autograd_inspect(a.device, a.dtype, a.batch, a.seq, a.dmodel, a.nhead)
    )


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv

    parser = argparse.ArgumentParser(prog='pytorch-gpu-anatomy')
    sub = parser.add_subparsers(dest='chapter', required=True)

    _add_c01_subcommands(sub)

    args = parser.parse_args(argv)

    # Each leaf parser sets args._run
    run = getattr(args, '_run', None)
    if run is None:
        parser.error('No command selected.')
    run(args)


if __name__ == '__main__':
    main()
