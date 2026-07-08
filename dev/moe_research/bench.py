"""CLI benchmark for MoE variants.

Usage:
  python bench.py --variants variants/v0_baseline.py [variants/v1_x.py ...] \
      [--T 128 1024 8192] [--modes forward backward full] [--profile] [--check] [--skew 0]
"""

import argparse

import torch

from common import bench_memory
from common import bench_variant
from common import check_correctness
from common import fmt_profile
from common import load_variant
from common import profile_kernels

QWEN3_30B = dict(E=128, H=2048, I=768, K=8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", nargs="+", required=True)
    p.add_argument("--T", nargs="+", type=int, default=[1024, 8192])
    p.add_argument("--E", type=int, default=QWEN3_30B["E"])
    p.add_argument("--H", type=int, default=QWEN3_30B["H"])
    p.add_argument("--I", type=int, default=QWEN3_30B["I"])
    p.add_argument("--K", type=int, default=QWEN3_30B["K"])
    p.add_argument("--modes", nargs="+", default=["forward", "full"])
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--skew", type=float, default=0.0)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--profile-mode", default="full")
    p.add_argument("--check", action="store_true")
    p.add_argument("--memory", action="store_true")
    args = p.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    mods = [(v, load_variant(v)) for v in args.variants]

    if args.check:
        for name, mod in mods:
            print(f"[check] {name}")
            ok = check_correctness(mod)
            if not ok:
                print(f"  !! {name} FAILED correctness — skipping benchmarks")
                return

    results = {}
    for T in args.T:
        for mode in args.modes:
            row = []
            for name, mod in mods:
                ms = bench_variant(
                    mod,
                    T,
                    args.E,
                    args.H,
                    args.I,
                    args.K,
                    dtype=dtype,
                    mode=mode,
                    skew=args.skew,
                )
                row.append((name, ms))
            results[(T, mode)] = row

    # print table
    names = [n for n, _ in mods]
    print(f"\nconfig: E={args.E} H={args.H} I={args.I} K={args.K} dtype={args.dtype} skew={args.skew}")
    header = f"{'T':>7} {'mode':>9} " + " ".join(f"{n.split('/')[-1][:28]:>30}" for n in names)
    print(header)
    for (T, mode), row in results.items():
        base_ms = row[0][1]
        cells = []
        for name, ms in row:
            speedup = base_ms / ms if ms > 0 else 0
            cells.append(f"{ms:>8.3f} ms ({speedup:>5.2f}x)".rjust(30))
        print(f"{T:>7} {mode:>9} " + " ".join(cells))

    if args.memory:
        for T in args.T:
            row = []
            for name, mod in mods:
                mb = bench_memory(
                    mod,
                    T,
                    args.E,
                    args.H,
                    args.I,
                    args.K,
                    dtype=dtype,
                    mode="full",
                    skew=args.skew,
                )
                row.append((name, mb))
            print(f"[mem full] T={T}: " + " | ".join(f"{n.split('/')[-1]}: {mb:.0f} MB" for n, mb in row))

    if args.profile:
        for name, mod in mods:
            for T in args.T:
                print(f"\n[profile {args.profile_mode}] {name} T={T}")
                rows = profile_kernels(
                    mod,
                    T,
                    args.E,
                    args.H,
                    args.I,
                    args.K,
                    dtype=dtype,
                    mode=args.profile_mode,
                )
                print(fmt_profile(rows))


if __name__ == "__main__":
    main()
