"""Final A/B: original PR #1179 implementation (v0 snapshot) vs shipped src version."""

import sys

sys.path.insert(0, ".")

import torch  # noqa: E402

from common import bench_memory, bench_variant, load_variant  # noqa: E402

import liger_kernel.ops.fused_moe as shipped  # noqa: E402

QWEN = dict(E=128, H=2048, I=768, K=8)


class _ShippedWrapper:
    LigerFusedMoEFunction = shipped.LigerFusedMoEFunction


def main():
    v0 = load_variant("variants/v0_baseline.py")
    for T in [128, 1024, 8192, 32768]:
        for mode in ["forward", "backward", "full", "infer"]:
            ms0 = bench_variant(v0, T, **QWEN, mode=mode)
            ms1 = bench_variant(_ShippedWrapper, T, **QWEN, mode=mode)
            print(f"T={T:>6} {mode:>8}: v0={ms0:8.3f} ms   final={ms1:8.3f} ms   ({ms0 / ms1:5.2f}x)")
    for T in [8192, 32768]:
        mb0 = bench_memory(v0, T, **QWEN, mode="full")
        mb1 = bench_memory(_ShippedWrapper, T, **QWEN, mode="full")
        print(f"T={T:>6} peak-mem full: v0={mb0:7.0f} MB  final={mb1:7.0f} MB  ({mb0 - mb1:+.0f})")


if __name__ == "__main__":
    with torch.cuda.device(0):
        main()
