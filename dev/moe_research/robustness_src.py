"""Robustness checks for the SHIPPED src kernel (same suite as robustness.py, which
targeted the v8 variant): skewed routing, Mixtral shape, CUDA graph capture."""

import sys

sys.path.insert(0, ".")

import torch

from common import bench_variant, load_variant, make_inputs

import liger_kernel.ops.fused_moe as shipped


class _Shipped:
    LigerFusedMoEFunction = shipped.LigerFusedMoEFunction


def perf_table(title, configs, modes=("forward", "full"), skew=0.0):
    print(f"\n=== {title} (skew={skew}) ===")
    m0 = load_variant("variants/v0_baseline.py")
    for T, E, H, I, K in configs:
        for mode in modes:
            ms0 = bench_variant(m0, T, E, H, I, K, mode=mode, skew=skew)
            ms1 = bench_variant(_Shipped, T, E, H, I, K, mode=mode, skew=skew)
            print(
                f"T={T:>6} E={E:>3} H={H} I={I:>5} K={K} {mode:>8}: "
                f"v0={ms0:7.3f} ms  final={ms1:7.3f} ms  ({ms0 / ms1:4.2f}x)"
            )


def cudagraph_test():
    print("\n=== CUDA graph capture (shipped src, sync-free) ===")
    fn = shipped.LigerFusedMoEFunction.apply
    T, E, H, I, K = 1024, 128, 2048, 768, 8

    x, gup, dn, idx, wts = make_inputs(T, E, H, I, K, requires_grad=False)
    with torch.no_grad():
        for _ in range(3):  # warmup + autotune
            ref = fn(x, gup, dn, idx, wts)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = fn(x, gup, dn, idx, wts)
        g.replay()
        torch.cuda.synchronize()
        ok = torch.allclose(out, ref, atol=1e-2, rtol=1e-2)
        print(f"inference capture+replay: {'OK' if ok else 'MISMATCH'}")

    x, gup, dn, idx, wts = make_inputs(T, E, H, I, K, requires_grad=True)
    dO = torch.randn(T, H, dtype=x.dtype, device=x.device)

    def moe_fn(x_, gup_, dn_, wts_):
        return fn(x_, gup_, dn_, idx, wts_)

    for _ in range(3):
        y = moe_fn(x, gup, dn, wts)
        y.backward(dO)
        for t in (x, gup, dn, wts):
            t.grad = None
    torch.cuda.synchronize()
    # Fresh leaves: warmup pins AccumulateGrad nodes to the legacy stream.
    x2, gup2, dn2, wts2 = (t.detach().clone().requires_grad_(True) for t in (x, gup, dn, wts))
    graphed = torch.cuda.make_graphed_callables(moe_fn, (x2, gup2, dn2, wts2))
    y = graphed(x2, gup2, dn2, wts2)
    y.backward(dO)
    torch.cuda.synchronize()
    print(f"training graphed-callable fwd+bwd: OK (dx norm={x2.grad.norm().item():.4f})")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("skew", "all"):
        perf_table("Qwen3-30B skewed routing", [(8192, 128, 2048, 768, 8)], skew=1.5)
    if which in ("mixtral", "all"):
        perf_table("Mixtral-8x7B shape", [(4096, 8, 4096, 14336, 2)])
    if which in ("graph", "all"):
        cudagraph_test()
