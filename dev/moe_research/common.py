"""Shared harness for MoE kernel autoresearch: input gen, correctness, timing, profiling."""

import importlib.util
import os
import sys

import torch
import torch.nn as nn
import triton

DEVICE = "cuda"


def load_variant(path, name=None):
    """Load a variant module from a file path. Each variant must expose LigerFusedMoEFunction."""
    name = name or os.path.splitext(os.path.basename(path))[0]
    # Unique module name per path so multiple variants can coexist (separate autotune caches).
    modname = f"moe_variant_{name}"
    if modname in sys.modules:
        return sys.modules[modname]
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


def make_inputs(T, E, H, I, K, dtype=torch.bfloat16, seed=42, skew=0.0, requires_grad=True):
    """Generate MoE inputs. skew=0 → uniform routing; skew>0 → Zipf-like imbalance."""
    torch.manual_seed(seed)
    x = torch.randn(T, H, dtype=dtype, device=DEVICE, requires_grad=requires_grad)
    gate_up_proj = torch.randn(E, 2 * I, H, dtype=dtype, device=DEVICE) * 0.02
    down_proj = torch.randn(E, H, I, dtype=dtype, device=DEVICE) * 0.02
    gate_up_proj.requires_grad_(requires_grad)
    down_proj.requires_grad_(requires_grad)

    logits = torch.randn(T, E, device=DEVICE)
    if skew > 0:
        # bias logits so lower-index experts get more tokens (Zipf-ish)
        bias = -skew * torch.log(torch.arange(1, E + 1, device=DEVICE, dtype=torch.float))
        logits = logits + bias[None, :]
    top_k_index = torch.topk(logits, K, dim=-1).indices.to(torch.int32)
    top_k_weights = (
        torch.softmax(torch.gather(logits, 1, top_k_index.long()), dim=-1).to(dtype).requires_grad_(requires_grad)
    )
    return x, gate_up_proj, down_proj, top_k_index, top_k_weights


def reference_moe_forward(x, gate_up_proj, down_proj, top_k_index, top_k_weights):
    """HF-style python loop reference."""
    T, H = x.shape
    E = gate_up_proj.shape[0]
    final = torch.zeros_like(x)
    with torch.no_grad():
        expert_mask = torch.nn.functional.one_hot(top_k_index.long(), num_classes=E)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for eh in expert_hit:
        eidx = eh[0]
        top_k_pos, token_idx = torch.where(expert_mask[eidx])
        curr = x[token_idx]
        gate, up = nn.functional.linear(curr, gate_up_proj[eidx]).chunk(2, dim=-1)
        curr = nn.functional.silu(gate) * up
        curr = nn.functional.linear(curr, down_proj[eidx])
        curr = curr * top_k_weights[token_idx, top_k_pos, None]
        final.index_add_(0, token_idx, curr.to(final.dtype))
    return final


def check_correctness(variant_mod, T=512, E=8, H=256, I=128, K=2, verbose=True):
    """fp32 fwd+bwd check vs reference; bf16 fwd check. Returns True/False."""
    fn = variant_mod.LigerFusedMoEFunction.apply
    ok = True
    for dtype, atol, rtol in [
        (torch.float32, 1e-3, 1e-4),
        (torch.bfloat16, 1e-1, 1e-2),
    ]:
        x, gup, dn, idx, wts = make_inputs(T, E, H, I, K, dtype=dtype, seed=7)
        x1, gup1, dn1, wts1 = (t.detach().clone().requires_grad_(True) for t in (x, gup, dn, wts))
        x2, gup2, dn2, wts2 = (t.detach().clone().requires_grad_(True) for t in (x, gup, dn, wts))
        ref = reference_moe_forward(x1, gup1, dn1, idx, wts1)
        out = fn(x2, gup2, dn2, idx, wts2)
        try:
            torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)
        except AssertionError as e:
            print(f"  FWD MISMATCH ({dtype}): {str(e.args[0])[:300]}")
            ok = False
        if dtype == torch.float32:
            g = torch.randn_like(ref)
            ref.backward(g)
            out.backward(g)
            for name, a, b in [
                ("dx", x2, x1),
                ("dW1", gup2, gup1),
                ("dW2", dn2, dn1),
                ("dS", wts2, wts1),
            ]:
                try:
                    torch.testing.assert_close(a.grad, b.grad, atol=3e-3, rtol=1e-2)
                except AssertionError as e:
                    print(f"  BWD {name} MISMATCH: {str(e.args[0])[:300]}")
                    ok = False
    # Edge cases: all tokens to one expert; tiny T
    x, gup, dn, idx, wts = make_inputs(32, 8, 64, 32, 2, dtype=torch.float32, seed=3)
    idx0 = torch.zeros_like(idx)
    ref = reference_moe_forward(x, gup, dn, idx0, wts)
    out = fn(x, gup, dn, idx0, wts)
    try:
        torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-4)
    except AssertionError as e:
        print(f"  EDGE all-to-one MISMATCH: {str(e.args[0])[:200]}")
        ok = False
    if verbose:
        print(f"  correctness: {'PASS' if ok else 'FAIL'}")
    return ok


def bench_variant(variant_mod, T, E, H, I, K, dtype=torch.bfloat16, mode="full", skew=0.0, seed=42):
    """Median ms via triton do_bench. mode in {forward, backward, full, infer}."""
    fn = variant_mod.LigerFusedMoEFunction.apply

    if mode == "infer":
        x, gup, dn, idx, wts = make_inputs(T, E, H, I, K, dtype=dtype, seed=seed, skew=skew, requires_grad=False)
        with torch.no_grad():
            fn(x, gup, dn, idx, wts)  # autotune warmup
            torch.cuda.synchronize()
            return triton.testing.do_bench(lambda: fn(x, gup, dn, idx, wts), warmup=25, rep=100)

    x, gup, dn, idx, wts = make_inputs(T, E, H, I, K, dtype=dtype, seed=seed, skew=skew)
    dO = torch.randn(T, H, dtype=dtype, device=DEVICE)

    # autotune warmup (fwd + bwd once)
    y = fn(x, gup, dn, idx, wts)
    y.backward(dO)
    for t in (x, gup, dn, wts):
        t.grad = None
    torch.cuda.synchronize()

    if mode == "forward":
        ms = triton.testing.do_bench(lambda: fn(x, gup, dn, idx, wts), warmup=25, rep=100)
    elif mode == "backward":
        y = fn(x, gup, dn, idx, wts)
        ms = triton.testing.do_bench(
            lambda: torch.autograd.grad(y, (x, gup, dn, wts), dO, retain_graph=True),
            warmup=25,
            rep=100,
            grad_to_none=(x, gup, dn, wts),
        )
    elif mode == "full":

        def run():
            y = fn(x, gup, dn, idx, wts)
            y.backward(dO)

        ms = triton.testing.do_bench(run, warmup=25, rep=100, grad_to_none=(x, gup, dn, wts))
    else:
        raise ValueError(mode)
    return ms


def bench_memory(variant_mod, T, E, H, I, K, dtype=torch.bfloat16, mode="full", skew=0.0):
    fn = variant_mod.LigerFusedMoEFunction.apply
    x, gup, dn, idx, wts = make_inputs(T, E, H, I, K, dtype=dtype, skew=skew)
    dO = torch.randn(T, H, dtype=dtype, device=DEVICE)
    # warmup for autotune first (do NOT count its allocations)
    y = fn(x, gup, dn, idx, wts)
    y.backward(dO)
    for t in (x, gup, dn, wts):
        t.grad = None
    del y
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.max_memory_allocated()
    y = fn(x, gup, dn, idx, wts)
    if mode in ("backward", "full"):
        y.backward(dO)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return (peak - base) / (1024**2)  # MB above inputs


def profile_kernels(variant_mod, T, E, H, I, K, dtype=torch.bfloat16, mode="full", iters=10, skew=0.0):
    """Return list of (kernel_name, total_us, count) sorted by time, using torch profiler."""
    from torch.profiler import ProfilerActivity
    from torch.profiler import profile

    fn = variant_mod.LigerFusedMoEFunction.apply
    x, gup, dn, idx, wts = make_inputs(T, E, H, I, K, dtype=dtype, skew=skew)
    dO = torch.randn(T, H, dtype=dtype, device=DEVICE)
    y = fn(x, gup, dn, idx, wts)
    y.backward(dO)  # autotune warmup
    for t in (x, gup, dn, wts):
        t.grad = None
    torch.cuda.synchronize()

    def run_once():
        if mode == "forward":
            fn(x, gup, dn, idx, wts)
        else:
            y = fn(x, gup, dn, idx, wts)
            y.backward(dO)
            for t in (x, gup, dn, wts):
                t.grad = None

    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(iters):
            run_once()
        torch.cuda.synchronize()

    agg = {}
    for evt in prof.key_averages():
        if evt.device_type == torch.autograd.DeviceType.CUDA and evt.self_device_time_total > 0:
            agg[evt.key] = (evt.self_device_time_total, evt.count)
    rows = sorted(agg.items(), key=lambda kv: -kv[1][0])
    return [(k, v[0] / iters, v[1] // iters) for k, v in rows]


def fmt_profile(rows, top=25):
    total = sum(us for _, us, _ in rows)
    lines = [f"{'kernel':<72} {'us/iter':>10} {'count':>6} {'%':>6}"]
    for k, us, c in rows[:top]:
        lines.append(f"{k[:72]:<72} {us:>10.1f} {c:>6} {100 * us / total:>6.1f}")
    lines.append(f"{'TOTAL':<72} {total:>10.1f}")
    return "\n".join(lines)
