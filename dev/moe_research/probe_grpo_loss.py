"""Profile Liger's chunked GRPO loss (LigerFusedLinearGRPOLoss) at Qwen3.5-35B-A3B
trainer shapes on B300.

Their config (run_train_grpo_multi.sh): beta=0.0, temperature=1.0, batch 4/device,
TORCH_DYNAMO_DISABLE=1 (so the torch.compile of the loss math is inert -> compiled=False
arm is the realistic one). Vocab 248320, hidden 2048 (transformers 5.13 Qwen3.5-MoE).

Reports: wall ms (fwd+bwd), peak memory, effective TFLOPS vs the 6*N*H*V analytic
FLOPs (fwd GEMM + dX + dW), and a per-kernel profiler breakdown at the largest shape.
"""

import sys

import torch

from liger_kernel.chunked_loss.grpo_loss import LigerFusedLinearGRPOLoss

H, V = 2048, 248320
DEV = "cuda"


def make_case(B, T, seed=7):
    g = torch.Generator(device=DEV).manual_seed(seed)
    hidden = torch.randn(B, T, H, device=DEV, dtype=torch.bfloat16, generator=g) * 0.5
    hidden.requires_grad_(True)
    weight = (torch.randn(V, H, device=DEV, dtype=torch.bfloat16, generator=g) * 0.02).requires_grad_(True)
    ids = torch.randint(0, V, (B, T), device=DEV, generator=g)
    mask = torch.ones(B, T, device=DEV, dtype=torch.long)
    adv = torch.randn(B, device=DEV, generator=g)
    old_logps = (-torch.rand(B, T, device=DEV, generator=g) * 3).to(torch.float32)
    return hidden, weight, ids, mask, adv, old_logps


def run_once(loss_fn, case):
    hidden, weight, ids, mask, adv, old = case
    hidden.grad = None
    weight.grad = None
    out = loss_fn(hidden, weight, ids, mask, adv, old_per_token_logps=old)
    loss = out[0] if isinstance(out, tuple) else out
    loss.backward()
    return loss


def bench(loss_fn, case, iters=5):
    run_once(loss_fn, case)  # warmup / compile
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.max_memory_allocated()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        run_once(loss_fn, case)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    peak = (torch.cuda.max_memory_allocated() - base) / 1024**2
    return ms, peak


def main():
    print(f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}, H={H} V={V}")

    # cuBLAS floor for context: the same 6*N*H*V FLOPs at dense-GEMM speed
    n_probe = 8192
    a = torch.randn(n_probe, H, device=DEV, dtype=torch.bfloat16)
    w = torch.randn(V, H, device=DEV, dtype=torch.bfloat16)
    torch.cuda.synchronize()
    import triton

    ms_gemm = triton.testing.do_bench(lambda: a @ w.t(), warmup=10, rep=25)
    gemm_tflops = 2 * n_probe * H * V / ms_gemm / 1e9
    print(f"cuBLAS bf16 lm_head GEMM rate: {gemm_tflops:.0f} TFLOPS "
          f"(floor for 6NHV = {6:.0f}x fwd GEMM work)\n")

    for B, T, cs, compiled in [
        (4, 4096, 1, False),
        (4, 16384, 1, False),   # ~their packed scale per rank
        (4, 16384, 4, False),   # all sequences in one chunk
        (4, 16384, 1, True),    # what compiled=True would buy (they run dynamo-disabled)
        (8, 8192, 1, False),
    ]:
        N = B * T
        flops = 6.0 * N * H * V
        loss_fn = LigerFusedLinearGRPOLoss(
            beta=0.0, use_ref_model=False, chunk_size=cs, compiled=compiled,
            temperature=1.0, epsilon_low=0.2, epsilon_high=0.2,
        )
        case = make_case(B, T)
        try:
            ms, peak = bench(loss_fn, case, iters=3 if N >= 65536 else 5)
            floor_ms = flops / (gemm_tflops * 1e9)
            print(f"B={B} T={T:>6} chunk={cs} compiled={int(compiled)}: "
                  f"{ms:8.1f} ms  {flops / ms / 1e9:6.0f} TFLOPS eff  "
                  f"(GEMM-floor {floor_ms:5.1f} ms -> {ms / floor_ms:4.1f}x)  peak +{peak:7.0f} MB")
        except Exception as e:
            print(f"B={B} T={T:>6} chunk={cs} compiled={int(compiled)}: ERR {str(e)[:100]}")
        del loss_fn, case
        torch.cuda.empty_cache()

    # per-kernel breakdown at their scale
    from torch.profiler import ProfilerActivity, profile

    B, T = 4, 16384
    loss_fn = LigerFusedLinearGRPOLoss(beta=0.0, use_ref_model=False, chunk_size=1, compiled=False, temperature=1.0)
    case = make_case(B, T)
    run_once(loss_fn, case)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(2):
            run_once(loss_fn, case)
        torch.cuda.synchronize()
    print(f"\nper-kernel breakdown (B={B}, T={T}, chunk=1, compiled=0):")
    agg = {}
    for evt in prof.key_averages():
        if evt.device_type == torch.autograd.DeviceType.CUDA and evt.self_device_time_total > 0:
            agg[evt.key] = (evt.self_device_time_total / 2, evt.count // 2)
    total = sum(us for us, _ in agg.values())
    for k, (us, c) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:12]:
        print(f"  {k[:86]:<86} {us:>9.0f} us  x{c:<5} {100 * us / total:5.1f}%")
    print(f"  {'TOTAL':<86} {total:>9.0f} us")


if __name__ == "__main__":
    main()
