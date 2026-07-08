"""Stage-0 probe (E27): CUTLASS-backed torch._grouped_mm as the GEMM engine for a
hybrid pipeline (Triton routing/gather/epilogues + CUTLASS grouped GEMM).

Measures, on the REAL routing distribution (Qwen3 shard, T=8192/32768, E=128, K=8):
  1. gather cost:      x_perm = x[x_gather_idx]                (materialization tax)
  2. up-proj GEMM:     _grouped_mm(x_perm, W1^T view, offs)    vs fused Triton kernel
  3. swiglu epilogue:  silu(gate)*up as a separate pass        (fusion loss tax)
  4. down-proj GEMM:   _grouped_mm(post, W2^T view, offs)      vs fused Triton kernel
  5. dW shapes:        (2d, 2d, offs) -> 3d support check      (backward feasibility)

Reference fused-kernel times (v18 profile @T=8192, this box): up_proj+swiglu 524 us,
down_proj 261 us, bwd_down 614 us, dX 394 us, dW1 653 us, dW2 371 us.
"""

import sys

sys.path.insert(0, ".")

import torch
import triton

from common import make_inputs

from liger_kernel.ops.fused_moe import _pick_block_m_token, compute_routing_metadata

E, H, I, KK = 128, 2048, 768, 8


def bench(fn, *args, **kwargs):
    fn(*args, **kwargs)
    torch.cuda.synchronize()
    return triton.testing.do_bench(lambda: fn(*args, **kwargs), warmup=25, rep=100)


def run(T):
    TK = T * KK
    x, gup, dn, idx, wts = make_inputs(T, E, H, I, KK, dtype=torch.bfloat16, seed=42, requires_grad=False)
    block_m = _pick_block_m_token(TK, E)
    (_, expert_start, x_gather_idx, _s, _r, _trs, _te, _eto) = compute_routing_metadata(idx, E, block_m)
    offs = expert_start[1:].contiguous()  # (E,) int32 cumulative segment ends
    gidx = x_gather_idx.long()

    print(f"\n=== T={T} (TK={TK}, avg tokens/expert={TK // E}) ===")

    # 1. gather (materialization tax the fused kernels don't pay)
    t_gather = bench(lambda: x[gidx])
    x_perm = x[gidx]
    print(f"gather x[idx] -> (TK, H):      {t_gather * 1e3:7.1f} us  "
          f"({(2 * TK * H * 2) / t_gather / 1e9:5.2f} TB/s)")

    # 2. up-proj grouped GEMM: (TK, H) x (E, H, 2I) -> (TK, 2I)
    w1t = gup.transpose(1, 2)  # (E, H, 2I) view
    flops_up = 2.0 * TK * H * (2 * I)
    try:
        t_up = bench(torch._grouped_mm, x_perm, w1t, offs=offs)
        pre = torch._grouped_mm(x_perm, w1t, offs=offs)
        print(f"grouped_mm up-proj (view B):   {t_up * 1e3:7.1f} us  ({flops_up / t_up / 1e9:6.0f} TFLOPS)"
              f"   [fused triton kernel incl. swiglu: 524 us @T=8192]")
    except Exception as e:
        w1c = w1t.contiguous()
        try:
            t_up = bench(torch._grouped_mm, x_perm, w1c, offs=offs)
            pre = torch._grouped_mm(x_perm, w1c, offs=offs)
            print(f"grouped_mm up-proj (contig B): {t_up * 1e3:7.1f} us  ({flops_up / t_up / 1e9:6.0f} TFLOPS)"
                  f"   [view-B failed: {str(e)[:60]}]")
        except Exception as e2:
            print(f"grouped_mm up-proj: FAILED     {str(e2)[:120]}")
            return

    # correctness spot-check vs segment-wise matmul
    e0 = int(expert_start[1])
    ref0 = x_perm[:e0].float() @ gup[0].float().t()
    rel = (pre[:e0].float() - ref0).abs().max().item() / (ref0.abs().mean().item() + 1e-9)
    print(f"   correctness (expert 0 segment): rel={rel:.4f} {'OK' if rel < 0.3 else 'BAD'}")

    # 3. swiglu epilogue as a separate pass (the fusion loss)
    gate, up = pre[:, :I], pre[:, I:]
    t_swiglu = bench(lambda: torch.nn.functional.silu(gate) * up)
    post = torch.nn.functional.silu(gate) * up
    print(f"separate swiglu epilogue:      {t_swiglu * 1e3:7.1f} us")

    # 4. down-proj grouped GEMM: (TK, I) x (E, I, H) -> (TK, H)
    w2t = dn.transpose(1, 2)  # (E, I, H) view
    flops_dn = 2.0 * TK * I * H
    try:
        t_dn = bench(torch._grouped_mm, post, w2t, offs=offs)
        print(f"grouped_mm down-proj (view B): {t_dn * 1e3:7.1f} us  ({flops_dn / t_dn / 1e9:6.0f} TFLOPS)"
              f"   [fused triton kernel: 261 us @T=8192]")
    except Exception as e:
        t_dn = bench(torch._grouped_mm, post, w2t.contiguous(), offs=offs)
        print(f"grouped_mm down-proj (contig): {t_dn * 1e3:7.1f} us  ({flops_dn / t_dn / 1e9:6.0f} TFLOPS)"
              f"   [view-B failed: {str(e)[:60]}]")

    # 5. dW shape: (2d^T, 2d, offs) -> 3d (jagged contraction dim)
    d_pre = torch.randn_like(pre)
    try:
        t_dw1 = bench(torch._grouped_mm, x_perm.t(), d_pre, offs=offs)
        flops_dw1 = 2.0 * TK * H * 2 * I
        print(f"grouped_mm dW1 (A^T B jagged): {t_dw1 * 1e3:7.1f} us  ({flops_dw1 / t_dw1 / 1e9:6.0f} TFLOPS)"
              f"   [fused triton dW1: 653 us @T=8192]")
    except Exception as e:
        print(f"grouped_mm dW1: NOT SUPPORTED  {str(e)[:120]}")

    # composite (forward): gather + up GEMM + swiglu + down GEMM vs fused pair
    hybrid_fwd = t_gather + t_up + t_swiglu + t_dn
    print(f"hybrid fwd GEMM-path total:    {hybrid_fwd * 1e3:7.1f} us   "
          f"vs fused triton up+down = 785 us @T=8192")


if __name__ == "__main__":
    print(f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}")
    run(8192)
    run(32768)
    run(1024)
