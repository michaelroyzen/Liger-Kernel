"""Eager DeepseekV32Experts loop vs Liger fused grouped-GEMM MoE at DeepSeek-3.2 shapes.

Usage:
    LD_LIBRARY_PATH= PYTHONPATH=src:~/forks/transformers/src python bench_deepseek_moe.py
"""

import torch

from transformers.models.deepseek_v32.configuration_deepseek_v32 import DeepseekV32Config
from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32Experts

from liger_kernel.transformers.swiglu import LigerExperts

E, H, I, K = 256, 7168, 2048, 8
T = 4096
DTYPE = torch.bfloat16


def bench(mod, x, idx, w, iters=5, warmup=2):
    for _ in range(warmup):
        out = mod(x, idx, w)
        out.sum().backward()
        x.grad = None
        mod.gate_up_proj.grad = None
        mod.down_proj.grad = None
    torch.cuda.synchronize()
    start, mid, end = (torch.cuda.Event(enable_timing=True) for _ in range(3))
    fwd = bwd = 0.0
    for _ in range(iters):
        start.record()
        out = mod(x, idx, w)
        mid.record()
        out.sum().backward()
        end.record()
        torch.cuda.synchronize()
        fwd += start.elapsed_time(mid)
        bwd += mid.elapsed_time(end)
        x.grad = None
        mod.gate_up_proj.grad = None
        mod.down_proj.grad = None
    return fwd / iters, bwd / iters, out


def main():
    torch.manual_seed(0)
    config = DeepseekV32Config(
        hidden_size=H, moe_intermediate_size=I, n_routed_experts=E, num_experts_per_tok=K
    )
    eager = DeepseekV32Experts(config).to("cuda", DTYPE)
    with torch.no_grad():
        eager.gate_up_proj.normal_(0, 0.02)
        eager.down_proj.normal_(0, 0.02)
    fused = LigerExperts(config).to("cuda", DTYPE)
    with torch.no_grad():
        fused.gate_up_proj.copy_(eager.gate_up_proj)
        fused.down_proj.copy_(eager.down_proj)

    x = torch.randn(T, H, device="cuda", dtype=DTYPE, requires_grad=True)
    logits = torch.randn(T, E, device="cuda")
    idx = logits.topk(K, dim=-1).indices
    w = torch.softmax(logits.gather(1, idx), dim=-1).to(DTYPE)

    f_fwd, f_bwd, f_out = bench(fused, x, idx.to(torch.int32), w)
    x2 = x.detach().clone().requires_grad_(True)
    e_fwd, e_bwd, e_out = bench(eager, x2, idx, w, iters=2, warmup=1)

    err = (f_out.float() - e_out.float()).abs().max().item()
    scale = e_out.float().abs().max().item()
    print(f"T={T} E={E} H={H} I={I} K={K} bf16")
    print(f"  eager loop : fwd {e_fwd:8.2f} ms  bwd {e_bwd:8.2f} ms  tot {e_fwd + e_bwd:8.2f} ms")
    print(f"  liger fused: fwd {f_fwd:8.2f} ms  bwd {f_bwd:8.2f} ms  tot {f_fwd + f_bwd:8.2f} ms")
    print(f"  max |diff| = {err:.4e} (out scale {scale:.2f})")


if __name__ == "__main__":
    main()
