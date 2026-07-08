"""Dump autotuner-selected configs for each MoE kernel on this GPU at given T."""

import sys

sys.path.insert(0, ".")

import torch

from common import load_variant, make_inputs

v = load_variant(sys.argv[1] if len(sys.argv) > 1 else "variants/v15_fixdx.py")

QW = dict(E=128, H=2048, I=768, K=8)

for T in [1024, 8192, 32768]:
    x, gup, dn, idx, wts = make_inputs(T, **QW, dtype=torch.bfloat16)
    y = v.LigerFusedMoEFunction.apply(x, gup, dn, idx, wts)
    y.backward(torch.randn_like(y))
    torch.cuda.synchronize()

names = [
    "_fused_up_proj_swiglu_kernel",
    "_fused_down_proj_kernel",
    "_moe_bwd_down_proj_kernel",
    "_moe_bwd_dW2_kernel",
    "_moe_bwd_dX_expanded_kernel",
    "_moe_bwd_dW1_kernel",
    "_token_gather_weighted_sum_kernel",
]
for n in names:
    k = getattr(v, n, None)
    if k is None:
        continue
    print(f"\n{n}:")
    for key, cfg in getattr(k, "cache", {}).items():
        print(f"  key={key}")
        print(f"    -> {cfg}")
