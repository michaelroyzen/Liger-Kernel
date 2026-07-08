"""Does the monorepo's pinned upstream Liger rev (96ce8e8) hit the sm103 dX
miscompile at Qwen3.5-35B-A3B's MoE dims (H=2048, E=256, I_moe=512, K=8)?

Run twice:
  PYTHONPATH=/tmp/liger-upstream/src python probe_monorepo_repro.py   # pinned upstream
  python probe_monorepo_repro.py                                       # patched fork
"""

import sys

sys.path.insert(0, "/home/ubuntu/Liger-Kernel/dev/moe_research")

import torch

import liger_kernel.ops.fused_moe as fm

from common import make_inputs, reference_moe_forward

print(f"liger_kernel from: {fm.__file__}")
print(f"{torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)}")

E, H, I, K = 256, 2048, 512, 8  # Qwen3.5-35B-A3B MoE dims (transformers 5.13 defaults)
fn = fm.LigerFusedMoEFunction.apply

for T, dtype, tag in [(8192, torch.float32, "fp32"), (8192, torch.bfloat16, "bf16")]:
    # TK/E = 256 -> BLOCK_M=128, the tcgen05 regime where the two-dot bug bites
    x, gup, dn, idx, wts = make_inputs(T, E, H, I, K, dtype=dtype, seed=17)
    x1, gup1, dn1, wts1 = (t.detach().clone().requires_grad_(True) for t in (x, gup, dn, wts))
    x2, gup2, dn2, wts2 = (t.detach().clone().requires_grad_(True) for t in (x, gup, dn, wts))
    ref = reference_moe_forward(x1, gup1, dn1, idx, wts1)
    out = fn(x2, gup2, dn2, idx, wts2)
    g = torch.randn_like(ref)
    ref.backward(g)
    out.backward(g)
    fwd_rel = (out - ref).abs().max().item() / (ref.abs().mean().item() + 1e-12)
    rows = []
    worst = 0.0
    for name, a, b in [("dx", x2, x1), ("dW1", gup2, gup1), ("dW2", dn2, dn1), ("dS", wts2, wts1)]:
        rel = (a.grad - b.grad).abs().max().item() / (b.grad.abs().mean().item() + 1e-12)
        if name == "dx":
            worst = rel
        rows.append(f"{name}={rel:8.3f}")
    # bf16 noise floor at these dims is ~0.05-0.1 rel; the miscompile shows as O(10-1000)
    bad = worst > 1.0
    print(f"T={T} {tag}: fwd_rel={fwd_rel:.4f}  " + "  ".join(rows) + f"   -> {'DX CORRUPTED' if bad else 'clean'}")

# show which dX config autotune picked (upstream: is it one of the broken ones?)
try:
    import liger_kernel.ops.fused_moe_kernels as fk

    for key, cfg in getattr(fk._moe_bwd_dX_expanded_kernel, "cache", {}).items():
        print(f"dX config picked: {cfg}")
except Exception as e:
    print("config dump failed:", e)
