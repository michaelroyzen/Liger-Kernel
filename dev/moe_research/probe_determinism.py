"""Bitwise run-to-run determinism check of the shipped fused-MoE op on B300,
mirroring the methodology of fla-org/flash-linear-attention#945 (position-weighted
bit-hash over 25 identical calls; any scheduling race shows as >1 distinct hash).

Expectation: out/dx/dW1/dW2 bitwise stable (no cross-CTA memory recurrences, no
fp atomics on their paths); dS *may* legitimately vary (fp32 atomic_add partial
sums, order depends on CTA scheduling).
"""

import sys

sys.path.insert(0, ".")

import torch

from common import make_inputs

import liger_kernel.ops.fused_moe as shipped

fn = shipped.LigerFusedMoEFunction.apply


def bithash(t, max_elems=16_777_216):
    v = t.detach().contiguous()
    if v.dtype == torch.bfloat16:
        v = v.view(torch.int16)
    elif v.dtype == torch.float32:
        v = v.view(torch.int32)
    v = v.flatten()
    if v.numel() > max_elems:
        v = v[:: (v.numel() + max_elems - 1) // max_elems]
    v = v.to(torch.int64)
    w = torch.arange(v.numel(), device=v.device, dtype=torch.int64) % 8191 + 1
    return int((v * w).sum().item())


def run(T, iters=25):
    x, gup, dn, idx, wts = make_inputs(T, 128, 2048, 768, 8, dtype=torch.bfloat16, seed=3)
    dO = torch.randn(T, 2048, dtype=torch.bfloat16, device="cuda")
    # autotune warmup
    y = fn(x, gup, dn, idx, wts)
    y.backward(dO)
    torch.cuda.synchronize()

    hashes = {k: set() for k in ["out", "dx", "dW1", "dW2", "dS"]}
    for _ in range(iters):
        for t in (x, gup, dn, wts):
            t.grad = None
        y = fn(x, gup, dn, idx, wts)
        y.backward(dO)
        torch.cuda.synchronize()
        hashes["out"].add(bithash(y))
        hashes["dx"].add(bithash(x.grad))
        hashes["dW1"].add(bithash(gup.grad))
        hashes["dW2"].add(bithash(dn.grad))
        hashes["dS"].add(bithash(wts.grad))

    row = "  ".join(f"{k}:{len(v)}" for k, v in hashes.items())
    verdict = "DETERMINISTIC" if all(len(v) == 1 for k, v in hashes.items() if k != "dS") else "RACE DETECTED"
    print(f"T={T:>6}: distinct hashes over {iters} calls -> {row}   [{verdict}"
          f"{'; dS varies (fp32 atomics, by design)' if len(hashes['dS']) > 1 else ''}]")


if __name__ == "__main__":
    print(f"{torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)}, "
          f"torch {torch.__version__}")
    run(8192)
    run(32768)
    run(1024)
