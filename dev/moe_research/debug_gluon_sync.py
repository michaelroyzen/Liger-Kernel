"""Bisect the gluon sync-MMA kernel: 1 K-iter vs many, identity vs random gather."""

import torch
import triton

from probe_gluon_gemm import SMEM_LAYOUT_16B, TensorDescriptor, _gather_offsets_layout, _gluon_sync_kernel


def run(T, M, N, K, BM, BN, BK, identity_idx=False, tag=""):
    torch.manual_seed(0)
    x = torch.randn(T, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
    if identity_idx:
        idx = torch.arange(M, device="cuda", dtype=torch.int32) % T
    else:
        idx = torch.randint(0, T, (M,), device="cuda", dtype=torch.int32)
    ref = (x[idx.long()].float() @ w.float().T)
    x_desc = TensorDescriptor.from_tensor(x, [1, BK], SMEM_LAYOUT_16B)
    w_desc = TensorDescriptor.from_tensor(w, [BN, BK], SMEM_LAYOUT_16B)
    c = torch.full((M, N), float("nan"), device="cuda", dtype=torch.bfloat16)
    _gluon_sync_kernel[(M // BM, N // BN)](x_desc, w_desc, idx, c, N, K, BM, BN, BK,
                                           NUM_WARPS=8, idx_layout=_gather_offsets_layout(BM, 8), num_warps=8)
    torch.cuda.synchronize()
    cf = c.float()
    scale = ref.abs().mean().item()
    rel = (cf - ref).abs().max().item() / scale
    frac_bad = ((cf - ref).abs() > 0.05 * scale).float().mean().item()
    print(f"{tag:>32}: rel={rel:8.4f} frac_bad={frac_bad:6.3f} nan={torch.isnan(cf).any().item()}")
    return cf, ref


if __name__ == "__main__":
    run(256, 128, 256, 64, 128, 256, 64, identity_idx=True, tag="1 iter, identity idx")
    run(256, 128, 256, 64, 128, 256, 64, identity_idx=False, tag="1 iter, random idx")
    run(256, 128, 256, 128, 128, 256, 64, identity_idx=False, tag="2 iters, random idx")
    run(256, 128, 128, 64, 128, 128, 64, identity_idx=False, tag="1 iter, BN=128")
    run(8192, 65536 // 512 * 512, 768, 2048, 128, 256, 64, identity_idx=False, tag="full shape")
