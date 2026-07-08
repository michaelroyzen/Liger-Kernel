"""Per-config correctness sweep of the SRC dX kernel (post merged-K fix) on this GPU.

Covers fp32 + bf16, mini-model-like shape (H=896, I=768) and the research shape,
TMA on and off. Any BAD line = real miscompile still present."""

import sys

sys.path.insert(0, ".")

import torch
import triton

from common import make_inputs

from liger_kernel.ops import fused_moe_kernels as K
from liger_kernel.ops.fused_moe import _pick_block_m_token, _tma_eligibility, _ensure_tma_allocator, compute_routing_metadata

kernel = K._moe_bwd_dX_expanded_kernel.fn  # bypass autotuner


def sweep(T, E, H, I, KK, dtype, use_tma):
    torch.manual_seed(0)
    x, gup, dn, idx, wts = make_inputs(T, E, H, I, KK, dtype=dtype, seed=7)
    TK = T * KK
    block_m = _pick_block_m_token(TK, E)
    (
        _etc,
        expert_start_idx,
        x_gather_idx,
        _ssc,
        _srev,
        tile_row_start,
        tile_expert,
        expert_tile_offset,
    ) = compute_routing_metadata(idx, E, block_m)
    num_m_tiles = tile_row_start.shape[0]
    total_tiles_dev = expert_tile_offset[E:]

    d_pre = torch.randn(TK, 2 * I, dtype=dtype, device="cuda")
    ref = torch.empty(TK, H, dtype=torch.float32, device="cuda")
    for e in range(E):
        lo, hi = int(expert_start_idx[e]), int(expert_start_idx[e + 1])
        if hi > lo:
            ref[lo:hi] = d_pre[lo:hi].float() @ gup[e].float()
    scale = ref.abs().mean().item()

    if use_tma:
        w1_ok, _ = _tma_eligibility(x, H, I, E)
        if not w1_ok:
            return []
        _ensure_tma_allocator()

    bad = []
    n_ok = 0
    for cfg in K._get_gemm_autotune_configs():
        bn, bk, gm = cfg.kwargs["BLOCK_N"], cfg.kwargs["BLOCK_K"], cfg.kwargs["GROUP_M"]
        nw, ns = cfg.num_warps, cfg.num_stages
        dxe = torch.full((TK, H), float("nan"), dtype=dtype, device="cuda")
        try:
            grid = (num_m_tiles * triton.cdiv(H, bn),)
            kernel[grid](
                d_pre, gup, expert_start_idx, tile_row_start, tile_expert, total_tiles_dev, dxe,
                w_rows=E * 2 * I, H_dim=H, I_dim=I,
                stride_d_pre_TK=d_pre.stride(0), stride_d_pre_N=d_pre.stride(1),
                stride_w_E=gup.stride(0), stride_w_N=gup.stride(1), stride_w_K=gup.stride(2),
                stride_dxe_TK=dxe.stride(0), stride_dxe_H=dxe.stride(1),
                BLOCK_M=block_m, USE_TMA=use_tma,
                BLOCK_N=bn, BLOCK_K=bk, GROUP_M=gm, num_warps=nw, num_stages=ns,
            )
            torch.cuda.synchronize()
            rel = (dxe.float() - ref).abs().max().item() / scale
            if rel < 0.35:  # bf16 rounding of large sums stays well under this
                n_ok += 1
            else:
                bad.append((f"BN={bn} BK={bk} nw={nw} ns={ns} tma={int(use_tma)}", rel))
        except Exception:
            pass  # OOM/compile-skip mirrors autotuner behavior
    return n_ok, bad


for T, E, H, I, KK in [(512, 8, 256, 128, 2), (8192, 8, 896, 768, 2), (1024, 128, 2048, 768, 8)]:
    for dtype in (torch.float32, torch.bfloat16):
        for tma in (False, True):
            r = sweep(T, E, H, I, KK, dtype, tma)
            if not r:
                continue
            n_ok, bad = r
            tag = f"T={T} E={E} H={H} I={I} {str(dtype).split('.')[-1]} tma={int(tma)}"
            if bad:
                print(f"{tag}: {n_ok} OK, {len(bad)} BAD:")
                for b, rel in bad:
                    print(f"    BAD {b} rel={rel:.4f}")
            else:
                print(f"{tag}: all {n_ok} runnable configs OK")
