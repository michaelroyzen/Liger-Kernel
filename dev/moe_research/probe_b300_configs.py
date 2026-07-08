"""Probe: which (kernel, config) pairs miscompile on B300/sm103?

Runs the dX_expanded kernel (the one implicated in the dx mismatch) config-by-config
with the autotuner bypassed, comparing each output against a torch reference.
"""

import sys

sys.path.insert(0, ".")

import torch
import triton

from common import make_inputs

from liger_kernel.ops.fused_moe import _pick_block_m_token, compute_routing_metadata
from liger_kernel.ops import fused_moe_kernels as K

torch.manual_seed(0)

T, E, H, I, KK = 512, 8, 256, 128, 2
dtype = torch.float32

x, gup, dn, idx, wts = make_inputs(T, E, H, I, KK, dtype=dtype, seed=7)
TK = T * KK
block_m = _pick_block_m_token(TK, E)
print(f"BLOCK_M={block_m}")

(
    etc,
    expert_start_idx,
    x_gather_idx,
    s_scatter_idx,
    s_rev,
    tile_row_start,
    tile_expert,
    expert_tile_offset,
) = compute_routing_metadata(idx, E, block_m)

num_m_tiles = tile_row_start.shape[0]
total_tiles_dev = expert_tile_offset[E:]

# fabricate a d_pre_act and compute reference dx_expanded = d_pre @ W1[e]
d_pre = torch.randn(TK, 2 * I, dtype=dtype, device="cuda")
# reference: for each sorted row r with expert e: dxe[r] = d_pre[r] @ W1[e]  (W1: (2I, H))
gather = x_gather_idx.long()
experts_sorted = torch.repeat_interleave(
    torch.arange(E, device="cuda"), (expert_start_idx[1:] - expert_start_idx[:-1]).long()
)
ref = torch.empty(TK, H, dtype=dtype, device="cuda")
for e in range(E):
    lo, hi = int(expert_start_idx[e]), int(expert_start_idx[e + 1])
    if hi > lo:
        ref[lo:hi] = d_pre[lo:hi] @ gup[e].to(dtype)

kernel = K._moe_bwd_dX_expanded_kernel.fn  # bypass autotuner

results = []
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
            BLOCK_M=block_m, USE_TMA=False,
            BLOCK_N=bn, BLOCK_K=bk, GROUP_M=gm, num_warps=nw, num_stages=ns,
        )
        torch.cuda.synchronize()
        err = (dxe - ref).abs().max().item()
        ok = err < 1e-2
        results.append((f"BN={bn} BK={bk} nw={nw} ns={ns}", "OK " if ok else "BAD", err))
    except Exception as ex:
        results.append((f"BN={bn} BK={bk} nw={nw} ns={ns}", "ERR", str(ex)[:120]))

for name, status, err in results:
    print(f"{status} {name}: {err}")
n_bad = sum(1 for _, s, _ in results if s != "OK ")
print(f"\n{n_bad}/{len(results)} configs bad")
