# Triton kernels for fused MoE expert computation.
#
# Routing metadata kernels (Kernels 1-3) are adapted from:
#   SonicMoE (https://github.com/linkedin/sonic-moe)
#   Copyright 2025 Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
#
# Grouped GEMM kernels and backward kernels are new Triton implementations
# inspired by the SonicMoE paper (arXiv:2512.14080), ported to portable Triton
# (no Hopper-specific WGMMA/TMA) for general GPU support.

import os

import triton
import triton.language as tl

# LIGER_FUSED_MOE_AUTOTUNE=0 pins each kernel to one config, skipping Triton's
# `do_bench` loop whose per-config working sets can OOM (see issue #1246). Must
# be set before importing liger_kernel. Temporary escape hatch until triton's
# autotuner handles such errors itself.
_AUTOTUNE_DISABLED = os.environ.get("LIGER_FUSED_MOE_AUTOTUNE", "1").lower() in (
    "0",
    "false",
    "no",
)

# ---------------------------------------------------------------------------
# Routing metadata overview
#
# Three kernels produce permutation arrays for grouped GEMM:
#
#   K1 — Histogram: count (token,k) assignments per expert per tile.
#   K2 — Prefix sums: convert tile counts to exclusive prefix sums;
#         compute expert_start_idx (token offsets) and tile offsets.
#   K3 — Scatter: sort by expert, assign globally sorted positions,
#         write x_gather_idx / s_scatter_idx / s_reverse_scatter_idx
#         and tile metadata (tile_row_start, tile_expert).
#
# GEMM kernels consume:
#   x_gather_idx          (TK,)   sorted_pos → original token index
#   s_scatter_idx         (TK,)   sorted_pos → flat (t,k) index
#   s_reverse_scatter_idx (TK,)   flat (t,k) → sorted_pos
#   expert_start_idx      (E+1,)  exclusive cumsum of tokens per expert
#   tile_row_start        (M,)    absolute row_start in sorted space per M-tile
#   tile_expert           (M,)    expert index per M-tile
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helper: associative combiner for the segmented scan in K3
# ---------------------------------------------------------------------------


@triton.jit
def _keyed_add(x, y):
    """Segment-aware addition for packed uint32 values (key << 16 | count).

    Used as the combine function of tl.associative_scan to compute per-expert
    run-lengths within a sorted tile.  The upper 16 bits carry the expert id
    (the segment key); the lower 16 bits carry the running count.

    Rule: if both operands belong to the same expert, add their counts;
    otherwise the right operand starts a new segment and its count wins.
    """
    key_mask: tl.constexpr = 0xFFFF0000
    kx = x & key_mask
    ky = y & key_mask
    # Same key → accumulate; different key → reset to right operand.
    z = tl.where(kx == ky, x + y - kx, y)
    return z


# ---------------------------------------------------------------------------
# Tiled histogram of expert token counts
# Adapted from sonic-moe _compute_col_partial_sum_kernel
# ---------------------------------------------------------------------------


@triton.jit
def _moe_router_histogram_kernel(
    topk_indices_ptr,  # (T, K) int32
    partial_sum_ptr,  # (E, n_tiles) int32 — output; partial_sum[e, tile] = count
    T,
    E: tl.constexpr,
    n_tiles,
    TOKENS_PER_TILE: tl.constexpr,
    K_POW2: tl.constexpr,
    K: tl.constexpr,
    E_POW2: tl.constexpr,
):
    """Count how many of this tile's (token, k) assignments route to each expert.

    Grid: (n_tiles,).  Each CTA owns one contiguous slice of TOKENS_PER_TILE
    tokens and atomically increments partial_sum[expert_id, tile_id] for every
    (token, k) pair it sees.

    partial_sum is stored row-major with shape (E, n_tiles) so that K2 can
    read each expert's column (partial_sum[e, :]) with a stride-1 access.
    """
    tile_id = tl.program_id(0)

    # Zero this tile's column before counting — partial_sum is not pre-cleared.
    e_offs = tl.arange(0, E_POW2)
    tl.store(
        partial_sum_ptr + e_offs * n_tiles + tile_id,
        tl.zeros([E_POW2], tl.int32),
        mask=e_offs < E,
    )

    # Load all (token, k) expert assignments for this tile as a 2-D block.
    # Using 2-D indexing avoids div/mod and is faster for non-power-of-2 K.
    tok_offs = tile_id * TOKENS_PER_TILE + tl.arange(0, TOKENS_PER_TILE)
    k_offs = tl.arange(0, K_POW2)
    tok_mask = tok_offs < T
    load_mask = tok_mask[:, None] & (k_offs[None, :] < K)
    safe_k = tl.minimum(k_offs, K - 1)  # clamp for out-of-bounds k slots
    expert_ids = tl.load(
        topk_indices_ptr + tok_offs[:, None] * K + safe_k[None, :],
        mask=load_mask,
        other=-1,
    )

    # Flatten and atomically histogram into partial_sum[:, tile_id].
    flat_experts = tl.reshape(expert_ids, [TOKENS_PER_TILE * K_POW2])
    flat_mask = tl.reshape(load_mask, [TOKENS_PER_TILE * K_POW2])
    safe_experts = tl.where(flat_mask, flat_experts, 0)  # redirect masked lanes to expert 0

    tl.atomic_add(
        partial_sum_ptr + safe_experts * n_tiles + tile_id,
        tl.full([TOKENS_PER_TILE * K_POW2], 1, dtype=tl.int32),
        mask=flat_mask,
    )


# ---------------------------------------------------------------------------
# Per-expert tile prefix sums + global token/tile offsets
# Adapted from sonic-moe _bitmatrix_metadata_compute_stage1
# ---------------------------------------------------------------------------


@triton.jit
def _moe_router_prefix_sum_kernel(
    expert_freq_ptr,  # (E,) int32 — total tokens assigned to each expert
    expert_freq_offs_ptr,  # (E+1,) int32 — output: exclusive cumsum of expert_frequency
    expert_tile_offset_ptr,  # (E+1,) int32 — output: exclusive cumsum of ceil(freq/BLOCK_M_TOKEN)
    E: tl.constexpr,
    partial_sum_ptr,  # (E, n_tiles) int32 — in-place: raw tile counts → tile prefix sums
    n_tiles,
    TK,  # T * K, written as sentinel into expert_freq_offs[E]
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M_TOKEN: tl.constexpr,
):
    """Convert histogram counts into prefix sums; compute token and tile offsets.

    Grid: (E+2,).  Three disjoint roles, all running concurrently:

    PIDs 0..E-1  — Per-expert tile prefix scan
        Each CTA converts its expert's row of partial_sum from raw tile
        counts into exclusive prefix sums across tiles.  After K3 reads
        partial_sum[e, tile_id], it knows how many of expert e's tokens
        appeared in earlier tiles, which it adds to within_expert_rank to
        get the global sorted position.

    PID E  — Global token and M-tile offset computation
        Sequentially scans all expert frequencies in blocks of BLOCK_N to
        build two exclusive cumsums in a single pass:
          expert_start_idx[e]    = sum of expert_frequency[0..e-1]
          expert_tile_offset[e]  = sum of ceil(freq[0..e-1] / BLOCK_M_TOKEN)
        Also writes the sentinel expert_tile_offset[E] = total M-tiles.

    PID E+1  — Token sentinel
        Writes expert_start_idx[E] = TK.
    """
    pid = tl.program_id(0)
    if pid < E:
        # Per-expert tile prefix scan: transform partial_sum[pid, :] from
        # raw counts to exclusive prefix sums for conflict-free output positions.
        expert_partial_sum_ptr = partial_sum_ptr + pid * n_tiles
        curr_sum = 0
        for start in range(0, n_tiles, BLOCK_M):
            offs = start + tl.arange(0, BLOCK_M)
            tile_counts = tl.load(expert_partial_sum_ptr + offs, mask=offs < n_tiles, other=0)
            excl_cumsum = tl.cumsum(tile_counts, 0) - tile_counts + curr_sum
            curr_sum += tl.sum(tile_counts, 0)
            tl.store(expert_partial_sum_ptr + offs, excl_cumsum, mask=offs < n_tiles)
    elif pid == E:
        # Global token and M-tile offsets (single sequential CTA).
        # Both expert_start_idx and expert_tile_offset are exclusive prefix sums
        # accumulated together to avoid a second pass.
        curr_freq_sum = 0
        curr_tile_sum = 0
        for start in tl.static_range(0, E, BLOCK_N):
            offs = start + tl.arange(0, BLOCK_N)
            expert_freq = tl.load(expert_freq_ptr + offs, mask=offs < E, other=0)

            excl_freq = tl.cumsum(expert_freq, 0) - expert_freq + curr_freq_sum
            curr_freq_sum += tl.sum(expert_freq, 0)
            tl.store(expert_freq_offs_ptr + offs, excl_freq, mask=offs < E)

            # Number of BLOCK_M_TOKEN-sized M-tiles needed for each expert.
            expert_m_tiles = (expert_freq + BLOCK_M_TOKEN - 1) // BLOCK_M_TOKEN
            excl_tile = tl.cumsum(expert_m_tiles, 0) - expert_m_tiles + curr_tile_sum
            curr_tile_sum += tl.sum(expert_m_tiles, 0)
            tl.store(expert_tile_offset_ptr + offs, excl_tile, mask=offs < E)

        # Write total M-tile count as the sentinel.
        tl.store(expert_tile_offset_ptr + E, curr_tile_sum)
    elif pid == E + 1:
        # Token sentinel: expert_start_idx[E] = TK.
        tl.store(expert_freq_offs_ptr + E, TK)


# ---------------------------------------------------------------------------
# Sort assignments by expert, compute output positions, emit tile metadata
# Adapted from sonic-moe _bitmatrix_metadata_compute_stage2
# ---------------------------------------------------------------------------


@triton.jit
def _moe_router_scatter_kernel(
    s_scatter_idx_ptr,  # (TK,) int32 — output: sorted_pos → flat (t,k) index
    s_reverse_scatter_idx_ptr,  # (TK,) int32 — output: flat (t,k) → sorted_pos
    x_gather_idx_ptr,  # (TK,) int32 — output: sorted_pos → token index t
    tile_row_start_ptr,  # (num_m_tiles,) int32 — output: absolute row_start per M-tile
    tile_expert_ptr,  # (num_m_tiles,) int32 — output: expert index per M-tile
    topk_indices_ptr,  # (T, K) int32
    T,
    partial_sum_ptr,  # (E, n_tiles) int32 — tile prefix sums from K2 (read-only here)
    n_tiles,
    expert_offs_ptr,  # (E,) int32 — expert_start_idx[0:E] from K2
    expert_tile_offset_ptr,  # (E,) int32 — expert_tile_offset[0:E] from K2
    K_POW2: tl.constexpr,
    K: tl.constexpr,
    TOKENS_PER_BLOCK: tl.constexpr,
    BLOCK_M_TOKEN: tl.constexpr,
):
    """Assign every (token, k) pair its globally-sorted output position.

    Grid: (n_tiles,). Each CTA packs assignments as (expert_id << 16 | local_offset),
    sorts within the tile, runs a segmented scan (_keyed_add) for within-expert ranks,
    then writes x_gather_idx, s_scatter_idx, s_reverse_scatter_idx, tile_row_start,
    and tile_expert.
    """
    BLOCK_SIZE: tl.constexpr = TOKENS_PER_BLOCK * K_POW2
    IS_POW2_K: tl.constexpr = K == K_POW2
    tl.static_assert(BLOCK_SIZE <= 32768)

    pid_m = tl.program_id(0)
    offs_local = tl.arange(0, BLOCK_SIZE)
    offs_global = pid_m * BLOCK_SIZE + offs_local
    mask = offs_global < T * K_POW2

    # Load expert ids and pack with local offsets for sorting.
    if IS_POW2_K:
        # Flat layout: topk_indices is already (T*K,) in row-major order.
        expert = tl.load(topk_indices_ptr + offs_global, mask=mask, other=-1).to(tl.uint32)
    else:
        # Non-power-of-2 K: reconstruct (token, k) from local offset via div/mod.
        token_i_local = offs_local // K_POW2
        k_slot = offs_local % K_POW2
        token_i_global = pid_m * TOKENS_PER_BLOCK + token_i_local
        load_mask = mask & (k_slot < K)
        safe_k = tl.minimum(k_slot, K - 1)
        expert = tl.load(
            topk_indices_ptr + token_i_global * K + safe_k,
            mask=load_mask,
            other=-1,
        ).to(tl.uint32)

    # Pack: upper 16 bits = expert_id (0xFFFF for padding/invalid),
    #       lower 16 bits = local offset (used to recover original position after sort).
    kv_pairs = tl.sort(((expert << 16) | offs_local).to(tl.uint32), 0)
    expert = kv_pairs >> 16
    mask = expert != 0xFFFF  # mask out padding entries introduced by K_POW2 rounding

    # Within-expert rank via segmented inclusive scan.
    scan_input = (kv_pairs & 0xFFFF0000) | 0x00000001
    inclusive_run_lengths = tl.associative_scan(scan_input, 0, _keyed_add)
    within_expert_rank = (inclusive_run_lengths - 1) & 0xFFFF  # convert to 0-based

    # Absolute sorted position.
    within_expert = tl.load(partial_sum_ptr + pid_m + expert * n_tiles, mask=mask, other=0) + within_expert_rank
    expert_start = tl.load(expert_offs_ptr + expert, mask=mask, other=0)
    s_reverse = expert_start + within_expert

    # Emit tile metadata for GEMM grid.
    is_tile_start = (within_expert % BLOCK_M_TOKEN) == 0
    t_within = within_expert // BLOCK_M_TOKEN
    tile_base = tl.load(
        expert_tile_offset_ptr + expert,
        mask=mask & is_tile_start,
        other=0,
    ).to(tl.int32)
    flat_tile_idx = tile_base + t_within
    tl.store(
        tile_row_start_ptr + flat_tile_idx,
        s_reverse.to(tl.int32),
        mask=mask & is_tile_start,
    )
    tl.store(tile_expert_ptr + flat_tile_idx, expert.to(tl.int32), mask=mask & is_tile_start)

    # Write permutation arrays.
    if IS_POW2_K:
        presort_offs = kv_pairs & 0xFFFF
        entry_idx = pid_m * BLOCK_SIZE + presort_offs  # flat (t, k) index in [0, TK)
        tl.store(s_reverse_scatter_idx_ptr + entry_idx, s_reverse, mask=mask)
        tl.store(s_scatter_idx_ptr + s_reverse, entry_idx, mask=mask)
        tl.store(x_gather_idx_ptr + s_reverse, entry_idx // K_POW2, mask=mask)
    else:
        presort_offs = kv_pairs & 0xFFFF
        token_i_global_s = pid_m * TOKENS_PER_BLOCK + presort_offs // K_POW2
        entry_idx = token_i_global_s * K + presort_offs % K_POW2
        tl.store(s_reverse_scatter_idx_ptr + entry_idx, s_reverse, mask=mask)
        tl.store(s_scatter_idx_ptr + s_reverse, entry_idx, mask=mask)
        tl.store(x_gather_idx_ptr + s_reverse, token_i_global_s, mask=mask)


# ---------------------------------------------------------------------------
# Shared autotune config for all GEMM kernels
# ---------------------------------------------------------------------------


def _get_gemm_autotune_configs():
    if _AUTOTUNE_DISABLED:
        return [triton.Config({"BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=2)]
    configs = []
    for bn in [64, 128]:
        for bk in [32, 64]:
            for nw in [4, 8]:
                for ns in [2, 3, 4, 5]:
                    configs.append(
                        triton.Config(
                            {"BLOCK_N": bn, "BLOCK_K": bk},
                            num_warps=nw,
                            num_stages=ns,
                        )
                    )
    return configs


def _get_dW_autotune_configs():
    """Configs for backward weight-grad kernels (dW1, dW2): include BLOCK_M sweep."""
    if _AUTOTUNE_DISABLED:
        return [
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32},
                num_warps=8,
                num_stages=2,
            )
        ]
    return [
        triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk}, num_warps=nw, num_stages=2)
        for bm in [64, 128]
        for bn in [64, 128]
        for bk in [16, 32]
        for nw in [4, 8]
    ]


# ---------------------------------------------------------------------------
# Forward — fused gather + grouped GEMM + SwiGLU
# 2D grid: (num_m_tiles, ceil(I/BLOCK_N))
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=_get_gemm_autotune_configs(),
    key=["H_dim", "I_dim"],
)
@triton.jit
def _fused_up_proj_swiglu_kernel(
    x_ptr,  # (T, H)
    gate_up_proj_ptr,  # (E, 2*I, H)
    x_gather_idx_ptr,  # (TK,) int32
    expert_start_ptr,  # (E+1,) int32
    tile_row_start_ptr,  # (num_m_tiles,) int32 — row_start per M-tile
    tile_expert_ptr,  # (num_m_tiles,) int32 — expert index per M-tile
    total_tiles_ptr,  # (1,) int32 — actual number of m-tiles (device scalar)
    pre_act_ptr,  # (TK, 2*I)  pre-SwiGLU activations [saved for backward]
    post_act_ptr,  # (TK, I)    post-SwiGLU activations
    H_dim: tl.constexpr,
    I_dim: tl.constexpr,
    stride_x_T,
    stride_x_H: tl.constexpr,
    stride_w_E,
    stride_w_N,
    stride_w_K: tl.constexpr,
    stride_pre_TK,
    stride_pre_N: tl.constexpr,
    stride_post_TK,
    stride_post_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Grid: (num_m_tiles_max, ceil(I/BLOCK_N)).
    pid_m selects M-tile via tile_row_start/tile_expert; pid_n selects N-tile.
    Grid dim0 is an upper bound; CTAs past the actual tile count exit early."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_m >= tl.load(total_tiles_ptr):
        return

    row_start = tl.load(tile_row_start_ptr + pid_m)
    # int64 prevents expert_idx * stride_w_E overflow at large E*I*H (see #1246).
    expert_idx = tl.load(tile_expert_ptr + pid_m).to(tl.int64)
    n_start = pid_n * BLOCK_N
    expert_end = tl.load(expert_start_ptr + expert_idx + 1)

    m_offs = tl.arange(0, BLOCK_M)
    n_offs = tl.arange(0, BLOCK_N)
    k_offs = tl.arange(0, BLOCK_K)

    # int64 prevents row_offs * stride overflow when TK is large (see #1246).
    row_offs = (row_start + m_offs).to(tl.int64)
    row_mask = row_offs < expert_end

    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    n_idx = n_start + n_offs
    n_mask = n_idx < I_dim
    # int64 prevents token_idx * stride_T overflow at large T*H (see #1246).
    token_idx = tl.load(x_gather_idx_ptr + row_offs, mask=row_mask, other=0).to(tl.int64)
    for k in tl.range(0, H_dim, BLOCK_K):
        k_idx = k + k_offs
        k_mask = k_idx < H_dim

        x_ptrs = x_ptr + token_idx[:, None] * stride_x_T + k_idx[None, :] * stride_x_H
        # Keep bf16 for dot operands → tensor cores. acc stays fp32 for precision.
        x_tile = tl.load(
            x_ptrs,
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
            eviction_policy="evict_first",  # token rows not reused; free L2 for weights
        )

        w_mask = n_mask[:, None] & k_mask[None, :]
        w_gate_ptrs = (
            gate_up_proj_ptr + expert_idx * stride_w_E + n_idx[:, None] * stride_w_N + k_idx[None, :] * stride_w_K
        )
        w_gate = tl.load(
            w_gate_ptrs,
            mask=w_mask,
            other=0.0,
        )
        acc_gate = tl.dot(x_tile, tl.trans(w_gate), acc=acc_gate)

        w_up_ptrs = w_gate_ptrs + I_dim * stride_w_N
        w_up = tl.load(
            w_up_ptrs,
            mask=w_mask,
            other=0.0,
        )

        acc_up = tl.dot(x_tile, tl.trans(w_up), acc=acc_up)

    out_mask = row_mask[:, None] & n_mask[None, :]

    pre_gate_ptrs = pre_act_ptr + row_offs[:, None] * stride_pre_TK + n_idx[None, :] * stride_pre_N
    pre_up_ptrs = pre_gate_ptrs + I_dim * stride_pre_N
    tl.store(pre_gate_ptrs, acc_gate.to(pre_act_ptr.dtype.element_ty), mask=out_mask)
    tl.store(pre_up_ptrs, acc_up.to(pre_act_ptr.dtype.element_ty), mask=out_mask)

    sig_gate = tl.sigmoid(acc_gate)
    silu_gate = acc_gate * sig_gate
    a_out = silu_gate * acc_up

    post_ptrs = post_act_ptr + row_offs[:, None] * stride_post_TK + n_idx[None, :] * stride_post_N
    tl.store(post_ptrs, a_out.to(post_act_ptr.dtype.element_ty), mask=out_mask)


# ---------------------------------------------------------------------------
# Forward — grouped GEMM down-projection
# 2D grid: (num_m_tiles, ceil(H/BLOCK_N))
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=_get_gemm_autotune_configs(),
    key=["H_dim", "I_dim"],
)
@triton.jit
def _fused_down_proj_kernel(
    post_act_ptr,  # (TK, I)
    down_proj_ptr,  # (E, H, I)
    expert_start_ptr,  # (E+1,) int32
    tile_row_start_ptr,  # (num_m_tiles,) int32
    tile_expert_ptr,  # (num_m_tiles,) int32
    total_tiles_ptr,  # (1,) int32 — actual number of m-tiles (device scalar)
    Y_ptr,  # (TK, H)
    H_dim: tl.constexpr,
    I_dim: tl.constexpr,
    stride_post_TK,
    stride_post_I: tl.constexpr,
    stride_w_E,
    stride_w_H,
    stride_w_I: tl.constexpr,
    stride_Y_TK,
    stride_Y_H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Grid: (num_m_tiles_max, ceil(H/BLOCK_N)).
    Each CTA: one (BLOCK_M, BLOCK_N) tile of Y = post_act @ down_proj[e]^T."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_m >= tl.load(total_tiles_ptr):
        return

    row_start = tl.load(tile_row_start_ptr + pid_m)
    # int64 prevents expert_idx * stride_w_E overflow at large E*I*H (see #1246).
    expert_idx = tl.load(tile_expert_ptr + pid_m).to(tl.int64)
    n_start = pid_n * BLOCK_N
    expert_end = tl.load(expert_start_ptr + expert_idx + 1)

    m_offs = tl.arange(0, BLOCK_M)
    n_offs = tl.arange(0, BLOCK_N)
    k_offs = tl.arange(0, BLOCK_K)

    # int64 prevents row_offs * stride overflow when TK is large (see #1246).
    row_offs = (row_start + m_offs).to(tl.int64)
    row_mask = row_offs < expert_end
    n_idx = n_start + n_offs
    n_mask = n_idx < H_dim

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in tl.range(0, I_dim, BLOCK_K):
        k_idx = k + k_offs
        k_mask = k_idx < I_dim

        a_ptrs = post_act_ptr + row_offs[:, None] * stride_post_TK + k_idx[None, :] * stride_post_I
        # Keep bf16 for dot operands → tensor cores. acc stays fp32.
        a_tile = tl.load(a_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0)

        w_ptrs = down_proj_ptr + expert_idx * stride_w_E + n_idx[:, None] * stride_w_H + k_idx[None, :] * stride_w_I
        w_tile = tl.load(
            w_ptrs,
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        )

        acc = tl.dot(a_tile, tl.trans(w_tile), acc=acc)

    Y_ptrs = Y_ptr + row_offs[:, None] * stride_Y_TK + n_idx[None, :] * stride_Y_H
    tl.store(Y_ptrs, acc.to(Y_ptr.dtype.element_ty), mask=row_mask[:, None] & n_mask[None, :])


# ---------------------------------------------------------------------------
# Forward — token gather + weighted sum
# Adapted from sonic-moe token_gather_sum_kernel
# ---------------------------------------------------------------------------


def _get_token_gather_autotune_configs():
    if _AUTOTUNE_DISABLED:
        return [triton.Config({"BLOCK_H": 128, "BLOCK_K": 4}, num_warps=4, num_stages=4)]
    configs = []
    for bh in [64, 128, 256, 512]:
        for bk in [1, 2, 4, 8, 16]:
            for nw in [4, 8]:
                if bk * bh <= 32768:
                    configs.append(triton.Config({"BLOCK_H": bh, "BLOCK_K": bk}, num_warps=nw, num_stages=4))
    return configs


@triton.autotune(
    configs=_get_token_gather_autotune_configs(),
    key=["H_dim", "K_dim", "w_is_None"],
)
@triton.jit
def _token_gather_weighted_sum_kernel(
    Y_ptr,  # (TK, H)
    w_ptr,  # (TK,) routing weights, or None when w_is_None=True
    s_rev_ptr,  # (TK,) int32 s_reverse_scatter_idx: flat(t,k) → sorted position
    out_ptr,  # (T, H)
    H_dim: tl.constexpr,
    K_dim: tl.constexpr,
    stride_Y_TK,
    stride_Y_H: tl.constexpr,
    stride_out_T,
    stride_out_H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    w_is_None: tl.constexpr,  # True → unweighted gather-sum (used for dx backward)
):
    """One CTA per token. Gathers K expert outputs, reduces with routing weights
    (forward) or without weights (backward dx via _token_broadcast_backward)."""
    # int64 prevents t * stride_out_T overflow at large T*H (see #1246).
    t = tl.program_id(0).to(tl.int64)

    for h_tile in tl.static_range(triton.cdiv(H_dim, BLOCK_H)):
        h_idx = (h_tile * BLOCK_H + tl.arange(0, BLOCK_H)).to(tl.uint32)
        h_mask = h_idx < H_dim
        acc = tl.zeros([BLOCK_H], dtype=tl.float32)

        for k_tile in tl.range(triton.cdiv(K_dim, BLOCK_K)):
            k_offs = (k_tile * BLOCK_K + tl.arange(0, BLOCK_K)).to(tl.uint32)
            k_mask = k_offs < K_dim

            flat_idx = t * K_dim + k_offs
            # int64 prevents perm_idx * stride overflow when TK is large (see #1246).
            perm_idx = tl.load(s_rev_ptr + flat_idx, mask=k_mask, other=0).to(tl.int64)

            y_ptrs = Y_ptr + perm_idx[:, None] * stride_Y_TK + h_idx[None, :] * stride_Y_H
            y_vals = tl.load(y_ptrs, mask=k_mask[:, None] & h_mask[None, :], other=0.0).to(tl.float32)

            if w_is_None:
                acc += tl.sum(y_vals, axis=0)
            else:
                w_vals = tl.load(w_ptr + flat_idx, mask=k_mask, other=0.0).to(tl.float32)
                acc += tl.sum(y_vals * w_vals[:, None], axis=0)

        out_ptrs = out_ptr + t * stride_out_T + h_idx * stride_out_H
        tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=h_mask)


# ---------------------------------------------------------------------------
# Backward — fused down-proj backward + SwiGLU backward
# 2D grid: (num_m_tiles, ceil(I/BLOCK_N))
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=_get_gemm_autotune_configs(),
    key=["H_dim", "I_dim"],
    reset_to_zero=["dS_ptr"],  # autotune runs multiple configs; atomic_add accumulates, so reset between runs
)
@triton.jit
def _moe_bwd_down_proj_kernel(
    dO_ptr,  # (T, H)   — ∂L/∂O, upstream gradient
    x_gather_idx_ptr,  # (TK,)    — σ_x: sorted_pos → original token index
    s_scatter_idx_ptr,  # (TK,)    — σ_s: sorted_pos → flat (t,k) index
    topk_weights_ptr,  # (TK,)    — s_k: routing weights in flat (t,k) order
    down_proj_ptr,  # (E, H, I) — W2
    pre_act_ptr,  # (TK, 2I) — z = [gate, up] saved from forward
    expert_start_ptr,  # (E+1,)   int32
    tile_row_start_ptr,  # (num_m_tiles,) int32
    tile_expert_ptr,  # (num_m_tiles,) int32
    total_tiles_ptr,  # (1,) int32 — actual number of m-tiles (device scalar)
    d_pre_act_ptr,  # (TK, 2I) — output: ∂L/∂z = [dgate, dup]
    weighted_act_ptr,  # (TK, I)  — output: s_k * y1 (for dW2 kernel)
    dS_ptr,  # (TK,)    — output: ∂L/∂s_k, indexed by flat (t,k)
    H_dim: tl.constexpr,
    I_dim: tl.constexpr,
    stride_dO_T,
    stride_dO_H: tl.constexpr,
    stride_w_E,
    stride_w_H,
    stride_w_I: tl.constexpr,
    stride_pre_TK,
    stride_pre_N: tl.constexpr,
    stride_d_pre_TK,
    stride_d_pre_N: tl.constexpr,
    stride_wact_TK,
    stride_wact_I: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Grid: (num_m_tiles_max, ceil(I/BLOCK_N)).
    Accumulates dA' = dO @ W2^T (dO stays in registers), recomputes y1 from
    pre_act, applies SwiGLU backward, writes d_pre_act, weighted_act, and dS."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_m >= tl.load(total_tiles_ptr):
        return

    row_start = tl.load(tile_row_start_ptr + pid_m)
    # int64 prevents expert_idx * stride_w_E overflow at large E*I*H (see #1246).
    expert_idx = tl.load(tile_expert_ptr + pid_m).to(tl.int64)
    n_start = pid_n * BLOCK_N
    expert_end = tl.load(expert_start_ptr + expert_idx + 1)

    m_offs = tl.arange(0, BLOCK_M)
    n_offs = tl.arange(0, BLOCK_N)
    k_offs = tl.arange(0, BLOCK_K)

    # int64 prevents row_offs * stride overflow when TK is large (see #1246).
    row_offs = (row_start + m_offs).to(tl.int64)
    row_mask = row_offs < expert_end
    n_idx = n_start + n_offs
    n_mask = n_idx < I_dim
    out_mask = row_mask[:, None] & n_mask[None, :]

    # Hoist per-row routing metadata (constant across H K-loop).
    # int64 prevents token_idx * stride_T overflow at large T*H (see #1246).
    token_idx = tl.load(x_gather_idx_ptr + row_offs, mask=row_mask, other=0).to(tl.int64)
    flat_tk_idx = tl.load(s_scatter_idx_ptr + row_offs, mask=row_mask, other=0)
    weights = tl.load(topk_weights_ptr + flat_tk_idx, mask=row_mask, other=0.0).to(tl.float32)

    # K-loop: accumulate dA' = dO @ W2^T (unscaled; scale once after loop).
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, H_dim, BLOCK_K):
        k_idx = k + k_offs
        k_mask = k_idx < H_dim

        dO_ptrs = dO_ptr + token_idx[:, None] * stride_dO_T + k_idx[None, :] * stride_dO_H
        dO_tile = tl.load(dO_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0)

        w_ptrs = down_proj_ptr + expert_idx * stride_w_E + k_idx[:, None] * stride_w_H + n_idx[None, :] * stride_w_I
        w_tile = tl.load(w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        acc = tl.dot(dO_tile, w_tile, acc=acc)

    # Epilogue: recompute y1 = silu(gate) * up from saved pre_act.
    # These loads need fp32 for the sigmoid/silu computation.
    gate_ptrs = pre_act_ptr + row_offs[:, None] * stride_pre_TK + n_idx[None, :] * stride_pre_N
    up_ptrs = gate_ptrs + I_dim * stride_pre_N
    gate = tl.load(gate_ptrs, mask=out_mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptrs, mask=out_mask, other=0.0).to(tl.float32)
    sig_gate = tl.sigmoid(gate)
    silu_gate = gate * sig_gate
    y1 = silu_gate * up  # (BLOCK_M, BLOCK_N)

    # Write weighted_act = s_k * y1 for dW2.
    wact_ptrs = weighted_act_ptr + row_offs[:, None] * stride_wact_TK + n_idx[None, :] * stride_wact_I
    tl.store(
        wact_ptrs,
        (weights[:, None] * y1).to(weighted_act_ptr.dtype.element_ty),
        mask=out_mask,
    )

    # dS: ∂L/∂s_k = sum_I((dO @ W2^T) * y1) — accumulate across all N-tiles.
    # IMPORTANT: use atomic_add, not store — the grid has ceil(I/BLOCK_N) N-tiles per
    # M-tile, each contributing a partial sum over its I-chunk.  tl.store would
    # overwrite previous tiles, leaving only the last chunk's contribution.
    dS_partial = tl.sum(acc * y1, axis=1)
    tl.atomic_add(dS_ptr + flat_tk_idx, dS_partial, mask=row_mask)

    # Scale once: dA' = s_k * (dO @ W2^T)
    acc = acc * weights[:, None]

    # SwiGLU backward: dgate = d_silu(gate) * up * dA', dup = silu(gate) * dA'.
    dgate = acc * (silu_gate * (1.0 - sig_gate) + sig_gate) * up
    dup = acc * silu_gate
    dgate_ptrs = d_pre_act_ptr + row_offs[:, None] * stride_d_pre_TK + n_idx[None, :] * stride_d_pre_N
    dup_ptrs = dgate_ptrs + I_dim * stride_d_pre_N
    tl.store(dgate_ptrs, dgate.to(d_pre_act_ptr.dtype.element_ty), mask=out_mask)
    tl.store(dup_ptrs, dup.to(d_pre_act_ptr.dtype.element_ty), mask=out_mask)


# ---------------------------------------------------------------------------
# Backward — dW2 = weighted_act^T @ dout_gathered (per expert, weight grad)
# Lambda grid: (E * ceil(I/BLOCK_M), ceil(H/BLOCK_N))
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=_get_dW_autotune_configs(),
    key=["H_dim", "I_dim"],
)
@triton.jit
def _moe_bwd_dW2_kernel(
    weighted_act_ptr,  # (TK, I) — s_k * y1 from backward down-proj kernel
    dout_ptr,  # (T, H)  — upstream gradient (gathered by x_gather_idx)
    x_gather_idx_ptr,  # (TK,)   — sorted_pos → original token index
    expert_start_ptr,  # (E+1,)  int32
    dW2_ptr,  # (E, H, I) — output
    H_dim: tl.constexpr,
    I_dim: tl.constexpr,
    stride_wact_TK,
    stride_wact_I: tl.constexpr,
    stride_dout_T,
    stride_dout_H: tl.constexpr,
    stride_dW2_E,
    stride_dW2_H,
    stride_dW2_I: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """dW2[e, h, i] = sum_t weighted_act[t, i] * dout[token(t), h] for tokens in e.
    Grid: (E * ceil(I/BLOCK_M), ceil(H/BLOCK_N)). Empty experts store zeros (no
    separate memset of dW2 needed — every output element is written exactly once)."""
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    N_M_TILES: tl.constexpr = (I_dim + BLOCK_M - 1) // BLOCK_M
    # int64 prevents expert_idx * stride_dW_E overflow at large E*I*H (see #1246).
    expert_idx = (pid0 // N_M_TILES).to(tl.int64)
    m_tile = pid0 % N_M_TILES

    expert_start = tl.load(expert_start_ptr + expert_idx)
    expert_end = tl.load(expert_start_ptr + expert_idx + 1)
    M_e = expert_end - expert_start

    m_start = m_tile * BLOCK_M
    n_start = pid1 * BLOCK_N

    m_offs = tl.arange(0, BLOCK_M)
    n_offs = tl.arange(0, BLOCK_N)
    k_offs = tl.arange(0, BLOCK_K)

    i_idx = m_start + m_offs
    h_idx = n_start + n_offs
    i_mask = i_idx < I_dim
    h_mask = h_idx < H_dim

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in tl.range(0, M_e, BLOCK_K):
        k_idx = k + k_offs
        k_mask = k_idx < M_e
        row_offs = (expert_start + k_idx).to(tl.int64)

        wact_ptrs = weighted_act_ptr + row_offs[None, :] * stride_wact_TK + i_idx[:, None] * stride_wact_I
        wact_tile = tl.load(wact_ptrs, mask=k_mask[None, :] & i_mask[:, None], other=0.0)

        # int64 prevents token_idx * stride_T overflow at large T*H (see #1246).
        token_idx = tl.load(x_gather_idx_ptr + row_offs, mask=k_mask, other=0).to(tl.int64)
        dout_ptrs = dout_ptr + token_idx[:, None] * stride_dout_T + h_idx[None, :] * stride_dout_H
        dout_tile = tl.load(dout_ptrs, mask=k_mask[:, None] & h_mask[None, :], other=0.0)

        acc = tl.dot(wact_tile, dout_tile, acc=acc)

    dW2_ptrs = dW2_ptr + expert_idx * stride_dW2_E + h_idx[None, :] * stride_dW2_H + i_idx[:, None] * stride_dW2_I
    tl.store(
        dW2_ptrs,
        acc.to(dW2_ptr.dtype.element_ty),
        mask=i_mask[:, None] & h_mask[None, :],
    )


# ---------------------------------------------------------------------------
# Backward — dx_expanded = d_pre_act @ W1^T (grouped GEMM, no atomics)
# 2D grid: (num_m_tiles, ceil(H/BLOCK_N))
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=_get_gemm_autotune_configs(),
    key=["H_dim", "I_dim"],
)
@triton.jit
def _moe_bwd_dX_expanded_kernel(
    d_pre_act_ptr,  # (TK, 2*I)
    gate_up_proj_ptr,  # (E, 2*I, H) — W1
    expert_start_ptr,  # (E+1,) int32
    tile_row_start_ptr,  # (num_m_tiles,) int32
    tile_expert_ptr,  # (num_m_tiles,) int32
    total_tiles_ptr,  # (1,) int32 — actual number of m-tiles (device scalar)
    dx_expanded_ptr,  # (TK, H) — output: clean write, indexed by sorted_pos
    H_dim: tl.constexpr,
    I_dim: tl.constexpr,
    stride_d_pre_TK,
    stride_d_pre_N: tl.constexpr,
    stride_w_E,
    stride_w_N,
    stride_w_K: tl.constexpr,
    stride_dxe_TK,
    stride_dxe_H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Grid: (num_m_tiles_max, ceil(H/BLOCK_N)).
    dx_expanded[sorted_pos] = d_gate @ W1_gate^T + d_up @ W1_up^T.
    No atomics — rows are unique per CTA in sorted space."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_m >= tl.load(total_tiles_ptr):
        return

    row_start = tl.load(tile_row_start_ptr + pid_m)
    # int64 prevents expert_idx * stride_w_E overflow at large E*I*H (see #1246).
    expert_idx = tl.load(tile_expert_ptr + pid_m).to(tl.int64)
    n_start = pid_n * BLOCK_N
    expert_end = tl.load(expert_start_ptr + expert_idx + 1)

    m_offs = tl.arange(0, BLOCK_M)
    n_offs = tl.arange(0, BLOCK_N)
    k_offs = tl.arange(0, BLOCK_K)

    # int64 prevents row_offs * stride overflow when TK is large (see #1246).
    row_offs = (row_start + m_offs).to(tl.int64)
    row_mask = row_offs < expert_end
    h_idx = n_start + n_offs
    h_mask = h_idx < H_dim

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in tl.range(0, I_dim, BLOCK_K):
        k_idx = k + k_offs
        k_mask = k_idx < I_dim

        d_gate_ptrs = d_pre_act_ptr + row_offs[:, None] * stride_d_pre_TK + k_idx[None, :] * stride_d_pre_N
        d_gate = tl.load(d_gate_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0)

        w_gate_ptrs = (
            gate_up_proj_ptr + expert_idx * stride_w_E + k_idx[:, None] * stride_w_N + h_idx[None, :] * stride_w_K
        )
        w_gate = tl.load(w_gate_ptrs, mask=k_mask[:, None] & h_mask[None, :], other=0.0)
        acc = tl.dot(d_gate, w_gate, acc=acc)

        d_up_ptrs = d_pre_act_ptr + row_offs[:, None] * stride_d_pre_TK + (I_dim + k_idx)[None, :] * stride_d_pre_N
        d_up = tl.load(d_up_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0)

        w_up_ptrs = (
            gate_up_proj_ptr
            + expert_idx * stride_w_E
            + (I_dim + k_idx)[:, None] * stride_w_N
            + h_idx[None, :] * stride_w_K
        )
        w_up = tl.load(w_up_ptrs, mask=k_mask[:, None] & h_mask[None, :], other=0.0)

        acc = tl.dot(d_up, w_up, acc=acc)

    dxe_ptrs = dx_expanded_ptr + row_offs[:, None] * stride_dxe_TK + h_idx[None, :] * stride_dxe_H
    tl.store(
        dxe_ptrs,
        acc.to(dx_expanded_ptr.dtype.element_ty),
        mask=row_mask[:, None] & h_mask[None, :],
    )


# ---------------------------------------------------------------------------
# Backward — dW1 = Gathered_X^T @ d_pre_act (per expert, weight grad)
# Lambda grid: (E * ceil(H/BLOCK_M), ceil(2I/BLOCK_N))
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=_get_dW_autotune_configs(),
    key=["H_dim", "I_dim"],
)
@triton.jit
def _moe_bwd_dW1_kernel(
    x_ptr,  # (T, H)
    d_pre_act_ptr,  # (TK, 2*I)
    x_gather_idx_ptr,  # (TK,) int32
    expert_start_ptr,  # (E+1,) int32
    dW1_ptr,  # (E, 2*I, H) — output
    H_dim: tl.constexpr,
    I_dim: tl.constexpr,
    stride_x_T,
    stride_x_H: tl.constexpr,
    stride_d_pre_TK,
    stride_d_pre_N: tl.constexpr,
    stride_dW1_E,
    stride_dW1_N,
    stride_dW1_H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """dW1[e, n, h] = sum_t X[token(t), h] * d_pre_act[t, n], where n in [0, 2I).
    Grid: (E * ceil(H/BLOCK_M), ceil(2I/BLOCK_N)). Empty experts store zeros (no
    separate memset of dW1 needed — every output element is written exactly once)."""
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    N_M_TILES: tl.constexpr = (H_dim + BLOCK_M - 1) // BLOCK_M
    # int64 prevents expert_idx * stride_dW_E overflow at large E*I*H (see #1246).
    expert_idx = (pid0 // N_M_TILES).to(tl.int64)
    m_tile = pid0 % N_M_TILES

    expert_start = tl.load(expert_start_ptr + expert_idx)
    expert_end = tl.load(expert_start_ptr + expert_idx + 1)
    M_e = expert_end - expert_start

    m_start = m_tile * BLOCK_M
    n_start = pid1 * BLOCK_N

    m_offs = tl.arange(0, BLOCK_M)
    n_offs = tl.arange(0, BLOCK_N)
    k_offs = tl.arange(0, BLOCK_K)

    h_idx = m_start + m_offs
    n_idx = n_start + n_offs
    h_mask = h_idx < H_dim
    n_mask = n_idx < 2 * I_dim

    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    for k in tl.range(0, M_e, BLOCK_K):
        k_idx = k + k_offs
        k_mask = k_idx < M_e
        row_offs = (expert_start + k_idx).to(tl.int64)

        # int64 prevents token_idx * stride_T overflow at large T*H (see #1246).
        token_idx = tl.load(x_gather_idx_ptr + row_offs, mask=k_mask, other=0).to(tl.int64)
        x_ptrs = x_ptr + token_idx[:, None] * stride_x_T + h_idx[None, :] * stride_x_H
        x_tile = tl.load(x_ptrs, mask=k_mask[:, None] & h_mask[None, :], other=0.0)

        d_pre_ptrs = d_pre_act_ptr + row_offs[:, None] * stride_d_pre_TK + n_idx[None, :] * stride_d_pre_N
        d_pre_tile = tl.load(d_pre_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

        acc = tl.dot(tl.trans(d_pre_tile), x_tile, acc=acc)

    dW1_ptrs = dW1_ptr + expert_idx * stride_dW1_E + n_idx[:, None] * stride_dW1_N + h_idx[None, :] * stride_dW1_H
    tl.store(
        dW1_ptrs,
        acc.to(dW1_ptr.dtype.element_ty),
        mask=n_mask[:, None] & h_mask[None, :],
    )


# ===========================================================================
# ===== autograd wrapper (merged from fused_moe.py) =====
# ===========================================================================

# MERGED OPS SECTION
"""
Fused MoE expert computation via Triton grouped GEMM.

Forward: routing metadata (3 kernels) → fused gather+GEMM+SwiGLU → down-proj → token aggregation
Backward: memory-efficient — recomputes dA' = dO@W2^T to avoid caching Y (TK×H bytes)
"""

import torch
import triton

from liger_kernel.ops.utils import ensure_contiguous

# Token-dimension tile size for M.
# Not in the inner-loop autotune because tile_row_start/tile_expert and the
# grid dim-0 (num_m_tiles) must be recomputed for every candidate value.
# To tune: change this constant and re-run benchmarks.
BLOCK_M_TOKEN = 64


# ---------------------------------------------------------------------------
# Routing metadata
# ---------------------------------------------------------------------------


def compute_routing_metadata(topk_indices: torch.Tensor, E: int, block_m_token: int = BLOCK_M_TOKEN):
    """Compute token→expert routing permutation metadata via 3 Triton kernels.

    Also computes GPU tile metadata (tile_row_start, tile_expert) inside
    Kernel 3 — no CPU loop, one .item() sync for num_m_tiles allocation.

    Args:
        topk_indices:  (T, K) int32 — pre-computed top-k expert indices per token
        E:             number of experts
        block_m_token: BLOCK_M for token-dimension tiling (default BLOCK_M_TOKEN)

    Returns:
        expert_token_count:     (E,)            int32
        expert_start_idx:       (E+1,)          int32
        x_gather_idx:           (TK,)           int32
        s_scatter_idx:          (TK,)           int32
        s_reverse_scatter_idx:  (TK,)           int32
        tile_row_start:         (num_m_tiles,)  int32 — absolute row_start per M-tile
        tile_expert:            (num_m_tiles,)  int32 — expert index per M-tile
    """
    T, K = topk_indices.shape
    TK = T * K
    device = topk_indices.device
    E_POW2 = triton.next_power_of_2(E)
    K_POW2 = triton.next_power_of_2(K)
    TOKENS_PER_BLOCK = max(1, 1024 // K_POW2)
    n_tiles = triton.cdiv(T, TOKENS_PER_BLOCK)

    # Kernel 1: tiled histogram → tile_expert_counts (E, n_tiles)
    tile_expert_counts = torch.empty(E, n_tiles, dtype=torch.int32, device=device)
    _moe_router_histogram_kernel[(n_tiles,)](
        topk_indices,
        tile_expert_counts,
        T,
        E=E,
        n_tiles=n_tiles,
        TOKENS_PER_TILE=TOKENS_PER_BLOCK,
        K_POW2=K_POW2,
        K=K,
        E_POW2=E_POW2,
    )

    expert_token_count = tile_expert_counts.sum(dim=1, dtype=torch.int32)  # (E,)

    # Kernel 2: prefix sums + expert offsets + tile offsets (all in one pass)
    expert_start_idx = torch.empty(E + 1, dtype=torch.int32, device=device)
    expert_tile_offset = torch.empty(E + 1, dtype=torch.int32, device=device)
    _moe_router_prefix_sum_kernel[(E + 2,)](
        expert_token_count,
        expert_start_idx,
        expert_tile_offset,
        E=E,
        partial_sum_ptr=tile_expert_counts,
        n_tiles=n_tiles,
        TK=TK,
        BLOCK_M=128,
        BLOCK_N=E_POW2,
        BLOCK_M_TOKEN=block_m_token,
    )

    # No host sync: allocate tile metadata at the worst-case bound and let GEMM
    # CTAs past the actual count (expert_tile_offset[E], read on device) exit early.
    # Bound: sum_e ceil(f_e / B) <= floor(TK / B) + #nonempty_experts <= TK//B + min(E, TK).
    num_m_tiles_max = TK // block_m_token + min(E, TK)

    tile_row_start = torch.empty(num_m_tiles_max, dtype=torch.int32, device=device)
    tile_expert = torch.empty(num_m_tiles_max, dtype=torch.int32, device=device)

    # Kernel 3: sort by expert + scatter permutation arrays + tile metadata
    s_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    s_reverse_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    x_gather_idx = torch.empty(TK, dtype=torch.int32, device=device)

    if TK > 0:
        _moe_router_scatter_kernel[(n_tiles,)](
            s_scatter_idx,
            s_reverse_scatter_idx,
            x_gather_idx,
            tile_row_start,
            tile_expert,
            topk_indices,
            T,
            tile_expert_counts,  # non-contiguous (E, n_tiles) view
            n_tiles,
            expert_start_idx[:E],  # E entries (without TK sentinel)
            expert_tile_offset[:E],  # E entries of cumulative tile counts
            K_POW2=K_POW2,
            K=K,
            TOKENS_PER_BLOCK=TOKENS_PER_BLOCK,
            BLOCK_M_TOKEN=block_m_token,
        )

    return (
        expert_token_count,
        expert_start_idx,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        tile_row_start,
        tile_expert,
        expert_tile_offset,
    )


def _token_aggregation(Y, topk_weights_flat, s_reverse_scatter_idx, T, K, H):
    """Weighted gather-sum: out[t] = sum_k w[t,k] * Y[s_rev[t*K+k]]."""
    out = torch.empty(T, H, dtype=Y.dtype, device=Y.device)
    _token_gather_weighted_sum_kernel[(T,)](
        Y,
        topk_weights_flat,
        s_reverse_scatter_idx,
        out,
        H_dim=H,
        K_dim=K,
        stride_Y_TK=Y.stride(0),
        stride_Y_H=Y.stride(1),
        stride_out_T=out.stride(0),
        stride_out_H=out.stride(1),
        w_is_None=False,
    )
    return out


# ---------------------------------------------------------------------------
# Autograd Function
# ---------------------------------------------------------------------------


class LigerFusedMoEFunction(torch.autograd.Function):
    """Fused grouped GEMM MoE forward + memory-efficient backward.

    Forward: routing metadata → fused gather+GEMM+SwiGLU → down-proj → token aggregation
    Backward: avoids caching Y (TK×H) by recomputing dA' = dO@W2^T in backward

    Troubleshooting:
        If Triton's autotune ``do_bench`` loop OOMs (each config holds its own
        working set — see issue #1246), set ``LIGER_FUSED_MOE_AUTOTUNE=0`` before
        importing liger_kernel to pin each kernel to a single config and skip the
        benchmark loop. Temporary escape hatch until triton's autotuner handles
        such errors itself.
    """

    @staticmethod
    @ensure_contiguous
    def forward(ctx, x, gate_up_proj, down_proj, top_k_index, top_k_weights):
        """
        Args:
            x:             (T, H)      input tokens
            gate_up_proj:  (E, 2*intermediate_dim, H) gate+up projection weights
            down_proj:     (E, H, intermediate_dim)   down projection weights
            top_k_index:   (T, K) int32 — pre-computed routing indices
            top_k_weights: (T, K) float — pre-computed routing scores
        Returns:
            output: (T, H)
        """
        T, K = top_k_index.shape
        E = gate_up_proj.shape[0]
        H = x.shape[1]
        intermediate_dim = gate_up_proj.shape[1] // 2
        TK = T * K

        with torch.no_grad():
            (
                _,
                expert_start_idx,
                x_gather_idx,
                s_scatter_idx,
                s_reverse_scatter_idx,
                tile_row_start,
                tile_expert,
                expert_tile_offset,
            ) = compute_routing_metadata(top_k_index, E)

        num_m_tiles = tile_row_start.shape[0]  # upper bound; actual count lives on device
        total_tiles_dev = expert_tile_offset[E:]

        pre_act = torch.empty(TK, 2 * intermediate_dim, dtype=x.dtype, device=x.device)
        post_act = torch.empty(TK, intermediate_dim, dtype=x.dtype, device=x.device)

        if num_m_tiles > 0:
            _fused_up_proj_swiglu_kernel[
                lambda meta: (
                    num_m_tiles,
                    triton.cdiv(intermediate_dim, meta["BLOCK_N"]),
                )
            ](
                x,
                gate_up_proj,
                x_gather_idx,
                expert_start_idx,
                tile_row_start,
                tile_expert,
                total_tiles_dev,
                pre_act,
                post_act,
                H_dim=H,
                I_dim=intermediate_dim,
                stride_x_T=x.stride(0),
                stride_x_H=x.stride(1),
                stride_w_E=gate_up_proj.stride(0),
                stride_w_N=gate_up_proj.stride(1),
                stride_w_K=gate_up_proj.stride(2),
                stride_pre_TK=pre_act.stride(0),
                stride_pre_N=pre_act.stride(1),
                stride_post_TK=post_act.stride(0),
                stride_post_N=post_act.stride(1),
                BLOCK_M=BLOCK_M_TOKEN,
            )

        Y = torch.empty(TK, H, dtype=x.dtype, device=x.device)

        if num_m_tiles > 0:
            _fused_down_proj_kernel[lambda meta: (num_m_tiles, triton.cdiv(H, meta["BLOCK_N"]))](
                post_act,
                down_proj,
                expert_start_idx,
                tile_row_start,
                tile_expert,
                total_tiles_dev,
                Y,
                H_dim=H,
                I_dim=intermediate_dim,
                stride_post_TK=post_act.stride(0),
                stride_post_I=post_act.stride(1),
                stride_w_E=down_proj.stride(0),
                stride_w_H=down_proj.stride(1),
                stride_w_I=down_proj.stride(2),
                stride_Y_TK=Y.stride(0),
                stride_Y_H=Y.stride(1),
                BLOCK_M=BLOCK_M_TOKEN,
            )

        topk_weights_flat = top_k_weights.flatten().contiguous()
        out = _token_aggregation(Y, topk_weights_flat, s_reverse_scatter_idx, T, K, H)

        ctx.save_for_backward(
            x,
            gate_up_proj,
            down_proj,
            pre_act,
            topk_weights_flat,
            expert_start_idx,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
            tile_row_start,
            tile_expert,
            total_tiles_dev,
        )
        ctx.T = T
        ctx.K = K
        ctx.E = E
        ctx.H = H
        ctx.intermediate_dim = intermediate_dim
        ctx.TK = TK
        ctx.num_m_tiles = num_m_tiles
        ctx.mark_non_differentiable(top_k_index)
        ctx.set_materialize_grads(False)

        return out

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dO):
        if dO is None:
            return None, None, None, None, None

        (
            x,
            gate_up_proj,
            down_proj,
            pre_act,
            topk_weights_flat,
            expert_start_idx,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
            tile_row_start,
            tile_expert,
            total_tiles_dev,
        ) = ctx.saved_tensors

        T = ctx.T
        K = ctx.K
        E = ctx.E
        H = ctx.H
        intermediate_dim = ctx.intermediate_dim
        TK = ctx.TK
        num_m_tiles = ctx.num_m_tiles

        # dA' = dO @ W2^T, SwiGLU backward, write d_pre_act and dS
        d_pre_act = torch.empty(TK, 2 * intermediate_dim, dtype=dO.dtype, device=dO.device)
        weighted_act = torch.empty(TK, intermediate_dim, dtype=dO.dtype, device=dO.device)
        dS = torch.zeros(TK, dtype=dO.dtype, device=dO.device)  # zeros: atomic_add in kernel accumulates across N-tiles

        if num_m_tiles > 0:
            _moe_bwd_down_proj_kernel[
                lambda meta: (
                    num_m_tiles,
                    triton.cdiv(intermediate_dim, meta["BLOCK_N"]),
                )
            ](
                dO,
                x_gather_idx,
                s_scatter_idx,
                topk_weights_flat,
                down_proj,
                pre_act,
                expert_start_idx,
                tile_row_start,
                tile_expert,
                total_tiles_dev,
                d_pre_act,
                weighted_act,
                dS,
                H_dim=H,
                I_dim=intermediate_dim,
                stride_dO_T=dO.stride(0),
                stride_dO_H=dO.stride(1),
                stride_w_E=down_proj.stride(0),
                stride_w_H=down_proj.stride(1),
                stride_w_I=down_proj.stride(2),
                stride_pre_TK=pre_act.stride(0),
                stride_pre_N=pre_act.stride(1),
                stride_d_pre_TK=d_pre_act.stride(0),
                stride_d_pre_N=d_pre_act.stride(1),
                stride_wact_TK=weighted_act.stride(0),
                stride_wact_I=weighted_act.stride(1),
                BLOCK_M=BLOCK_M_TOKEN,
            )

        # dW2 = (s_k * y1)^T @ dO_gathered
        # empty (not zeros): the kernel writes every element, storing 0 for empty experts.
        ddown_proj = torch.empty_like(down_proj)
        _moe_bwd_dW2_kernel[
            lambda meta: (
                E * triton.cdiv(intermediate_dim, meta["BLOCK_M"]),
                triton.cdiv(H, meta["BLOCK_N"]),
            )
        ](
            weighted_act,
            dO,
            x_gather_idx,
            expert_start_idx,
            ddown_proj,
            H_dim=H,
            I_dim=intermediate_dim,
            stride_wact_TK=weighted_act.stride(0),
            stride_wact_I=weighted_act.stride(1),
            stride_dout_T=dO.stride(0),
            stride_dout_H=dO.stride(1),
            stride_dW2_E=ddown_proj.stride(0),
            stride_dW2_H=ddown_proj.stride(1),
            stride_dW2_I=ddown_proj.stride(2),
        )

        # dx_expanded = d_pre_act @ W1^T
        dx_expanded = torch.empty(TK, H, dtype=dO.dtype, device=dO.device)

        if num_m_tiles > 0:
            _moe_bwd_dX_expanded_kernel[lambda meta: (num_m_tiles, triton.cdiv(H, meta["BLOCK_N"]))](
                d_pre_act,
                gate_up_proj,
                expert_start_idx,
                tile_row_start,
                tile_expert,
                total_tiles_dev,
                dx_expanded,
                H_dim=H,
                I_dim=intermediate_dim,
                stride_d_pre_TK=d_pre_act.stride(0),
                stride_d_pre_N=d_pre_act.stride(1),
                stride_w_E=gate_up_proj.stride(0),
                stride_w_N=gate_up_proj.stride(1),
                stride_w_K=gate_up_proj.stride(2),
                stride_dxe_TK=dx_expanded.stride(0),
                stride_dxe_H=dx_expanded.stride(1),
                BLOCK_M=BLOCK_M_TOKEN,
            )

        # dx = unweighted gather-sum of dx_expanded
        # empty (not zeros): the gather-sum kernel stores every (t, h) element.
        dx = torch.empty(T, H, dtype=dO.dtype, device=dO.device)
        if TK > 0:
            _token_gather_weighted_sum_kernel[(T,)](
                dx_expanded,
                dS,  # dummy w_ptr — never loaded when w_is_None=True
                s_reverse_scatter_idx,
                dx,
                H_dim=H,
                K_dim=K,
                stride_Y_TK=dx_expanded.stride(0),
                stride_Y_H=dx_expanded.stride(1),
                stride_out_T=dx.stride(0),
                stride_out_H=dx.stride(1),
                w_is_None=True,
            )

        # dW1 = X_gathered^T @ d_pre_act
        # empty (not zeros): the kernel writes every element, storing 0 for empty experts.
        dgate_up_proj = torch.empty_like(gate_up_proj)
        _moe_bwd_dW1_kernel[
            lambda meta: (
                E * triton.cdiv(H, meta["BLOCK_M"]),
                triton.cdiv(2 * intermediate_dim, meta["BLOCK_N"]),
            )
        ](
            x,
            d_pre_act,
            x_gather_idx,
            expert_start_idx,
            dgate_up_proj,
            H_dim=H,
            I_dim=intermediate_dim,
            stride_x_T=x.stride(0),
            stride_x_H=x.stride(1),
            stride_d_pre_TK=d_pre_act.stride(0),
            stride_d_pre_N=d_pre_act.stride(1),
            stride_dW1_E=dgate_up_proj.stride(0),
            stride_dW1_N=dgate_up_proj.stride(1),
            stride_dW1_H=dgate_up_proj.stride(2),
        )

        return dx, dgate_up_proj, ddown_proj, None, dS.view(T, K)
