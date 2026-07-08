"""Stage-0 Gluon probe (E26): tcgen05 gathered GEMM on the MoE up-proj shape.

C[m, n] = sum_k X[idx[m], k] * W[n, k]   (X: (T,K) bf16, W: (N,K) bf16, idx int32)

Compares three kernels on the SAME shape/GPU/process:
  1. `ptr`    — the standard-Triton pointer-gather GEMM that won all Phase-B probes
                (BM=128 BN=256 BK=64 ns=3 nw=8; 1103 TFLOPS on this box).
  2. `gluon0` — Gluon tcgen05, synchronous MMA, no pipelining (correctness gate).
  3. `gluonP` — Gluon tcgen05, TMA gather4 + async MMA, NUM_STAGES-deep pipeline.

Go/no-go for the full-kernel build: gluonP must beat `ptr` by >10% while matching
the torch reference.
"""

import sys

import torch
import triton
import triton.language as tl

from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    async_copy,
    fence_async_shared,
    get_tmem_reg_layout,
    tcgen05_commit,
    tcgen05_mma,
    tma,
)
from triton.experimental.gluon.language.nvidia.hopper import mbarrier
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor


def alloc(size, align, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


triton.set_allocator(alloc)

SMEM_LAYOUT_16B = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)


def _gather_offsets_layout(BM, num_warps):
    """1D layout for TMA-gather row offsets, mirroring what standard Triton emits
    for desc.gather (see dump_gather_layout.py): each thread holds groups of 4
    contiguous offsets, all 32 lanes of a warp broadcast the same values, and
    warps partition the row range (gather4 issues 4 rows per TMA op)."""
    assert BM >= 4 * num_warps, "gather needs >= 4 offsets per warp"
    reg_bases = [[1], [2]]                              # 4 contiguous elems per thread
    lane_bases = [[0]] * 5                              # broadcast across the 32 lanes
    warp_bases = [[4 << i] for i in range(num_warps.bit_length() - 1)]
    covered = 4 * num_warps
    while covered < BM:                                 # extra register replicas
        reg_bases.append([covered])
        covered *= 2
    return ttgl.DistributedLinearLayout(reg_bases, lane_bases, warp_bases, [], [BM])


# ---------------------------------------------------------------------------
# 1. standard-Triton pointer-gather baseline (identical to probe_ctas_bigm.py)
# ---------------------------------------------------------------------------


@triton.jit
def _gemm_ptr_gather(x_ptr, w_ptr, idx_ptr, c_ptr, T, N, K,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m = pid_m * BM + tl.arange(0, BM)
    n = pid_n * BN + tl.arange(0, BN)
    token = tl.load(idx_ptr + m).to(tl.int64)
    acc = tl.zeros((BM, BN), tl.float32)
    for kk in tl.range(0, K, BK):
        k = kk + tl.arange(0, BK)
        x = tl.load(x_ptr + token[:, None] * K + k[None, :])
        w = tl.load(w_ptr + n[:, None] * K + k[None, :])
        acc = tl.dot(x, tl.trans(w), acc=acc)
    tl.store(c_ptr + m[:, None] * N + n[None, :], acc.to(c_ptr.dtype.element_ty))


# ---------------------------------------------------------------------------
# 2. Gluon tcgen05, synchronous MMA (correctness gate, no overlap)
# ---------------------------------------------------------------------------


@gluon.jit
def _gluon_sync_kernel(x_desc, w_desc, idx_ptr, c_ptr, N: ttgl.constexpr, K: ttgl.constexpr,
                       BM: ttgl.constexpr, BN: ttgl.constexpr, BK: ttgl.constexpr,
                       NUM_WARPS: ttgl.constexpr, idx_layout: ttgl.constexpr):
    pid_m = ttgl.program_id(0)
    pid_n = ttgl.program_id(1)
    offs_m = pid_m * BM + ttgl.arange(0, BM, layout=idx_layout)
    tokens = ttgl.load(idx_ptr + offs_m)

    a_smem = ttgl.allocate_shared_memory(ttgl.bfloat16, [BM, BK], SMEM_LAYOUT_16B)
    b_smem = ttgl.allocate_shared_memory(ttgl.bfloat16, [BN, BK], SMEM_LAYOUT_16B)

    tmem_layout: ttgl.constexpr = TensorMemoryLayout((BM, BN), col_stride=1)
    acc_tmem = allocate_tensor_memory(ttgl.float32, [BM, BN], tmem_layout)

    bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mma_bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    mbarrier.init(mma_bar, count=1)

    phase = 0
    for i in range(K // BK):
        k = i * BK
        mbarrier.expect(bar, BM * BK * 2 + BN * BK * 2)
        tma.async_gather(x_desc, tokens, k, bar, a_smem)
        tma.async_copy_global_to_shared(w_desc, [pid_n * BN, k], bar, b_smem)
        mbarrier.wait(bar, phase)
        # fully serial: async MMA + explicit completion wait every iteration
        tcgen05_mma(a_smem, b_smem.permute((1, 0)), acc_tmem, use_acc=(i > 0), mbarriers=[mma_bar])
        mbarrier.wait(mma_bar, phase)
        phase ^= 1

    mbarrier.invalidate(bar)
    mbarrier.invalidate(mma_bar)

    reg_layout: ttgl.constexpr = get_tmem_reg_layout(ttgl.float32, (BM, BN), tmem_layout, NUM_WARPS)
    acc = acc_tmem.load(reg_layout)

    offs_cm = pid_m * BM + ttgl.arange(0, BM, layout=ttgl.SliceLayout(1, reg_layout))
    offs_cn = pid_n * BN + ttgl.arange(0, BN, layout=ttgl.SliceLayout(0, reg_layout))
    c_ptrs = c_ptr + offs_cm[:, None] * N + offs_cn[None, :]
    ttgl.store(c_ptrs, acc.to(ttgl.bfloat16))


# ---------------------------------------------------------------------------
# 3. Gluon tcgen05, pipelined: TMA gather4 + async MMA, NUM_STAGES buffers
# ---------------------------------------------------------------------------


@gluon.jit
def _gluon_pipe_kernel(x_desc, w_desc, idx_ptr, c_ptr, N: ttgl.constexpr, K: ttgl.constexpr,
                       BM: ttgl.constexpr, BN: ttgl.constexpr, BK: ttgl.constexpr,
                       NUM_STAGES: ttgl.constexpr, NUM_WARPS: ttgl.constexpr, idx_layout: ttgl.constexpr):
    pid_m = ttgl.program_id(0)
    pid_n = ttgl.program_id(1)
    K_ITERS: ttgl.constexpr = K // BK
    offs_m = pid_m * BM + ttgl.arange(0, BM, layout=idx_layout)
    tokens = ttgl.load(idx_ptr + offs_m)

    a_bufs = ttgl.allocate_shared_memory(ttgl.bfloat16, [NUM_STAGES, BM, BK], SMEM_LAYOUT_16B)
    b_bufs = ttgl.allocate_shared_memory(ttgl.bfloat16, [NUM_STAGES, BN, BK], SMEM_LAYOUT_16B)

    tmem_layout: ttgl.constexpr = TensorMemoryLayout((BM, BN), col_stride=1)
    acc_tmem = allocate_tensor_memory(ttgl.float32, [BM, BN], tmem_layout)

    load_bars = ttgl.allocate_shared_memory(ttgl.int64, [NUM_STAGES, 1], mbarrier.MBarrierLayout())
    mma_bars = ttgl.allocate_shared_memory(ttgl.int64, [NUM_STAGES, 1], mbarrier.MBarrierLayout())
    for st in ttgl.static_range(NUM_STAGES):
        mbarrier.init(load_bars.index(st), count=1)
        mbarrier.init(mma_bars.index(st), count=1)

    BYTES: ttgl.constexpr = BM * BK * 2 + BN * BK * 2

    # prologue: fill the first NUM_STAGES-1 buffers
    for pj in ttgl.static_range(NUM_STAGES - 1):
        mbarrier.expect(load_bars.index(pj), BYTES)
        tma.async_gather(x_desc, tokens, pj * BK, load_bars.index(pj), a_bufs.index(pj))
        tma.async_copy_global_to_shared(w_desc, [pid_n * BN, pj * BK], load_bars.index(pj), b_bufs.index(pj))

    for i in range(K_ITERS):
        s = i % NUM_STAGES
        p = (i // NUM_STAGES) & 1
        mbarrier.wait(load_bars.index(s), p)
        tcgen05_mma(a_bufs.index(s), b_bufs.index(s).permute((1, 0)), acc_tmem,
                    use_acc=(i > 0), mbarriers=[mma_bars.index(s)])

        # issue the load for k-iteration j = i + NUM_STAGES - 1; it reuses smem
        # stage sj, last consumed by the MMA of iteration i-1 (same stage index).
        j = i + NUM_STAGES - 1
        sj = j % NUM_STAGES
        do_load = j < K_ITERS
        mbarrier.wait(mma_bars.index(sj), ((i - 1) // NUM_STAGES) & 1, pred=do_load & (i > 0))
        mbarrier.expect(load_bars.index(sj), BYTES, pred=do_load)
        tma.async_gather(x_desc, tokens, j * BK, load_bars.index(sj), a_bufs.index(sj), pred=do_load)
        tma.async_copy_global_to_shared(w_desc, [pid_n * BN, j * BK], load_bars.index(sj), b_bufs.index(sj),
                                        pred=do_load)

    # drain: wait for the last MMA before reading TMEM
    last = K_ITERS - 1
    mbarrier.wait(mma_bars.index(last % NUM_STAGES), (last // NUM_STAGES) & 1)

    for st in ttgl.static_range(NUM_STAGES):
        mbarrier.invalidate(load_bars.index(st))
        mbarrier.invalidate(mma_bars.index(st))

    reg_layout: ttgl.constexpr = get_tmem_reg_layout(ttgl.float32, (BM, BN), tmem_layout, NUM_WARPS)
    acc = acc_tmem.load(reg_layout)

    offs_cm = pid_m * BM + ttgl.arange(0, BM, layout=ttgl.SliceLayout(1, reg_layout))
    offs_cn = pid_n * BN + ttgl.arange(0, BN, layout=ttgl.SliceLayout(0, reg_layout))
    c_ptrs = c_ptr + offs_cm[:, None] * N + offs_cn[None, :]
    ttgl.store(c_ptrs, acc.to(ttgl.bfloat16))


# ---------------------------------------------------------------------------
# 4. Gluon tcgen05, cp.async gathered A + TMA B (LDG-style A loads beat TMA
#    gather4 in the standard-Triton probes; test if that carries over here)
# ---------------------------------------------------------------------------


@gluon.jit
def _gluon_cpasync_kernel(x_ptr, w_desc, idx_ptr, c_ptr, T, N: ttgl.constexpr, K: ttgl.constexpr,
                          BM: ttgl.constexpr, BN: ttgl.constexpr, BK: ttgl.constexpr,
                          NUM_STAGES: ttgl.constexpr, NUM_WARPS: ttgl.constexpr):
    pid_m = ttgl.program_id(0)
    pid_n = ttgl.program_id(1)
    K_ITERS: ttgl.constexpr = K // BK

    # 2D layout for the gathered A pointers: 8 bf16 per thread = 16B cp.async chunks
    a_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [4, 8], [NUM_WARPS, 1], [1, 0])
    offs_am = pid_m * BM + ttgl.arange(0, BM, layout=ttgl.SliceLayout(1, a_layout))
    offs_k = ttgl.arange(0, BK, layout=ttgl.SliceLayout(0, a_layout))
    tokens = ttgl.load(idx_ptr + offs_am)
    a_base = x_ptr + tokens.to(ttgl.int64)[:, None] * K + offs_k[None, :]

    a_bufs = ttgl.allocate_shared_memory(ttgl.bfloat16, [NUM_STAGES, BM, BK], SMEM_LAYOUT_16B)
    b_bufs = ttgl.allocate_shared_memory(ttgl.bfloat16, [NUM_STAGES, BN, BK], SMEM_LAYOUT_16B)

    tmem_layout: ttgl.constexpr = TensorMemoryLayout((BM, BN), col_stride=1)
    acc_tmem = allocate_tensor_memory(ttgl.float32, [BM, BN], tmem_layout)

    load_bars = ttgl.allocate_shared_memory(ttgl.int64, [NUM_STAGES, 1], mbarrier.MBarrierLayout())
    mma_bars = ttgl.allocate_shared_memory(ttgl.int64, [NUM_STAGES, 1], mbarrier.MBarrierLayout())
    for st in ttgl.static_range(NUM_STAGES):
        mbarrier.init(load_bars.index(st), count=1)
        mbarrier.init(mma_bars.index(st), count=1)

    B_BYTES: ttgl.constexpr = BN * BK * 2

    # prologue
    for pj in ttgl.static_range(NUM_STAGES - 1):
        mbarrier.expect(load_bars.index(pj), B_BYTES)
        tma.async_copy_global_to_shared(w_desc, [pid_n * BN, pj * BK], load_bars.index(pj), b_bufs.index(pj))
        async_copy.async_copy_global_to_shared(a_bufs.index(pj), a_base + pj * BK)
        async_copy.commit_group()

    for i in range(K_ITERS):
        s = i % NUM_STAGES
        p = (i // NUM_STAGES) & 1
        mbarrier.wait(load_bars.index(s), p)  # B ready (TMA)
        # A ready: allow the NUM_STAGES-2 younger cp.async groups to be in flight
        async_copy.wait_group(NUM_STAGES - 2)
        ttgl.barrier()  # all warps' cp.async chunks visible CTA-wide
        fence_async_shared()  # generic-proxy smem writes -> async proxy (tcgen05)
        tcgen05_mma(a_bufs.index(s), b_bufs.index(s).permute((1, 0)), acc_tmem,
                    use_acc=(i > 0), mbarriers=[mma_bars.index(s)])

        j = i + NUM_STAGES - 1
        sj = j % NUM_STAGES
        do_load = j < K_ITERS
        mbarrier.wait(mma_bars.index(sj), ((i - 1) // NUM_STAGES) & 1, pred=do_load & (i > 0))
        mbarrier.expect(load_bars.index(sj), B_BYTES, pred=do_load)
        tma.async_copy_global_to_shared(w_desc, [pid_n * BN, j * BK], load_bars.index(sj), b_bufs.index(sj),
                                        pred=do_load)
        if do_load:
            async_copy.async_copy_global_to_shared(a_bufs.index(sj), a_base + j * BK)
        async_copy.commit_group()

    last = K_ITERS - 1
    mbarrier.wait(mma_bars.index(last % NUM_STAGES), (last // NUM_STAGES) & 1)

    for st in ttgl.static_range(NUM_STAGES):
        mbarrier.invalidate(load_bars.index(st))
        mbarrier.invalidate(mma_bars.index(st))

    reg_layout: ttgl.constexpr = get_tmem_reg_layout(ttgl.float32, (BM, BN), tmem_layout, NUM_WARPS)
    acc = acc_tmem.load(reg_layout)

    offs_cm = pid_m * BM + ttgl.arange(0, BM, layout=ttgl.SliceLayout(1, reg_layout))
    offs_cn = pid_n * BN + ttgl.arange(0, BN, layout=ttgl.SliceLayout(0, reg_layout))
    c_ptrs = c_ptr + offs_cm[:, None] * N + offs_cn[None, :]
    ttgl.store(c_ptrs, acc.to(ttgl.bfloat16))


# ---------------------------------------------------------------------------
# 5. Gluon tcgen05, cp.async for BOTH operands (no TMA barriers in the loop;
#    the whole stage syncs on one wait_group like the automatic pipeliner)
# ---------------------------------------------------------------------------


@gluon.jit
def _gluon_cpasync2_kernel(x_ptr, w_ptr, idx_ptr, c_ptr, T, N: ttgl.constexpr, K: ttgl.constexpr,
                           BM: ttgl.constexpr, BN: ttgl.constexpr, BK: ttgl.constexpr,
                           NUM_STAGES: ttgl.constexpr, NUM_WARPS: ttgl.constexpr):
    pid_m = ttgl.program_id(0)
    pid_n = ttgl.program_id(1)
    K_ITERS: ttgl.constexpr = K // BK

    a_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [4, 8], [NUM_WARPS, 1], [1, 0])
    offs_am = pid_m * BM + ttgl.arange(0, BM, layout=ttgl.SliceLayout(1, a_layout))
    offs_k = ttgl.arange(0, BK, layout=ttgl.SliceLayout(0, a_layout))
    tokens = ttgl.load(idx_ptr + offs_am)
    a_base = x_ptr + tokens.to(ttgl.int64)[:, None] * K + offs_k[None, :]

    b_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [4, 8], [NUM_WARPS, 1], [1, 0])
    offs_bn = pid_n * BN + ttgl.arange(0, BN, layout=ttgl.SliceLayout(1, b_layout))
    offs_bk = ttgl.arange(0, BK, layout=ttgl.SliceLayout(0, b_layout))
    b_base = w_ptr + offs_bn.to(ttgl.int64)[:, None] * K + offs_bk[None, :]

    a_bufs = ttgl.allocate_shared_memory(ttgl.bfloat16, [NUM_STAGES, BM, BK], SMEM_LAYOUT_16B)
    b_bufs = ttgl.allocate_shared_memory(ttgl.bfloat16, [NUM_STAGES, BN, BK], SMEM_LAYOUT_16B)

    tmem_layout: ttgl.constexpr = TensorMemoryLayout((BM, BN), col_stride=1)
    acc_tmem = allocate_tensor_memory(ttgl.float32, [BM, BN], tmem_layout)

    mma_bars = ttgl.allocate_shared_memory(ttgl.int64, [NUM_STAGES, 1], mbarrier.MBarrierLayout())
    for st in ttgl.static_range(NUM_STAGES):
        mbarrier.init(mma_bars.index(st), count=1)

    for pj in ttgl.static_range(NUM_STAGES - 1):
        async_copy.async_copy_global_to_shared(a_bufs.index(pj), a_base + pj * BK)
        async_copy.async_copy_global_to_shared(b_bufs.index(pj), b_base + pj * BK)
        async_copy.commit_group()

    for i in range(K_ITERS):
        s = i % NUM_STAGES
        async_copy.wait_group(NUM_STAGES - 2)
        ttgl.barrier()
        fence_async_shared()
        tcgen05_mma(a_bufs.index(s), b_bufs.index(s).permute((1, 0)), acc_tmem,
                    use_acc=(i > 0), mbarriers=[mma_bars.index(s)])

        j = i + NUM_STAGES - 1
        sj = j % NUM_STAGES
        do_load = j < K_ITERS
        mbarrier.wait(mma_bars.index(sj), ((i - 1) // NUM_STAGES) & 1, pred=do_load & (i > 0))
        if do_load:
            async_copy.async_copy_global_to_shared(a_bufs.index(sj), a_base + j * BK)
            async_copy.async_copy_global_to_shared(b_bufs.index(sj), b_base + j * BK)
        async_copy.commit_group()

    last = K_ITERS - 1
    mbarrier.wait(mma_bars.index(last % NUM_STAGES), (last // NUM_STAGES) & 1)

    for st in ttgl.static_range(NUM_STAGES):
        mbarrier.invalidate(mma_bars.index(st))

    reg_layout: ttgl.constexpr = get_tmem_reg_layout(ttgl.float32, (BM, BN), tmem_layout, NUM_WARPS)
    acc = acc_tmem.load(reg_layout)

    offs_cm = pid_m * BM + ttgl.arange(0, BM, layout=ttgl.SliceLayout(1, reg_layout))
    offs_cn = pid_n * BN + ttgl.arange(0, BN, layout=ttgl.SliceLayout(0, reg_layout))
    c_ptrs = c_ptr + offs_cm[:, None] * N + offs_cn[None, :]
    ttgl.store(c_ptrs, acc.to(ttgl.bfloat16))


# ---------------------------------------------------------------------------
# 6. Gluon tcgen05, DUAL accumulator: each CTA owns two adjacent N-tiles and
#    reads the gathered A tile ONCE for both (A re-reads dominate HBM traffic:
#    N/BN passes over 268 MB). TMEM holds 2x(128,256) fp32 = the full 256 KB.
#    One tcgen05_commit tracks both MMAs per stage.
# ---------------------------------------------------------------------------


@gluon.jit
def _gluon_dualacc_kernel(x_ptr, w_desc, idx_ptr, c_ptr, T, N: ttgl.constexpr, K: ttgl.constexpr,
                          BM: ttgl.constexpr, BN: ttgl.constexpr, BK: ttgl.constexpr,
                          NUM_STAGES: ttgl.constexpr, NUM_WARPS: ttgl.constexpr):
    pid_m = ttgl.program_id(0)
    pid_n = ttgl.program_id(1)
    K_ITERS: ttgl.constexpr = K // BK
    n0 = pid_n * (2 * BN)
    has_n1 = n0 + BN < N
    # clamp instead of predicating: keeps mbarrier.expect byte counts constexpr;
    # a clamped (duplicate) tile is computed into acc1 and discarded at the store
    n1 = ttgl.minimum(n0 + BN, N - BN)

    a_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [4, 8], [NUM_WARPS, 1], [1, 0])
    offs_am = pid_m * BM + ttgl.arange(0, BM, layout=ttgl.SliceLayout(1, a_layout))
    offs_k = ttgl.arange(0, BK, layout=ttgl.SliceLayout(0, a_layout))
    tokens = ttgl.load(idx_ptr + offs_am)
    a_base = x_ptr + tokens.to(ttgl.int64)[:, None] * K + offs_k[None, :]

    a_bufs = ttgl.allocate_shared_memory(ttgl.bfloat16, [NUM_STAGES, BM, BK], SMEM_LAYOUT_16B)
    b0_bufs = ttgl.allocate_shared_memory(ttgl.bfloat16, [NUM_STAGES, BN, BK], SMEM_LAYOUT_16B)
    b1_bufs = ttgl.allocate_shared_memory(ttgl.bfloat16, [NUM_STAGES, BN, BK], SMEM_LAYOUT_16B)

    tmem_layout: ttgl.constexpr = TensorMemoryLayout((BM, BN), col_stride=1)
    acc0 = allocate_tensor_memory(ttgl.float32, [BM, BN], tmem_layout)
    acc1 = allocate_tensor_memory(ttgl.float32, [BM, BN], tmem_layout)

    load_bars = ttgl.allocate_shared_memory(ttgl.int64, [NUM_STAGES, 1], mbarrier.MBarrierLayout())
    mma_bars = ttgl.allocate_shared_memory(ttgl.int64, [NUM_STAGES, 1], mbarrier.MBarrierLayout())
    for st in ttgl.static_range(NUM_STAGES):
        mbarrier.init(load_bars.index(st), count=1)
        mbarrier.init(mma_bars.index(st), count=1)

    B_BYTES: ttgl.constexpr = 2 * BN * BK * 2

    for pj in ttgl.static_range(NUM_STAGES - 1):
        mbarrier.expect(load_bars.index(pj), B_BYTES)
        tma.async_copy_global_to_shared(w_desc, [n0, pj * BK], load_bars.index(pj), b0_bufs.index(pj))
        tma.async_copy_global_to_shared(w_desc, [n1, pj * BK], load_bars.index(pj), b1_bufs.index(pj))
        async_copy.async_copy_global_to_shared(a_bufs.index(pj), a_base + pj * BK)
        async_copy.commit_group()

    for i in range(K_ITERS):
        s = i % NUM_STAGES
        p = (i // NUM_STAGES) & 1
        mbarrier.wait(load_bars.index(s), p)
        async_copy.wait_group(NUM_STAGES - 2)
        ttgl.barrier()
        fence_async_shared()
        tcgen05_mma(a_bufs.index(s), b0_bufs.index(s).permute((1, 0)), acc0, use_acc=(i > 0))
        tcgen05_mma(a_bufs.index(s), b1_bufs.index(s).permute((1, 0)), acc1, use_acc=(i > 0))
        tcgen05_commit(mma_bars.index(s))

        j = i + NUM_STAGES - 1
        sj = j % NUM_STAGES
        do_load = j < K_ITERS
        mbarrier.wait(mma_bars.index(sj), ((i - 1) // NUM_STAGES) & 1, pred=do_load & (i > 0))
        mbarrier.expect(load_bars.index(sj), B_BYTES, pred=do_load)
        tma.async_copy_global_to_shared(w_desc, [n0, j * BK], load_bars.index(sj), b0_bufs.index(sj), pred=do_load)
        tma.async_copy_global_to_shared(w_desc, [n1, j * BK], load_bars.index(sj), b1_bufs.index(sj), pred=do_load)
        if do_load:
            async_copy.async_copy_global_to_shared(a_bufs.index(sj), a_base + j * BK)
        async_copy.commit_group()

    last = K_ITERS - 1
    mbarrier.wait(mma_bars.index(last % NUM_STAGES), (last // NUM_STAGES) & 1)

    for st in ttgl.static_range(NUM_STAGES):
        mbarrier.invalidate(load_bars.index(st))
        mbarrier.invalidate(mma_bars.index(st))

    reg_layout: ttgl.constexpr = get_tmem_reg_layout(ttgl.float32, (BM, BN), tmem_layout, NUM_WARPS)
    offs_cm = pid_m * BM + ttgl.arange(0, BM, layout=ttgl.SliceLayout(1, reg_layout))
    offs_cn0 = n0 + ttgl.arange(0, BN, layout=ttgl.SliceLayout(0, reg_layout))
    out0 = acc0.load(reg_layout)
    ttgl.store(c_ptr + offs_cm[:, None] * N + offs_cn0[None, :], out0.to(ttgl.bfloat16))
    if has_n1:
        out1 = acc1.load(reg_layout)
        ttgl.store(c_ptr + offs_cm[:, None] * N + (offs_cn0 + BN)[None, :], out1.to(ttgl.bfloat16))


# ---------------------------------------------------------------------------
# 7. Gluon tcgen05 + MANUAL warp specialization: a 1-warp load partition issues
#    TMA gather4 + TMA B copies, a 1-warp MMA partition issues async MMAs, and
#    the default warps only run the epilogue. Producer/consumer sync via
#    full/empty mbarrier pairs (the canonical Blackwell WS matmul shape).
# ---------------------------------------------------------------------------


@gluon.jit
def _ws_load_partition(x_desc, w_desc, idx_ptr, a_bufs, b_bufs, full_bars, empty_bars,
                       pid_m, pid_n, K: ttgl.constexpr,
                       BM: ttgl.constexpr, BN: ttgl.constexpr, BK: ttgl.constexpr,
                       NUM_STAGES: ttgl.constexpr, idx_layout: ttgl.constexpr):
    offs_m = pid_m * BM + ttgl.arange(0, BM, layout=idx_layout)
    tokens = ttgl.load(idx_ptr + offs_m)
    BYTES: ttgl.constexpr = BM * BK * 2 + BN * BK * 2
    K_ITERS: ttgl.constexpr = K // BK
    for i in range(K_ITERS):
        s = i % NUM_STAGES
        # stage s was last consumed by MMA i - NUM_STAGES; wait for its empty signal
        mbarrier.wait(empty_bars.index(s), ((i // NUM_STAGES) - 1) & 1, pred=(i >= NUM_STAGES))
        mbarrier.expect(full_bars.index(s), BYTES)
        tma.async_gather(x_desc, tokens, i * BK, full_bars.index(s), a_bufs.index(s))
        tma.async_copy_global_to_shared(w_desc, [pid_n * BN, i * BK], full_bars.index(s), b_bufs.index(s))


@gluon.jit
def _ws_mma_partition(a_bufs, b_bufs, acc_tmem, full_bars, empty_bars, done_bar,
                      K: ttgl.constexpr, BK: ttgl.constexpr, NUM_STAGES: ttgl.constexpr):
    K_ITERS: ttgl.constexpr = K // BK
    for i in range(K_ITERS):
        s = i % NUM_STAGES
        mbarrier.wait(full_bars.index(s), (i // NUM_STAGES) & 1)
        tcgen05_mma(a_bufs.index(s), b_bufs.index(s).permute((1, 0)), acc_tmem,
                    use_acc=(i > 0), mbarriers=[empty_bars.index(s)])
    tcgen05_commit(done_bar)


@gluon.jit
def _ws_default_partition():
    pass


@gluon.jit
def _gluon_ws_kernel(x_desc, w_desc, idx_ptr, c_ptr, N: ttgl.constexpr, K: ttgl.constexpr,
                     BM: ttgl.constexpr, BN: ttgl.constexpr, BK: ttgl.constexpr,
                     NUM_STAGES: ttgl.constexpr, NUM_WARPS: ttgl.constexpr,
                     idx_layout: ttgl.constexpr):
    pid_m = ttgl.program_id(0)
    pid_n = ttgl.program_id(1)

    a_bufs = ttgl.allocate_shared_memory(ttgl.bfloat16, [NUM_STAGES, BM, BK], SMEM_LAYOUT_16B)
    b_bufs = ttgl.allocate_shared_memory(ttgl.bfloat16, [NUM_STAGES, BN, BK], SMEM_LAYOUT_16B)
    tmem_layout: ttgl.constexpr = TensorMemoryLayout((BM, BN), col_stride=1)
    acc_tmem = allocate_tensor_memory(ttgl.float32, [BM, BN], tmem_layout)

    full_bars = ttgl.allocate_shared_memory(ttgl.int64, [NUM_STAGES, 1], mbarrier.MBarrierLayout())
    empty_bars = ttgl.allocate_shared_memory(ttgl.int64, [NUM_STAGES, 1], mbarrier.MBarrierLayout())
    done_bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    for st in ttgl.static_range(NUM_STAGES):
        mbarrier.init(full_bars.index(st), count=1)
        mbarrier.init(empty_bars.index(st), count=1)
    mbarrier.init(done_bar, count=1)
    ttgl.barrier()

    ttgl.warp_specialize(
        [
            (_ws_default_partition, ()),
            (_ws_load_partition, (x_desc, w_desc, idx_ptr, a_bufs, b_bufs, full_bars, empty_bars,
                                  pid_m, pid_n, K, BM, BN, BK, NUM_STAGES, idx_layout)),
            (_ws_mma_partition, (a_bufs, b_bufs, acc_tmem, full_bars, empty_bars, done_bar,
                                 K, BK, NUM_STAGES)),
        ],
        [1, 1],  # one warp each for load and MMA partitions
        [40, 40],
    )

    mbarrier.wait(done_bar, 0)

    for st in ttgl.static_range(NUM_STAGES):
        mbarrier.invalidate(full_bars.index(st))
        mbarrier.invalidate(empty_bars.index(st))
    mbarrier.invalidate(done_bar)

    reg_layout: ttgl.constexpr = get_tmem_reg_layout(ttgl.float32, (BM, BN), tmem_layout, NUM_WARPS)
    acc = acc_tmem.load(reg_layout)
    offs_cm = pid_m * BM + ttgl.arange(0, BM, layout=ttgl.SliceLayout(1, reg_layout))
    offs_cn = pid_n * BN + ttgl.arange(0, BN, layout=ttgl.SliceLayout(0, reg_layout))
    ttgl.store(c_ptr + offs_cm[:, None] * N + offs_cn[None, :], acc.to(ttgl.bfloat16))


# ---------------------------------------------------------------------------
# 8. Tightened pipeline: outer loop steps NUM_STAGES at a time with an unrolled
#    static inner loop, so stage indices/phases are compile-time constants (no
#    div/mod on the hot path) and cp.async completion rides an mbarrier instead
#    of wait_group + CTA barrier. Requires K_ITERS % NUM_STAGES == 0.
# ---------------------------------------------------------------------------


@gluon.jit
def _gluon_tight_kernel(x_ptr, w_desc, idx_ptr, c_ptr, T, N: ttgl.constexpr, K: ttgl.constexpr,
                        BM: ttgl.constexpr, BN: ttgl.constexpr, BK: ttgl.constexpr,
                        NUM_STAGES: ttgl.constexpr, NUM_WARPS: ttgl.constexpr):
    pid_m = ttgl.program_id(0)
    pid_n = ttgl.program_id(1)
    K_ITERS: ttgl.constexpr = K // BK
    ttgl.static_assert(K_ITERS % NUM_STAGES == 0)
    N_BLOCKS: ttgl.constexpr = K_ITERS // NUM_STAGES

    a_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [4, 8], [NUM_WARPS, 1], [1, 0])
    offs_am = pid_m * BM + ttgl.arange(0, BM, layout=ttgl.SliceLayout(1, a_layout))
    offs_k = ttgl.arange(0, BK, layout=ttgl.SliceLayout(0, a_layout))
    tokens = ttgl.load(idx_ptr + offs_am)
    a_base = x_ptr + tokens.to(ttgl.int64)[:, None] * K + offs_k[None, :]

    a_bufs = ttgl.allocate_shared_memory(ttgl.bfloat16, [NUM_STAGES, BM, BK], SMEM_LAYOUT_16B)
    b_bufs = ttgl.allocate_shared_memory(ttgl.bfloat16, [NUM_STAGES, BN, BK], SMEM_LAYOUT_16B)

    tmem_layout: ttgl.constexpr = TensorMemoryLayout((BM, BN), col_stride=1)
    acc_tmem = allocate_tensor_memory(ttgl.float32, [BM, BN], tmem_layout)

    full_bars = ttgl.allocate_shared_memory(ttgl.int64, [NUM_STAGES, 1], mbarrier.MBarrierLayout())
    mma_bars = ttgl.allocate_shared_memory(ttgl.int64, [NUM_STAGES, 1], mbarrier.MBarrierLayout())
    for st in ttgl.static_range(NUM_STAGES):
        mbarrier.init(full_bars.index(st), count=1)
        mbarrier.init(mma_bars.index(st), count=1)

    B_BYTES: ttgl.constexpr = BN * BK * 2

    # prologue: fill all NUM_STAGES buffers (cp.async A synced via wait_group)
    for pj in ttgl.static_range(NUM_STAGES):
        mbarrier.expect(full_bars.index(pj), B_BYTES)
        tma.async_copy_global_to_shared(w_desc, [pid_n * BN, pj * BK], full_bars.index(pj), b_bufs.index(pj))
        async_copy.async_copy_global_to_shared(a_bufs.index(pj), a_base + pj * BK)
        async_copy.commit_group()

    phase = 0
    for blk in range(N_BLOCKS):
        base_k = blk * NUM_STAGES
        for st in ttgl.static_range(NUM_STAGES):
            i = base_k + st
            mbarrier.wait(full_bars.index(st), phase)  # B ready (TMA bytes)
            async_copy.wait_group(NUM_STAGES - 1)      # A of stage st landed
            ttgl.barrier()
            fence_async_shared()
            tcgen05_mma(a_bufs.index(st), b_bufs.index(st).permute((1, 0)), acc_tmem,
                        use_acc=(i > 0), mbarriers=[mma_bars.index(st)])
            # refill stage st for iteration i + NUM_STAGES
            j = i + NUM_STAGES
            do_load = j < K_ITERS
            mbarrier.wait(mma_bars.index(st), phase, pred=do_load)
            mbarrier.expect(full_bars.index(st), B_BYTES, pred=do_load)
            tma.async_copy_global_to_shared(w_desc, [pid_n * BN, j * BK], full_bars.index(st),
                                            b_bufs.index(st), pred=do_load)
            if do_load:
                async_copy.async_copy_global_to_shared(a_bufs.index(st), a_base + j * BK)
            async_copy.commit_group()
        phase ^= 1

    mbarrier.wait(mma_bars.index(NUM_STAGES - 1), phase ^ 1)

    for st in ttgl.static_range(NUM_STAGES):
        mbarrier.invalidate(full_bars.index(st))
        mbarrier.invalidate(mma_bars.index(st))

    reg_layout: ttgl.constexpr = get_tmem_reg_layout(ttgl.float32, (BM, BN), tmem_layout, NUM_WARPS)
    acc = acc_tmem.load(reg_layout)
    offs_cm = pid_m * BM + ttgl.arange(0, BM, layout=ttgl.SliceLayout(1, reg_layout))
    offs_cn = pid_n * BN + ttgl.arange(0, BN, layout=ttgl.SliceLayout(0, reg_layout))
    ttgl.store(c_ptr + offs_cm[:, None] * N + offs_cn[None, :], acc.to(ttgl.bfloat16))


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------


def main():
    torch.manual_seed(0)
    T, M, N, K = 8192, 65536, 768, 2048  # up-proj shape at T=8192, K_route=8
    dtype = torch.bfloat16
    x = torch.randn(T, K, device="cuda", dtype=dtype)
    w = torch.randn(N, K, device="cuda", dtype=dtype) * 0.02
    idx = torch.randint(0, T, (M,), device="cuda", dtype=torch.int32)
    ref = (x[idx.long()].float() @ w.float().T).to(dtype)
    scale = ref.float().abs().mean().item()
    flops = 2.0 * M * N * K

    def report(name, c, ms):
        rel = (c.float() - ref.float()).abs().max().item() / scale
        status = "OK " if rel < 0.35 else f"BAD(rel={rel:.3f})"
        print(f"{name:>28}: {status} {ms:7.3f} ms  {flops / ms / 1e9:7.0f} TFLOPS")
        return rel < 0.35

    # 1. baseline
    BM, BN, BK = 128, 256, 64
    grid = (M // BM, N // BN)
    c0 = torch.empty(M, N, device="cuda", dtype=dtype)
    _gemm_ptr_gather[grid](x, w, idx, c0, T, N, K, BM, BN, BK, num_warps=8, num_stages=3)
    torch.cuda.synchronize()
    ms0 = triton.testing.do_bench(lambda: _gemm_ptr_gather[grid](x, w, idx, c0, T, N, K, BM, BN, BK,
                                                                 num_warps=8, num_stages=3))
    report("ptr (triton, BN=256 ns=3)", c0, ms0)

    # descriptors shared by both gluon kernels
    def descs(BK_):
        x_desc = TensorDescriptor.from_tensor(x, [1, BK_], SMEM_LAYOUT_16B)
        w_desc = TensorDescriptor.from_tensor(w, [BN, BK_], SMEM_LAYOUT_16B)
        return x_desc, w_desc

    # 2. gluon sync (correctness gate)
    x_desc, w_desc = descs(BK)
    c1 = torch.full((M, N), float("nan"), device="cuda", dtype=dtype)
    _gluon_sync_kernel[grid](x_desc, w_desc, idx, c1, N, K, BM, BN, BK, NUM_WARPS=8,
                             idx_layout=_gather_offsets_layout(BM, 8), num_warps=8)
    torch.cuda.synchronize()
    ms1 = triton.testing.do_bench(lambda: _gluon_sync_kernel[grid](x_desc, w_desc, idx, c1, N, K, BM, BN, BK,
                                                                   NUM_WARPS=8,
                                                                   idx_layout=_gather_offsets_layout(BM, 8),
                                                                   num_warps=8))
    ok1 = report("gluon0 (sync mma)", c1, ms1)
    if not ok1:
        print("correctness gate FAILED at the sync kernel — stopping")
        sys.exit(1)

    # 3. gluon pipelined, a few (BM, BN, BK, NUM_STAGES) points
    for bm, bn, bk, ns, nw in [
        (128, 256, 64, 2, 8),
        (128, 256, 64, 3, 8),
        (128, 256, 64, 4, 8),
        (128, 128, 64, 4, 8),
        (128, 256, 128, 2, 8),
        (128, 256, 64, 3, 4),
        (64, 256, 64, 4, 4),
    ]:
        try:
            g = (M // bm, N // bn)
            xd, wd = (TensorDescriptor.from_tensor(x, [1, bk], SMEM_LAYOUT_16B),
                      TensorDescriptor.from_tensor(w, [bn, bk], SMEM_LAYOUT_16B))
            c2 = torch.full((M, N), float("nan"), device="cuda", dtype=dtype)
            _gluon_pipe_kernel[g](xd, wd, idx, c2, N, K, bm, bn, bk, NUM_STAGES=ns, NUM_WARPS=nw,
                                  idx_layout=_gather_offsets_layout(bm, nw), num_warps=nw)
            torch.cuda.synchronize()
            ms2 = triton.testing.do_bench(
                lambda: _gluon_pipe_kernel[g](xd, wd, idx, c2, N, K, bm, bn, bk,
                                              NUM_STAGES=ns, NUM_WARPS=nw,
                                              idx_layout=_gather_offsets_layout(bm, nw), num_warps=nw))
            report(f"gluonP BM{bm} BN{bn} BK{bk} ns{ns} nw{nw}", c2, ms2)
        except Exception as e:
            print(f"{f'gluonP BM{bm} BN{bn} BK{bk} ns{ns} nw{nw}':>28}: ERR {str(e)[:110]}")

    # 4. cp.async gathered A variant
    for bm, bn, bk, ns, nw in [
        (128, 256, 64, 3, 8),
        (128, 256, 64, 4, 8),
        (128, 256, 128, 3, 8),
        (128, 256, 64, 4, 4),
    ]:
        try:
            g = (M // bm, N // bn)
            wd = TensorDescriptor.from_tensor(w, [bn, bk], SMEM_LAYOUT_16B)
            c3 = torch.full((M, N), float("nan"), device="cuda", dtype=dtype)
            _gluon_cpasync_kernel[g](x, wd, idx, c3, T, N, K, bm, bn, bk,
                                     NUM_STAGES=ns, NUM_WARPS=nw, num_warps=nw)
            torch.cuda.synchronize()
            ms3 = triton.testing.do_bench(
                lambda: _gluon_cpasync_kernel[g](x, wd, idx, c3, T, N, K, bm, bn, bk,
                                                 NUM_STAGES=ns, NUM_WARPS=nw, num_warps=nw))
            report(f"gluonA BM{bm} BN{bn} BK{bk} ns{ns} nw{nw}", c3, ms3)
        except Exception as e:
            print(f"{f'gluonA BM{bm} BN{bn} BK{bk} ns{ns} nw{nw}':>28}: ERR {str(e)[:110]}")

    # 5. cp.async for BOTH operands
    for bm, bn, bk, ns, nw in [
        (128, 256, 64, 3, 8),
        (128, 256, 64, 4, 8),
        (128, 256, 64, 4, 4),
        (128, 128, 64, 4, 8),
    ]:
        try:
            g = (M // bm, N // bn)
            c4 = torch.full((M, N), float("nan"), device="cuda", dtype=dtype)
            _gluon_cpasync2_kernel[g](x, w, idx, c4, T, N, K, bm, bn, bk,
                                      NUM_STAGES=ns, NUM_WARPS=nw, num_warps=nw)
            torch.cuda.synchronize()
            ms4 = triton.testing.do_bench(
                lambda: _gluon_cpasync2_kernel[g](x, w, idx, c4, T, N, K, bm, bn, bk,
                                                  NUM_STAGES=ns, NUM_WARPS=nw, num_warps=nw))
            report(f"gluonB BM{bm} BN{bn} BK{bk} ns{ns} nw{nw}", c4, ms4)
        except Exception as e:
            print(f"{f'gluonB BM{bm} BN{bn} BK{bk} ns{ns} nw{nw}':>28}: ERR {str(e)[:110]}")

    # 6. dual-accumulator (A read once per two N-tiles)
    for bm, bn, bk, ns, nw in [
        (128, 256, 64, 2, 8),
        (128, 256, 32, 4, 8),
        (128, 256, 32, 5, 8),
        (128, 256, 64, 2, 4),
        (128, 128, 64, 3, 8),
    ]:
        try:
            g = (M // bm, triton.cdiv(N, 2 * bn))
            wd = TensorDescriptor.from_tensor(w, [bn, bk], SMEM_LAYOUT_16B)
            c5 = torch.full((M, N), float("nan"), device="cuda", dtype=dtype)
            _gluon_dualacc_kernel[g](x, wd, idx, c5, T, N, K, bm, bn, bk,
                                     NUM_STAGES=ns, NUM_WARPS=nw, num_warps=nw)
            torch.cuda.synchronize()
            ms5 = triton.testing.do_bench(
                lambda: _gluon_dualacc_kernel[g](x, wd, idx, c5, T, N, K, bm, bn, bk,
                                                 NUM_STAGES=ns, NUM_WARPS=nw, num_warps=nw))
            report(f"gluonC BM{bm} BN{bn} BK{bk} ns{ns} nw{nw}", c5, ms5)
        except Exception as e:
            print(f"{f'gluonC BM{bm} BN{bn} BK{bk} ns{ns} nw{nw}':>28}: ERR {str(e)[:110]}")

    # 7. manual warp specialization (load warp + mma warp + epilogue warps)
    for bm, bn, bk, ns, nw in [
        (128, 256, 64, 3, 4),
        (128, 256, 64, 4, 4),
        (128, 256, 64, 4, 8),
        (128, 128, 64, 4, 4),
        (128, 256, 128, 2, 4),
    ]:
        try:
            g = (M // bm, N // bn)
            xd = TensorDescriptor.from_tensor(x, [1, bk], SMEM_LAYOUT_16B)
            wd = TensorDescriptor.from_tensor(w, [bn, bk], SMEM_LAYOUT_16B)
            c6 = torch.full((M, N), float("nan"), device="cuda", dtype=dtype)
            lay = _gather_offsets_layout(bm, 1)  # 1-warp load partition
            _gluon_ws_kernel[g](xd, wd, idx, c6, N, K, bm, bn, bk,
                                NUM_STAGES=ns, NUM_WARPS=nw, idx_layout=lay, num_warps=nw)
            torch.cuda.synchronize()
            ms6 = triton.testing.do_bench(
                lambda: _gluon_ws_kernel[g](xd, wd, idx, c6, N, K, bm, bn, bk,
                                            NUM_STAGES=ns, NUM_WARPS=nw, idx_layout=lay, num_warps=nw))
            report(f"gluonWS BM{bm} BN{bn} BK{bk} ns{ns} nw{nw}", c6, ms6)
        except Exception as e:
            print(f"{f'gluonWS BM{bm} BN{bn} BK{bk} ns{ns} nw{nw}':>28}: ERR {str(e)[:130]}")

    # 8. tightened unrolled pipeline
    for bm, bn, bk, ns, nw in [
        (128, 256, 64, 4, 8),
        (128, 256, 64, 4, 4),
        (128, 256, 64, 8, 4),
        (128, 256, 128, 2, 4),
        (128, 256, 64, 2, 4),
    ]:
        try:
            g = (M // bm, N // bn)
            wd = TensorDescriptor.from_tensor(w, [bn, bk], SMEM_LAYOUT_16B)
            c7 = torch.full((M, N), float("nan"), device="cuda", dtype=dtype)
            _gluon_tight_kernel[g](x, wd, idx, c7, T, N, K, bm, bn, bk,
                                   NUM_STAGES=ns, NUM_WARPS=nw, num_warps=nw)
            torch.cuda.synchronize()
            ms7 = triton.testing.do_bench(
                lambda: _gluon_tight_kernel[g](x, wd, idx, c7, T, N, K, bm, bn, bk,
                                               NUM_STAGES=ns, NUM_WARPS=nw, num_warps=nw))
            report(f"gluonT BM{bm} BN{bn} BK{bk} ns{ns} nw{nw}", c7, ms7)
        except Exception as e:
            print(f"{f'gluonT BM{bm} BN{bn} BK{bk} ns{ns} nw{nw}':>28}: ERR {str(e)[:130]}")


if __name__ == "__main__":
    print(f"{torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)}, "
          f"triton {triton.__version__}")
    main()
