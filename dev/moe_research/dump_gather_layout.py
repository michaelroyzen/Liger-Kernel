"""Dump the TTGIR linear layout that standard Triton uses for desc.gather offsets,
so the Gluon probe can request the exact layout ttng.async_tma_gather expects."""

import torch
import triton
import triton.language as tl


def alloc(size, align, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


triton.set_allocator(alloc)


@triton.jit
def _gemm_desc_gather(x_ptr, w_ptr, idx_ptr, c_ptr, T, N, K,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m = pid_m * BM + tl.arange(0, BM)
    token = tl.load(idx_ptr + m)
    x_desc = tl.make_tensor_descriptor(x_ptr, shape=[T, K], strides=[K, 1], block_shape=[1, BK])
    w_desc = tl.make_tensor_descriptor(w_ptr, shape=[N, K], strides=[K, 1], block_shape=[BN, BK])
    acc = tl.zeros((BM, BN), tl.float32)
    for kk in tl.range(0, K, BK):
        x = x_desc.gather(token, kk)
        w = w_desc.load([pid_n * BN, kk])
        acc = tl.dot(x, tl.trans(w), acc=acc)
    n = pid_n * BN + tl.arange(0, BN)
    tl.store(c_ptr + m[:, None] * N + n[None, :], acc.to(c_ptr.dtype.element_ty))


T, M, N, K = 1024, 256, 256, 256
x = torch.randn(T, K, device="cuda", dtype=torch.bfloat16)
w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
idx = torch.randint(0, T, (M,), device="cuda", dtype=torch.int32)
c = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
kern = _gemm_desc_gather[(M // 128, N // 128)](x, w, idx, c, T, N, K, 128, 128, 64, num_warps=8)
ttgir = kern.asm["ttgir"]
for line in ttgir.splitlines():
    low = line.lower()
    if line.lstrip().startswith("#") and "=" in line and "loc(" not in line or "tcgen05" in low or "tc_gen5" in low or "memdesc_trans" in low:
        print(line.strip()[:380])
