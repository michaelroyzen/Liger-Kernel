"""Minimal probe: does tl.range(warp_specialize=True) work on Hopper + triton 3.7?"""

import torch
import triton
import triton.language as tl


def alloc(size, align, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


triton.set_allocator(alloc)


@triton.jit
def k(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    a_desc = tl.make_tensor_descriptor(a_ptr, [M, K], [K, 1], [BM, BK])
    b_desc = tl.make_tensor_descriptor(b_ptr, [N, K], [K, 1], [BN, BK])
    acc = tl.zeros((BM, BN), tl.float32)
    for kk in tl.range(0, K, BK, warp_specialize=True):
        a = a_desc.load([pid_m * BM, kk])
        b = b_desc.load([pid_n * BN, kk])
        acc = tl.dot(a, tl.trans(b), acc=acc)
    c_desc = tl.make_tensor_descriptor(c_ptr, [M, N], [N, 1], [BM, BN])
    c_desc.store([pid_m * BM, pid_n * BN], acc.to(tl.bfloat16))


if __name__ == "__main__":
    M = N = K = 1024
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    c = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    k[(M // 128, N // 128)](a, b, c, M, N, K, BM=128, BN=128, BK=64, num_warps=8, num_stages=3)
    torch.cuda.synchronize()
    print("max err:", (c - a @ b.T).abs().max().item())
