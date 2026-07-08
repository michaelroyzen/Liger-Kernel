"""Probe TMA gather4 (desc.gather) + warp specialization on B300/sm103, triton 3.7.1.

Tests, on a gathered GEMM shaped like the MoE up-proj inner loop
(C[m, n] = sum_k X[idx[m], k] * W[n, k]):
  1. baseline: masked pointer gather loads (what ships today)
  2. desc.gather for X rows, desc.load for W (no WS)
  3. same as 2 with warp_specialize=True
Correctness vs torch and rough do_bench timings.
"""

import torch
import triton
import triton.language as tl


def alloc(size, align, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


triton.set_allocator(alloc)


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


@triton.jit
def _gemm_desc_gather(x_ptr, w_ptr, idx_ptr, c_ptr, T, N, K,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                      WS: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m = pid_m * BM + tl.arange(0, BM)
    token = tl.load(idx_ptr + m)  # int32 row indices for gather
    x_desc = tl.make_tensor_descriptor(x_ptr, shape=[T, K], strides=[K, 1], block_shape=[1, BK])
    w_desc = tl.make_tensor_descriptor(w_ptr, shape=[N, K], strides=[K, 1], block_shape=[BN, BK])
    n_start = pid_n * BN
    acc = tl.zeros((BM, BN), tl.float32)
    if WS:
        for kk in tl.range(0, K, BK, warp_specialize=True):
            x = x_desc.gather(token, kk)
            w = w_desc.load([n_start, kk])
            acc = tl.dot(x, tl.trans(w), acc=acc)
    else:
        for kk in tl.range(0, K, BK):
            x = x_desc.gather(token, kk)
            w = w_desc.load([n_start, kk])
            acc = tl.dot(x, tl.trans(w), acc=acc)
    n = n_start + tl.arange(0, BN)
    tl.store(c_ptr + m[:, None] * N + n[None, :], acc.to(c_ptr.dtype.element_ty))


def main():
    torch.manual_seed(0)
    T, M, N, K = 8192, 8192, 768, 2048  # up-proj-like: M gathered rows, N=I, K=H
    dtype = torch.bfloat16
    x = torch.randn(T, K, device="cuda", dtype=dtype)
    w = torch.randn(N, K, device="cuda", dtype=dtype) * 0.02
    idx = torch.randint(0, T, (M,), device="cuda", dtype=torch.int32)
    ref = (x[idx.long()].float() @ w.float().T).to(dtype)
    # rel-err check: bf16 K=2048 accumulation-order noise is ~1e-2 relative
    scale = ref.float().abs().mean().item()

    for BM, BN, BK in [(128, 128, 64), (64, 128, 64), (128, 256, 64), (256, 128, 64)]:
        grid = (M // BM, N // BN)
        row = f"BM={BM:>3} BN={BN:>3} BK={BK:>3}: "
        try:
            c0 = torch.empty(M, N, device="cuda", dtype=dtype)
            _gemm_ptr_gather[grid](x, w, idx, c0, T, N, K, BM, BN, BK)
            torch.cuda.synchronize()
            e0 = (c0 - ref).abs().max().item() / scale
            t0 = triton.testing.do_bench(lambda: _gemm_ptr_gather[grid](x, w, idx, c0, T, N, K, BM, BN, BK))
            row += f"ptr[{'OK' if e0 < 0.3 else 'BAD:%.3f' % e0} {t0:6.3f}ms] "
        except Exception as ex:
            row += f"ptr[ERR {str(ex)[:60]}] "
        try:
            c1 = torch.empty(M, N, device="cuda", dtype=dtype)
            _gemm_desc_gather[grid](x, w, idx, c1, T, N, K, BM, BN, BK, WS=False)
            torch.cuda.synchronize()
            e1 = (c1 - ref).abs().max().item() / scale
            t1 = triton.testing.do_bench(lambda: _gemm_desc_gather[grid](x, w, idx, c1, T, N, K, BM, BN, BK, WS=False))
            row += f"gather[{'OK' if e1 < 0.3 else 'BAD:%.3f' % e1} {t1:6.3f}ms] "
        except Exception as ex:
            row += f"gather[ERR {str(ex)[:60]}] "
        try:
            c2 = torch.empty(M, N, device="cuda", dtype=dtype)
            _gemm_desc_gather[grid](x, w, idx, c2, T, N, K, BM, BN, BK, WS=True)
            torch.cuda.synchronize()
            e2 = (c2 - ref).abs().max().item() / scale
            t2 = triton.testing.do_bench(lambda: _gemm_desc_gather[grid](x, w, idx, c2, T, N, K, BM, BN, BK, WS=True))
            row += f"gather+ws[{'OK' if e2 < 0.3 else 'BAD:%.3f' % e2} {t2:6.3f}ms]"
        except Exception as ex:
            row += f"gather+ws[ERR {str(ex)[:80]}]"
        print(row)


if __name__ == "__main__":
    print(f"{torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)}")
    main()
