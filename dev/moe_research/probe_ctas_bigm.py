"""Probe num_ctas=2 and BLOCK_M=256 viability/perf on sm103 for a gathered grouped-GEMM-like loop."""

import torch
import triton
import triton.language as tl


def alloc(size, align, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


triton.set_allocator(alloc)


@triton.jit
def _gemm(x_ptr, w_ptr, idx_ptr, c_ptr, T, N, K,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, USE_TMA_W: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m = pid_m * BM + tl.arange(0, BM)
    n = pid_n * BN + tl.arange(0, BN)
    token = tl.load(idx_ptr + m).to(tl.int64)
    if USE_TMA_W:
        w_desc = tl.make_tensor_descriptor(w_ptr, shape=[N, K], strides=[K, 1], block_shape=[BN, BK])
    acc = tl.zeros((BM, BN), tl.float32)
    for kk in tl.range(0, K, BK):
        k = kk + tl.arange(0, BK)
        x = tl.load(x_ptr + token[:, None] * K + k[None, :])
        if USE_TMA_W:
            w = w_desc.load([pid_n * BN, kk])
        else:
            w = tl.load(w_ptr + n[:, None] * K + k[None, :])
        acc = tl.dot(x, tl.trans(w), acc=acc)
    tl.store(c_ptr + m[:, None] * N + n[None, :], acc.to(c_ptr.dtype.element_ty))


def main():
    torch.manual_seed(0)
    T, M, N, K = 8192, 65536, 768, 2048  # large-T regime (TK=65536 rows)
    dtype = torch.bfloat16
    x = torch.randn(T, K, device="cuda", dtype=dtype)
    w = torch.randn(N, K, device="cuda", dtype=dtype) * 0.02
    idx = torch.randint(0, T, (M,), device="cuda", dtype=torch.int32)
    ref = (x[idx.long()].float() @ w.float().T).to(dtype)
    scale = ref.float().abs().mean().item()
    flops = 2.0 * M * N * K

    for BM, BN, BK, nw, ns, nctas, tma in [
        (128, 128, 64, 8, 4, 1, True),   # ~what autotune picked
        (128, 128, 64, 8, 4, 2, True),   # + cluster
        (128, 256, 64, 8, 3, 1, True),
        (128, 256, 64, 8, 3, 2, True),
        (256, 128, 64, 8, 3, 1, True),   # big M
        (256, 128, 64, 8, 3, 2, True),
        (256, 256, 64, 8, 2, 1, True),
        (256, 256, 64, 8, 2, 2, True),
        (256, 128, 128, 8, 2, 1, True),
        (128, 128, 64, 4, 4, 1, True),   # fewer warps
        (256, 256, 64, 4, 2, 1, True),
    ]:
        tag = f"BM={BM:>3} BN={BN:>3} BK={BK:>3} nw={nw} ns={ns} ctas={nctas} tma={int(tma)}"
        try:
            grid = (M // BM, N // BN if N % BN == 0 else (N + BN - 1) // BN)
            c = torch.empty(M, N, device="cuda", dtype=dtype)
            k = _gemm[grid](x, w, idx, c, T, N, K, BM, BN, BK, tma,
                            num_warps=nw, num_stages=ns, num_ctas=nctas)
            torch.cuda.synchronize()
            err = (c - ref).abs().max().item() / scale
            ms = triton.testing.do_bench(
                lambda: _gemm[grid](x, w, idx, c, T, N, K, BM, BN, BK, tma,
                                    num_warps=nw, num_stages=ns, num_ctas=nctas))
            print(f"{tag}: {'OK ' if err < 0.3 else 'BAD'} {ms:6.3f} ms  {flops / ms / 1e9:6.0f} TFLOPS")
        except Exception as ex:
            print(f"{tag}: ERR {str(ex)[:90]}")


if __name__ == "__main__":
    print(f"{torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)}")
    main()
