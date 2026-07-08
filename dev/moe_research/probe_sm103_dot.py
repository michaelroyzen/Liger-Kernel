"""Minimal repro: two tl.dot calls chained through one accumulator in a K-loop,
on sm103 (B300), triton 3.7.1. Compares against torch.

Also tests: single dot per loop, two dots w/ separate accumulators, and dtype sweep.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _two_dots_one_acc(a1_ptr, a2_ptr, b1_ptr, b2_ptr, c_ptr, M, N, K,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """C = A1 @ B1 + A2 @ B2, two dots per K-iteration, one shared acc."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m = pid_m * BM + tl.arange(0, BM)
    n = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for kk in tl.range(0, K, BK):
        k = kk + tl.arange(0, BK)
        a1 = tl.load(a1_ptr + m[:, None] * K + k[None, :])
        b1 = tl.load(b1_ptr + k[:, None] * N + n[None, :])
        acc = tl.dot(a1, b1, acc=acc)
        a2 = tl.load(a2_ptr + m[:, None] * K + k[None, :])
        b2 = tl.load(b2_ptr + k[:, None] * N + n[None, :])
        acc = tl.dot(a2, b2, acc=acc)
    tl.store(c_ptr + m[:, None] * N + n[None, :], acc.to(c_ptr.dtype.element_ty))


@triton.jit
def _one_dot(a1_ptr, b1_ptr, c_ptr, M, N, K,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m = pid_m * BM + tl.arange(0, BM)
    n = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for kk in tl.range(0, K, BK):
        k = kk + tl.arange(0, BK)
        a1 = tl.load(a1_ptr + m[:, None] * K + k[None, :])
        b1 = tl.load(b1_ptr + k[:, None] * N + n[None, :])
        acc = tl.dot(a1, b1, acc=acc)
    tl.store(c_ptr + m[:, None] * N + n[None, :], acc.to(c_ptr.dtype.element_ty))


@triton.jit
def _two_dots_two_accs(a1_ptr, a2_ptr, b1_ptr, b2_ptr, c_ptr, M, N, K,
                       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """Same math, but each dot has its own accumulator; summed at the end."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m = pid_m * BM + tl.arange(0, BM)
    n = pid_n * BN + tl.arange(0, BN)
    acc1 = tl.zeros((BM, BN), tl.float32)
    acc2 = tl.zeros((BM, BN), tl.float32)
    for kk in tl.range(0, K, BK):
        k = kk + tl.arange(0, BK)
        a1 = tl.load(a1_ptr + m[:, None] * K + k[None, :])
        b1 = tl.load(b1_ptr + k[:, None] * N + n[None, :])
        acc1 = tl.dot(a1, b1, acc=acc1)
        a2 = tl.load(a2_ptr + m[:, None] * K + k[None, :])
        b2 = tl.load(b2_ptr + k[:, None] * N + n[None, :])
        acc2 = tl.dot(a2, b2, acc=acc2)
    acc = acc1 + acc2
    tl.store(c_ptr + m[:, None] * N + n[None, :], acc.to(c_ptr.dtype.element_ty))


def run(dtype, BM, BN, BK, M=256, N=256, K=256):
    torch.manual_seed(0)
    a1 = torch.randn(M, K, device="cuda", dtype=dtype)
    a2 = torch.randn(M, K, device="cuda", dtype=dtype)
    b1 = torch.randn(K, N, device="cuda", dtype=dtype)
    b2 = torch.randn(K, N, device="cuda", dtype=dtype)
    ref = (a1.float() @ b1.float() + a2.float() @ b2.float()).to(dtype)
    tol = dict(atol=1e-1, rtol=1e-2) if dtype != torch.float32 else dict(atol=5e-2, rtol=1e-2)

    grid = (M // BM, N // BN)
    out = {}
    c = torch.full((M, N), float("nan"), device="cuda", dtype=dtype)
    _two_dots_one_acc[grid](a1, a2, b1, b2, c, M, N, K, BM, BN, BK)
    torch.cuda.synchronize()
    out["2dots_1acc"] = (c - ref).abs().max().item()

    c1 = torch.full((M, N), float("nan"), device="cuda", dtype=dtype)
    _one_dot[grid](a1, b1, c1, M, N, K, BM, BN, BK)
    torch.cuda.synchronize()
    ref1 = (a1.float() @ b1.float()).to(dtype)
    out["1dot"] = (c1 - ref1).abs().max().item()

    c2 = torch.full((M, N), float("nan"), device="cuda", dtype=dtype)
    _two_dots_two_accs[grid](a1, a2, b1, b2, c2, M, N, K, BM, BN, BK)
    torch.cuda.synchronize()
    out["2dots_2accs"] = (c2 - ref).abs().max().item()
    return out


if __name__ == "__main__":
    print(f"device: {torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)}")
    print(f"triton {triton.__version__}, torch {torch.__version__}")
    # TF32/bf16-K256 noise floor is ~0.1-0.3; miscompiles show up as errors > 1.
    for dtype in (torch.float32, torch.bfloat16):
        for BM, BN, BK in [(128, 64, 32), (128, 128, 32), (64, 64, 64), (16, 64, 32), (64, 64, 128), (128, 64, 256)]:
            try:
                r = run(dtype, BM, BN, BK)
            except Exception as e:
                print(f"{str(dtype):>15} BM={BM:>3} BN={BN:>3} BK={BK:>3}: SKIP ({str(e)[:60]})")
                continue
            flags = {k: ("OK" if v < 1.0 else "BAD") for k, v in r.items()}
            print(f"{str(dtype):>15} BM={BM:>3} BN={BN:>3} BK={BK:>3}: "
                  + "  ".join(f"{k}={flags[k]}({r[k]:.4f})" for k in r))
