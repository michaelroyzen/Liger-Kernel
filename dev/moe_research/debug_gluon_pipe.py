"""Run the pipelined gluon kernel once and show the full compile error."""

import torch

from probe_gluon_gemm import SMEM_LAYOUT_16B, TensorDescriptor, _gather_offsets_layout, _gluon_pipe_kernel

T, M, N, K = 256, 128, 256, 256
BM, BN, BK, NS, NW = 128, 256, 64, 2, 8
torch.manual_seed(0)
x = torch.randn(T, K, device="cuda", dtype=torch.bfloat16)
w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
idx = torch.randint(0, T, (M,), device="cuda", dtype=torch.int32)
xd = TensorDescriptor.from_tensor(x, [1, BK], SMEM_LAYOUT_16B)
wd = TensorDescriptor.from_tensor(w, [BN, BK], SMEM_LAYOUT_16B)
c = torch.full((M, N), float("nan"), device="cuda", dtype=torch.bfloat16)
_gluon_pipe_kernel[(M // BM, N // BN)](xd, wd, idx, c, N, K, BM, BN, BK,
                                       NUM_STAGES=NS, NUM_WARPS=NW,
                                       idx_layout=_gather_offsets_layout(BM, NW), num_warps=NW)
torch.cuda.synchronize()
ref = x[idx.long()].float() @ w.float().T
rel = (c.float() - ref).abs().max().item() / ref.abs().mean().item()
print("rel:", rel)
