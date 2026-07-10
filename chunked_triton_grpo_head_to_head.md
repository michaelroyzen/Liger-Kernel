# Chunked Triton GRPO loss: head-to-head vs existing GRPO loss implementations

Benchmarks a new **chunked Triton GRPO loss** (`liger_kernel/transformers/chunked_grpo_loss.py` + `liger_kernel/ops/chunked_grpo_loss.py`) against the two existing Liger GRPO loss implementations. The new kernel fuses the lm_head projection into the loss with a flash-attention-style online logsumexp, so the `(N, V)` logits tensor is **never materialized** — the memory profile of the chunked fused-linear path at roughly Triton speed.

## TL;DR

| | speed | peak memory (loss stage) | 64k+ completions |
|---|---|---|---|
| `LigerFusedLinearGRPOLoss` (torch chunked) | 1x (baseline) | ~4–5 GiB, flat | works |
| `triton_grpo_loss` (unchunked) | 3.9–6.9x | O(logits): 3→123 GiB | **crashes at L ≥ 65,536** (CUDA grid-dim limit) |
| **`chunked_triton_grpo_loss` (new)** | **1.8–2.5x** | **~5–7 GiB, flat** | **works** |

The new kernel is 1.8–2.5x faster than the torch chunked path at the same flat memory profile (2.3x+ from 4k context up), and within ~1.6x of the unchunked Triton kernel at 16k+ contexts while using up to ~18x less memory. Short contexts show a larger relative gap to the unchunked kernel because fixed per-launch overheads dominate there.

## Setup

| | |
|---|---|
| GPU | NVIDIA B300 SXM6 (sm_103), 275 GiB |
| Model dims | hidden 2048, vocab 248,320 (Qwen3.5-MoE-35B-A3B), bf16 |
| Loss config | `loss_type="dapo"`, `importance_sampling_level="sequence"`, `beta=0`, temperature 1.0, eps 0.2/0.2, `num_items_in_batch` passed |
| Batch | 4 sequences per micro-batch, variable mask lengths in `[L/2, L]` |
| Software | torch 2.10 cu130, Triton 3.6, `TORCH_DYNAMO_DISABLE=1` (torch chunked path measured with `compiled=False`, matching production env) |

## Methodology

All three implementations are measured **from the hidden-state boundary** with the lm_head GEMM included, since the chunked variants fuse it: the unchunked Triton path is timed as `logits = hidden @ W.T` followed by `triton_grpo_loss(logits, ...)` (with `inplace=True`, its production setting), while both chunked paths consume `(hidden, W)` directly. Timing is forward + backward via CUDA events (2 warmup, 5 measured iterations, mean ± std); memory is peak allocation above the resident inputs. Gradients flow to both `hidden` and `weight`.

## Results

Per micro-batch of 4 sequences, forward + backward:

| completion len | logits size | torch chunked | Triton unchunked | **Triton chunked (new)** |
|---:|---:|---:|---:|---:|
| 1,024 | 1.9 GiB | 52.9 ms / 3.81 GiB | 7.7 ms / 2.86 GiB | 28.8 ms / 4.77 GiB |
| 4,096 | 7.6 GiB | 149.2 ms / 3.88 GiB | 34.2 ms / 8.59 GiB | 62.2 ms / 4.86 GiB |
| 16,384 | 30.3 GiB | 580.7 ms / 4.16 GiB | 144.7 ms / 31.51 GiB | 249.7 ms / 5.24 GiB |
| 32,768 | 60.6 GiB | 1156.6 ms / 4.54 GiB | 298.3 ms / 62.07 GiB | 486.6 ms / 5.74 GiB |
| 65,535 | 121.2 GiB | 2307.0 ms / 5.29 GiB | 599.0 ms / 123.20 GiB | 935.1 ms / 6.74 GiB |
| 65,536 | 121.3 GiB | 2306.5 ms / 5.29 GiB | **launch failure** | 979.5 ms / 6.74 GiB |

Loss values agree across all three implementations to full print precision on identical inputs (`1.1300394535064697` at L=1024).

Notes:

- The unchunked Triton path's peak is dominated by the materialized logits and scales linearly with context; both chunked paths stay flat (the ~2 GiB fp32 `grad_weight` accumulator plus a reusable per-chunk buffer).
- **Unchunked `triton_grpo_loss` hard-fails at L ≥ 65,536**: its kernels launch with grid `(B, L)`, mapping `L` onto `gridDim.y`, which CUDA caps at 65,535 (`Triton Error [CUDA]: invalid argument`). Both chunked paths are unaffected. (Fixable separately by transposing the launch grid.)
- In end-to-end training the unchunked path's logits are produced by the model forward either way, so its practical extra cost vs the chunked paths is holding logits + writing gradients into them during the loss stage; the chunked paths remove the `(N, V)` tensor from the model forward entirely by consuming hidden states.

## Correctness

Covered by `test/transformers/test_chunked_grpo_loss.py` (51 tests, all passing), which validates at context lengths 1–16,384 (including row-tile and backward-chunk boundaries), vocab sizes from 2 to 248,320 (including sub-tile and prime sizes exercising the vocab-tail masking path), bf16/fp16/fp32 inputs, boundary target ids, zero masks, extreme logit scales, and 10 loss configurations (grpo / bnpo / dr_grpo / dapo / cispo / sapo, token- and sequence-level IS, KL with bias correction, `delta` two-sided clipping, `vllm_is_ratio`, `num_items_in_batch`):

- **Intermediates**: per-token `logp`/`lse` vs an fp32 ground truth (atol 1e-2; only bf16-GEMM accumulation order separates them) and vs `fused_selective_log_softmax`.
- **Per-token outputs** (`reduce=False`): `per_token_loss` / `per_token_kl` / `is_clipped` vs unchunked `triton_grpo_loss`.
- **End results** (`reduce=True`): loss, metrics, `grad_hidden`, `grad_weight` vs a plain-torch reference, the unchunked Triton kernel, and `LigerFusedLinearGRPOLoss` (gradient cosine > 0.999, norm ratio within 2%).
- **Bitwise determinism**: repeated fwd+bwd produce bit-identical loss and gradients — fixed launch configs, no autotune, no atomics (`grad_weight` accumulates through deterministic cuBLAS GEMMs into an fp32 buffer).

## Kernel design notes (sm_103 / B300)

- Every K-loop issues exactly **one `tl.dot` per accumulator per iteration**, and each vocab tile gets a fresh accumulator — the two-dots-through-one-accumulator pattern that miscompiles on tcgen05 tiles ([triton-lang/triton#10821](https://github.com/triton-lang/triton/issues/10821)) never occurs, while still using BLOCK_M ≥ 64 (tcgen05 MMA) for tensor-core throughput.
- Backward recomputes logits tiles in a fused kernel that emits `grad_logits` for one 4,096-token chunk into a reusable buffer in the input dtype; the two large grad GEMMs run in cuBLAS (deterministic) with the cross-chunk `grad_weight` accumulator kept in fp32.
- Tile config (BM=128, BN=256, BK=64, 8 warps, 3 stages) chosen by manual sweep at V=248,320, H=2048, 65k tokens: 249 ms fwd+bwd vs 377 ms for the initial (64, 128, 128) guess; `num_warps=16` regressed ~2x, larger tiles exhaust shared memory. fp32 inputs run with 2 pipeline stages (3-stage fp32 tiles exceed sm_103's 228 KB shared memory).

## Reproduce

```bash
PYTHONPATH=src python benchmark/scripts/benchmark_chunked_grpo_loss_head_to_head.py
PYTHONPATH=src:. python -m pytest test/transformers/test_chunked_grpo_loss.py -q --override-ini addopts=
```
