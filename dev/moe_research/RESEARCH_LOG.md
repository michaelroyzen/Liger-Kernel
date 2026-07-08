# MoE Triton Kernel Autoresearch Log

> **Phase B (B300/Blackwell) lives at the bottom of this file** — it found and fixed an
> sm103 miscompile that made the shipped dx gradients silently wrong on Blackwell (E19),
> re-ran the E18 backlog with probes (three of its top ideas died cheaply), and shipped
> Blackwell-gated config spaces + per-regime dW autotune keys: v0→final on B300 is
> **2.07×/2.35× full step @ T=8192/32768** with correct gradients. H100 results below
> are unchanged.

## FINAL RESULT (shipped in `src/liger_kernel/ops/fused_moe*.py`)

Qwen3-MoE-30B shard (E=128, H=2048, I=768, K=8, bf16, H100 80GB, torch 2.12.1/triton 3.7.1),
PR #1179 baseline → final (`dev/moe_research/results/final_vs_v0.txt`):

| T | fwd | bwd | full step | inference fwd |
|---|-----|-----|-----------|---------------|
| 128 | 0.551→0.444 (**1.24×**) | 1.250→0.840 (**1.49×**) | 1.818→1.275 (**1.43×**) | 0.551→0.440 (**1.25×**) |
| 1024 | 0.646→0.512 (**1.26×**) | 1.694→1.149 (**1.47×**) | 2.354→1.663 (**1.42×**) | 0.644→0.501 (**1.28×**) |
| 8192 | 2.659→1.456 (**1.83×**) | 7.210→3.998 (**1.80×**) | 9.920→5.417 (**1.83×**) | 2.651→1.398 (**1.90×**) |
| 32768 | 9.901→5.066 (**1.95×**) | 26.076→14.477 (**1.80×**) | 36.249→19.005 (**1.91×**) | 9.969→5.174 (**1.93×**) |

Memory: inference forward no longer materializes pre-activations (−201 MB @T=8192, −805 MB
@T=32768 of activation traffic+allocation); ~1.2 GB of memset traffic per backward removed.
Opt-in `LIGER_FUSED_MOE_MEMORY_EFFICIENT=1` additionally cuts training peak by −288 MB @8192 /
−1151 MB @32768 (in-place SwiGLU backward + weighted_act recompute) at 12-18% slower backward;
default mode keeps standard autograd semantics (retain_graph re-backward works and is tested).

Plus: skewed routing full step **1.44×**, Mixtral-8x7B shape fwd **1.36×** / full **1.19×**,
CUDA-graph capture now works (fwd and fwd+bwd; blocked before by the `.item()` sync).
All 22 unit tests + Qwen3-MoE bf16 convergence test pass; `make checkstyle` green; the PR's
CSV benchmark data refreshed on this H100.

**Shipped changes** (each one validated by A/B experiments below):
1. Sync-free tile scheduling (no `.item()`; upper-bound grid + device-side early exit) — E2
2. No-memset weight/dx grads (kernels fully overwrite; empty experts store zeros) — E3
3. L2-aware CTA swizzles: GROUP_M remap on grouped GEMMs + expert-major dW grids — E4
4. Adaptive `BLOCK_M_TOKEN` from tokens/expert (16→128) + per-BLOCK_M autotune keys — E5
5. 2D token-gather grid (T × H-tiles) — E5
6. TMA descriptors for all expert-weight loads on Hopper+ (portable fallback kept) — E6
7. fp32 dS accumulator (precision + native fp32 atomics) — E7/E8
8. Inference fast path (skip pre_act store + ctx save, `STORE_PREACT` constexpr) — E11
9. Wider-tile autotune configs (BN=256/BK=128) — E12
10. Opt-in `LIGER_FUSED_MOE_MEMORY_EFFICIENT=1`: in-place SwiGLU backward (version-bumped,
    autotuner restore_value-guarded) + weighted_act elimination — E7/E8
11. Shape-aware autotune config pruning (cold-start unit tests 32:30 → 16:14; warm 6:40)

**Rejected after measurement**: fused-aggregation epilogue via fp32 atomics (E9, fwd 0.73×),
warp specialization (E10, Triton/sm90 can't WS gathered loads), deep-K dW configs (E13, neutral),
interior-tile mask specialization (E14, 0.95×).

**Portability**: only TMA is hardware-gated (cap ≥ 9 + alignment, with a portable fallback —
A100 degrades gracefully, B200 auto-enables); everything else is architecture-agnostic.
Full audit + Blackwell warp-spec follow-up design (TMA gather4 + sm100 gate): see E17.
Ranked Blackwell research backlog (WS+gather4, un-capped tiles via TMEM, cluster multicast,
L2 retune, fusion revisit): see E18.

Improving [PR #1179](https://github.com/linkedin/Liger-Kernel/pull/1179) (fused MoE Triton kernels,
already merged on this fork's `main`) with ideas from [SonicMoE](https://github.com/Dao-AILab/sonic-moe)
(arXiv:2512.14080).

## Environment

- GPU: NVIDIA H100 80GB HBM3 (SM90), driver 580.105.08, CUDA 13.0
- torch 2.12.1+cu130, triton 3.7.1, python 3.12 (uv venv at `.venv`)
- Benchmark config (default): Qwen3-MoE-30B expert shard — E=128, H=2048, I=768, K=8, bf16
- Harness: `dev/moe_research/bench.py` (triton `do_bench`, median ms; torch profiler for per-kernel breakdown)

## Roofline notes (Qwen3 config)

- Fwd FLOPs = 6·TK·H·I (up 4·TK·H·I + down 2·TK·H·I). T=8192,K=8 → 618 GFLOP → ~0.62 ms @ 990 TFLOPS.
- Weights = W1 805 MB + W2 402 MB = 1.21 GB → 0.36 ms to read once @ 3.35 TB/s.
- Arithmetic intensity at T=8192 ≈ H100 bf16 ridge (~295 flops/B) → both compute AND memory matter.
- At small T (≤1024) the op is weight-read-bound: floor ≈ 0.36 ms regardless of T.

## Baseline anatomy (PR #1179 as merged, `variants/v0_baseline.py`)

Forward: 3 routing kernels + torch `.sum` + **`.item()` host sync** → up_proj+SwiGLU grouped GEMM
(grid: m-tiles × I-tiles) → down_proj grouped GEMM → token gather-sum (grid: T).
Backward: bwd_down (dA' recompute + SwiGLU bwd + dS atomics) → dW2 → dX_expanded → unweighted
gather-sum (dx) → dW1. dW1/dW2 use `torch.zeros_like` outputs + early-exit for empty experts.
BLOCK_M_TOKEN=64 module constant; GEMM autotune over BN∈{64,128}, BK∈{32,64}, warps∈{4,8}, stages∈{2..5}.

## Hypothesis backlog (updated as results come in)

| # | Hypothesis | Expected effect | Status |
|---|------------|-----------------|--------|
| H1 | Remove `.item()` sync: upper-bound m-tiles, GEMM CTAs early-exit past actual count (grid = bound) | Removes host stall each fwd; CUDA-graph-safe | |
| H2 | Adaptive BLOCK_M_TOKEN (16/32/64/128 by TK/E); add BLOCK_M to autotune key | Less padded MMA work at small T; better weight reuse at large T | |
| H3 | dW1/dW2: `empty_like` + in-kernel zero store for empty experts (drop 805 MB memset) | Backward speedup | |
| H4 | fp32 dS accumulator (atomics precision + native fp32 atomics), cast at end | Precision + minor perf | |
| H5 | Token gather-sum: 2D grid (T × H-tiles) | Small-T parallelism | |
| H6 | L2 swizzle (GROUP_M-style pid remap) on grouped GEMMs | Better L2 weight reuse at large T | |
| H7 | TMA (tl.make_tensor_descriptor) for weight loads on H100 | Higher BW, deeper pipelining | |
| H8 | Skip pre_act store + ctx save when no grad needed (inference path) | Inference fwd speedup | |
| H9 | Expand autotune space (BN=256, BK=128) for the GEMMs | Better configs at large T | |
| H10 | Fuse torch `.sum` into K2 (or drop it) | One less launch | |
| H11 | Full-tile branch specialization (skip row masks on interior tiles) | Fewer predicated ops | |

## Goals (clarified by user)

1. **Maximum memory efficiency** — keep data in SRAM/L2, minimize HBM traffic and activation
   allocations.
2. **Maximum compute utilization** — highest achievable MFU/bandwidth utilization on H100.

## Experiment log

### E0 — Environment + baseline correctness (PASS)

- `uv venv` + `uv pip install -e ".[dev]"` OK.
- `pytest test/transformers/test_fused_moe.py`: **19 passed** in 759 s (autotune dominates).
- Harness smoke test: fwd T=1024 = 0.661 ms.

### E1 — Baseline benchmark + profile (results/baseline_full.txt)

| T | fwd ms | bwd ms | full ms | mem full MB |
|---|--------|--------|---------|-------------|
| 128 | 0.551 | 1.252 | 1.825 | 1165 |
| 1024 | 0.645 | 1.698 | 2.361 | 1252 |
| 8192 | 2.646 | 7.187 | 9.894 | 1953 |
| 32768 | 9.900 | 26.157 | 36.190 | 4356 |

Profile (full, T=8192, µs/iter of 9711 total): dW1 2269 (23%), dX_exp 1936 (20%), dW2 1567 (16%),
down_proj 1297 (13%), up_proj 1103 (11%), bwd_down 932 (10%), **memsets (FillFunctor) 377 (4 calls)**,
token_gather 208×2, routing ≈20 total. At T=128 the memsets are 367µs = **22% of the step**.

Analysis vs roofline: dW1 ideal ≈ 0.42 ms (is 2.27) — kernels are bound by redundant tile re-reads
(x re-read per N-tile, d_pre re-read per M-tile) with no L2-friendly CTA ordering; dW autotune is
locked at num_stages=2, BK≤32. Priorities re-ranked: grid ordering/L2 (H6), dW autotune space (H9),
kill memsets (H3), then sync removal (H1), BLOCK_M adaptivity (H2), TMA (H7).

### E2 — V1 `v1_nosync`: remove `.item()` host sync (H1) → ACCEPTED

Upper-bound m-tile count host-side (`TK//BLOCK_M + min(E,TK)`), allocate metadata at the bound,
GEMM CTAs load the actual count (`expert_tile_offset[E]`) and early-exit. Also makes the op
CUDA-graph-capturable.

| T | fwd v0→v1 | full v0→v1 |
|---|-----------|------------|
| 128 | 0.557→0.455 (**1.22×**) | 1.832→1.704 (1.08×) |
| 1024 | 0.646→0.551 (**1.17×**) | 2.419→2.236 (1.08×) |
| 8192 | 2.646→2.557 (1.03×) | 9.939→9.791 (1.02×) |

### E3 — V2 `v2_nomemset`: kill dW/dx memsets (H3) → ACCEPTED

`zeros_like→empty_like` for dW1/dW2 (kernels now always store, writing 0 for empty experts;
dropped `reset_to_zero` + early-return), `zeros→empty` for dx (gather-sum writes every element).
Removes ~1.2 GB of fill traffic per backward.

| T | full v1→v2 |
|---|------------|
| 128 | 1.699→1.325 (**1.28×**) |
| 8192 | 9.774→9.461 (1.03×; later remeasured 8.01 with fresh tune) |

### E4 — V3 `v3_swizzle` + V4 `v4_dw_swizzle`: L2-aware CTA ordering (H6) → ACCEPTED

V3: grouped-GEMM grids flattened to 1D with matmul-tutorial GROUP_M=8 swizzle (m-tiles are
expert-major, so a group shares both x rows and the expert's weight n-tiles in L2).
V4: dW1/dW2 grids reordered expert-major 1D (+GROUP_M=4 swizzle inside an expert), so one
expert's token rows are read from L2 instead of HBM per output tile; dW autotune space extended
(BN 256, BK 64, stages 3).

| T | mode | v2→v3 | v3→v4 |
|---|------|-------|-------|
| 1024 | fwd | 0.543→0.511 (1.06×) | — |
| 8192 | fwd | 2.044→1.939 (1.05×) | — |
| 8192 | full | 8.012→6.650 (**1.20×**) | — |
| 32768 | full | 30.846→24.333 (**1.27×**) | — |
| 8192 | bwd | — | 4.614→3.846 (**1.20×**) |
| 32768 | bwd | — | 16.843→14.401 (**1.17×**) |

Cumulative vs baseline (T=8192): bwd 7.187→3.846 (**1.87×**), full 9.894→~6 (**~1.6×**).

### E5 — V5 `v5_adaptive`: adaptive BLOCK_M_TOKEN + 2D token-gather grid (H2+H5) → ACCEPTED

`BLOCK_M_TOKEN = clamp(next_pow2(TK/E), 16, 128)` chosen host-side per call (no sync);
"BLOCK_M" added to GEMM autotune keys so each tile size tunes separately. Token gather-sum
grid became (T, ceil(H/BLOCK_H)) so small-T launches still fill 132 SMs.

| T | fwd v4→v5 | full v4→v5 |
|---|-----------|------------|
| 128 | 0.450→0.449 (1.00×) | 1.301→1.283 (1.01×) |
| 8192 | 1.964→1.531 (**1.28×**) | 6.038→5.566 (1.08×) |
| 32768 | 7.360→5.696 (**1.29×**) | 22.646→20.095 (**1.13×**) |

BLOCK_M=128 at T≥8192 halves per-CTA weight re-reads; at T=128 (BLOCK_M=16) the op is
weight-read-bound (~0.36 ms floor) so tile shape doesn't matter, as predicted by roofline.

### E6 — V6 `v6_tma`: TMA descriptors for expert-weight loads (H7) → ACCEPTED

`tl.make_tensor_descriptor` over flattened (E·2I, H) / (E·H, I) weight views; per-CTA
device-side descriptors (weights only — token rows are gathered, and TMA gather doesn't
exist on Hopper). Gotcha found: Triton stores the TMA scratch allocator in a **ContextVar,
which doesn't propagate to autograd's backward thread** — must call `triton.set_allocator`
inside both forward() and backward().

| T | fwd v5→v6 | full v5→v6 |
|---|-----------|------------|
| 1024 | 0.510→0.510 (1.00×) | 1.640→1.664 (0.99×) |
| 8192 | 1.577→1.487 (**1.06×**) | 5.299→5.294 (1.00×) |
| 32768 | 5.540→5.250 (**1.06×**) | 20.630→18.947 (**1.09×**) |

### E8 — V8 `v8_best`: v6 TMA + v7 keepers (in-place d_pre_act, fp32 dS, opt-in no-wact) → ACCEPTED

Also adds inference path (no `pre_act` store / no ctx save when no input requires grad —
`STORE_PREACT` constexpr) and dW2-before-bwd-down ordering so the opt-in
`LIGER_FUSED_MOE_MEMORY_EFFICIENT=1` recompute mode stays correct with the in-place alias.

*Post-port correction (E16):* the PR's own benchmark harness times backward with
`retain_graph=True` repeats, which the unconditional in-place alias breaks (by design it
raises after a version bump — better than silent corruption, but a behavior regression vs
stock autograd). Shipped resolution: the in-place alias moved under
`LIGER_FUSED_MOE_MEMORY_EFFICIENT=1` together with the weighted_act recompute; default mode
keeps a separate d_pre_act buffer and full retain_graph support (both paths under test).
The autotuner's `restore_value=["pre_act_ptr"]` is likewise conditional on the flag, since
its `copy_()` during tuning would version-bump the saved tensor even in default mode.

| T | v6→v8 fwd | v6→v8 full | peak mem full v6→v8 |
|---|-----------|------------|---------------------|
| 1024 | 0.509→0.511 | 1.661→1.660 | 1252→~1228 MB |
| 8192 | 1.594→1.519 (1.05×) | 5.350→5.275 | 1953→**1761 MB** |
| 32768 | 5.195→5.297 | 20.592→**18.772 (1.10×)** | 4356→**3589 MB** |

With `LIGER_FUSED_MOE_MEMORY_EFFICIENT=1`: 1665 MB @8192 (−288 vs v6), 3205 MB @32768
(−1151 vs v6) at 12–18% slower full step — documented tradeoff, default off.

### E9 — V10 `v10_fuseagg`: fuse token aggregation into down-proj epilogue (M2) → REJECTED

Replaced the (TK,H) Y buffer + gather kernel with fp32 atomic accumulation directly into
out[token]. **Forward 10–27% slower** (0.80× @8192, 0.73× @32768): TK·H·4B of fp32 atomic
traffic (537 MB @8192) is strictly worse than 2·TK·H·2B of coalesced store+gather-load, and
peak memory didn't even improve (the pre_act+post_act+metadata high-water mark dominates once
Y is gone... peak unchanged because backward's dW1/dx_expanded allocations set the peak).
Also would have made forward nondeterministic. Not worth it — rejected.

### E10 — V9 `v9_ws`: warp specialization (`tl.range(..., warp_specialize=True)`) → REJECTED (toolchain)

Two successive Triton 3.7/sm90 compiler failures:
1. `TaskIdBackwardPropagation` assertion — fixed by restructuring the early-exit
   `if pid >= n: return` into `if pid < n: <body>` (WS pass can't handle `cf.cond_br` from
   data-dependent early returns).
2. `LLVM ERROR: unsupported load type for producer commit` — fundamental: Hopper WS producer
   warps only support **TMA** loads, and every MoE GEMM here has a *gathered* token-row operand
   (regular masked load). A minimal all-TMA matmul probe (`probe_ws.py`) compiles and runs fine,
   isolating the gathered load as the blocker. Not fixable at kernel level today; revisit when
   Triton supports mixed producer loads (or Blackwell TMA gather4 — see E17 for the concrete
   sm100-gated design).

### E11 — V8 inference path (STORE_PREACT=False + no ctx save) → ACCEPTED

No-grad forward: v0 0.632→v8 0.501 ms (**1.26×**) @T=1024, 2.183→1.459 ms (**1.50×**) @T=8192,
plus the (TK, 2I) pre_act allocation disappears entirely in inference (201 MB @T=8192).

### E12 — V11 `v11_bigtiles`: wider GEMM tiles (BN=256, BK=128 configs) → ACCEPTED

8 extra autotune configs for the grouped GEMMs. fwd @8192: 1.587→1.488 (**1.07×**),
full @32768: 19.049→18.523 (1.03×), neutral elsewhere. Cheap win, tuner picks per-shape.

### E13 — V13 deep-K dW configs (BK=128 for dW1/dW2) → REJECTED (neutral: 1.01×/0.99×)

### E14 — V14 interior-tile specialization (unmasked loads on full tiles) → REJECTED

fwd 0.95× @8192, 0.94× @32768 — the extra branch hurts more than predicate elimination helps
(Triton already hoists the row predicate well; code bloat splits the instruction cache).

### E15 — Robustness of final candidate (v8/v11 family)

- **Skewed routing** (Zipf, skew=1.5, T=8192): v0 7.710 → v8 5.340 ms full (**1.44×**) — the
  gains hold under load imbalance (adaptive BLOCK_M + upper-bound grid absorb skew).
- **Mixtral-8x7B shape** (E=8, H=4096, I=14336, K=2, T=4096): fwd **1.36×**, full **1.19×** vs v0.
- **CUDA graphs** (enabled by removing the `.item()` sync): inference capture+replay OK;
  training via `torch.cuda.make_graphed_callables` OK. Gotcha: warmup pins leaves'
  AccumulateGrad nodes to the legacy stream — use fresh leaf tensors for capture.

### E7 — V7 `v7_mempack`: in-place d_pre_act + drop weighted_act + fp32 dS → PARTIAL

Three memory changes tested together: (a) d_pre_act written in-place over pre_act
(saves TK×2I bytes), (b) weighted_act eliminated, dW2 recomputes s_k·y1 from pre_act
(saves TK×I bytes), (c) fp32 dS accumulator (precision).
Memory: peak full-step −288 MB @T=8192 (1953→1665), −1151 MB @T=32768 (4356→3205). Fwd
unchanged, but **full-step regressed 9-14%** at large T — the dW2 recompute reads pre_act
(2 tiles) instead of weighted_act (1 tile) across all ceil(H/BN) output column tiles.
Decision: keep (a)+(c) unconditionally (a is pure win: −TK·2I bytes, no perf cost);
make (b) opt-in via `LIGER_FUSED_MOE_MEMORY_EFFICIENT=1` → folded into V8.

### E17 — Architecture portability & gating audit (A100 / H100 / B200) — analysis, no code changes

How Hopper-specific are the shipped changes, and what would other architectures need?
(All experiments above ran on H100/sm90 only; the following is a static audit plus checks
against the installed triton 3.7.1 sources — nothing here was executed on A100/B200.)

**Portability inventory.** 10 of 12 shipped changes are architecture-agnostic pure
Triton/scheduling logic (sync-free grids, no-memset grads, L2 swizzles, adaptive BLOCK_M,
2D token gather, fp32 dS, inference path, config pruning, memory-efficient mode) — they are
also where most of the 1.4–1.9× lives. The only hardware-gated feature is **TMA**, and it is
already gated correctly: `_tma_eligibility` requires `get_device_capability >= 9` plus
16 B row alignment, and `USE_TMA` is a constexpr, so the same kernel source compiles a
portable pointer-load fallback.

**A100 (sm80, cap 8) — works today via fallback, no new gating needed for correctness.**
- TMA path auto-disabled → pointer loads (TMA was only worth 6–9% on H100, so little is lost).
- fp32 dS is *more* portable than the baseline it replaced: bf16 `atomic_add` needs sm90 PTX
  and would be CAS-emulated on sm80.
- Soft spot: wide-tile configs (BN=256/BK=128, deep stages) exceed A100's 164 KB smem
  (H100/B200: 228 KB). Verified in the installed autotuner (`runtime/autotuner.py::_bench`)
  that `OutOfResources` / `CompileTimeAssertionFailure` / `PTXASError` are caught and the
  config scored `inf` → skipped safely, just wasted tuning time. Optional hardening: add an
  smem-budget term to our `early_config_prune`. GROUP_M=8 was tuned for 50 MB L2 (A100: 40 MB)
  — retune candidate, not a correctness issue.

**B200 (sm100, cap 10) — shipped TMA path auto-enables (cap ≥ 9 covers it); no code changes
needed to run.** Triton lowers `tl.dot` to tcgen05 MMAs on sm100 by itself. Would want
on-device retuning (tiles/GROUP_M vs the much larger L2) and revalidation; the repo's
`infer_device_arch()` helper (already used by swiglu/CE for Blackwell gating) is the natural
hook for arch-specific config spaces if profiling justifies them.

**Warp specialization revisited for Blackwell (follow-up design, extends E10).** Two distinct
gates must not be conflated:

| feature | gate | H100 | B200 |
|---|---|---|---|
| TMA weight loads (shipped) | cap ≥ 9 | on | on |
| `desc.gather` token loads + `warp_specialize=True` (proposed) | sm100 only | impossible (no TMA gather4) | new code |

Even on B200, the *current* kernel would still fail WS compilation: WS requires the producer
warp group to own **every** load in the K-loop, and our loop mixes TMA weight loads with one
regular gathered token-row load (the exact E10 blocker, which is architecture-independent at
the Triton level). Blackwell's TMA gather4 removes the hardware limitation, and triton 3.7
already exposes it — verified `tensor_descriptor.gather(x_offsets, y_offset)` /
`semantic.descriptor_gather` in the installed sources, with constraints that fit our shapes:
2D descriptor with block rows = 1, ≥ 8 row indices per gather (BLOCK_M ∈ 16…128 ✓), and for
bf16 ≥ 16 columns (BLOCK_K ∈ 32…128 ✓). Plan: convert token-row loads to descriptor gathers
and set `warp_specialize=True`, gated to sm100 specifically (NOT cap ≥ 9 — it would break
Hopper; sm120 consumer Blackwell unverified). Details to handle: gather has no load mask →
clamp invalid row indices and rely on the existing store-side masks, and explicitly zero
invalid rows feeding the dS reduction. Expected order 10–25% on the GEMMs (Triton Blackwell
tutorial ballpark) — a hypothesis to measure on a B200, not a claim; Hopper keeps the current
path (its WS remains blocked unless tokens are pre-gathered, which re-adds the TK×H round
trip this project eliminated).

### E18 — Blackwell (sm100) research directions — ranked backlog, NOT YET RUN

Everything on H100 is measured; everything below is a ranked hypothesis for a future B200/B300
box, ordered by expected payoff × confidence. Relevant sm100 hardware deltas vs sm90: tcgen05
MMA with accumulators in a dedicated 256 KB/SM tensor memory (TMEM) instead of registers,
2-CTA MMA (CTA pairs share operands), TMA gather4/scatter4 (row-indexed tensor loads/stores),
~2.5× larger (dual-die, partitioned) L2, HBM3e ≈ 8 TB/s, same 228 KB smem/SM.
Gating hook: `infer_device_arch()` + autotune config spaces; the `dev/moe_research/` harness
runs unchanged on a new GPU.

1. **Warp specialization + TMA gather4** (the E17 design, sm100-gated). Expected 10–25% on the
   six GEMMs. Split into two experiments: (a) descriptor-gather for token rows *without* WS —
   removes per-row address math and LSU pressure from the hot loop on its own; (b) gather + WS.
2. **Un-cap the tile-size vector (extends E5/E12).** On Hopper, BLOCK_M is register/occupancy-
   capped (fp32 accumulators live in the register file); on sm100, TMEM holds accumulators and
   2-CTA MMA shares the weight operand across an SM pair. Raise `_pick_block_m_token`'s cap
   (128 → 256) and widen BN/BK/stages under an sm100 gate. Same physics as E5: weight re-reads
   per expert = ceil(tokens/BLOCK_M), so each doubling halves the dominant large-T traffic term.
   Config-space work only — the metadata layer already parameterizes BLOCK_M.
3. **Cluster multicast for expert weights.** TMA multicast within a thread-block cluster loads
   one weight tile into several SMs' smem per HBM transaction — the only lever that attacks the
   small-T weight-read floor itself (0.36 ms on H100; nothing shipped could lower it, only
   approach it). Most speculative: Triton cluster support is the least mature piece involved.
4. **GROUP_M / swizzle retune for the larger L2 (extends E4).** GROUP_M=8/4 were tuned against
   50 MB L2; B200 has ~2.5× that but split across two dies — bigger groups fit, yet an expert's
   working set straddling dies could regress. Cheap sweep, modest expected gain.
5. **Fused up+SwiGLU+down forward becomes more attractive.** TMEM-freed smem lets the 96 KB y1
   slice (BLOCK_M=64, I=768) coexist with deeper pipelines, softening the fusion's main cost.
   The BLOCK_M=64-vs-128 tension (see E5) still needs to be settled empirically per shape.
6. **Free riders (no work needed):** bandwidth-bound pieces (token gather, memset savings) scale
   ~2.4× with HBM3e automatically; sync-free scheduling / CUDA-graph capture / fp32 dS /
   inference path are architecture-neutral; shipped TMA weight loads auto-enable via cap ≥ 9.
7. **(Tradeoff option, flagged not endorsed):** Blackwell-native MXFP8 block-scaled storage of
   the saved pre_act — the only lever that touches the ~9.7 GB @T=8192 (48 layers) of saved
   activations, but a numerics tradeoff by construction, unlike items 1–5.

Caution from this project's own base rate: 5 of 12 paper-plausible hypotheses lost on silicon
(E9/E10/E13/E14 + part of E7). Treat this list as an experiment queue, not a promise.

---

# Phase B: Blackwell (B300, sm103) — continuation of the E18 backlog

## Environment (Phase B)

- GPU: NVIDIA B300 SXM6 AC (sm103, cap (10,3)), 148 SMs, 275 GB HBM3e, 132.6 MB L2,
  driver 595.71.05, CUDA 13.2
- torch 2.12.1+cu130, triton 3.7.1, python 3.12 (fresh `uv venv` at `.venv`,
  `uv pip install torch --index-url .../cu130` + `uv pip install -e ".[dev]"`)
- Measured peaks (this box): bf16 matmul 1480-1580 TFLOPS (torch/cuBLAS n=4096-16384),
  device-to-device copy 6.41 TB/s
- Roofline (Qwen3 config): ridge ≈ 230 flops/B. Fwd 618 GFLOP @T=8192 → ideal ≈ 0.42 ms;
  weight read floor 1.21 GB → 0.19 ms. Same "compute AND memory both matter" regime as H100,
  but both ceilings ≈ 1.5-1.9× higher.

### E19 — B300 environment + correctness gate → found sm103 MISCOMPILE; fixed in V15 (`v15_fixdx`)

`check_correctness` on the shipped src kernel **FAILED on B300**: dx wrong by ~44% of elements
(fp32, T=512/E=8/H=256/I=128/K=2; H100 numbers with the identical wheel stack were clean).
Isolated with `probe_sm103_dot.py` to a **Triton 3.7.1 sm103 miscompile**: two `tl.dot` calls
chained through ONE accumulator inside a K-loop produce garbage under the tcgen05 MMA path
(max err ~60 on values O(50)); one dot per loop is fine; two dots into two accumulators is
fine; BM=16 (non-tcgen05 lowering) is fine. Exactly one MoE kernel used the bad pattern:
`_moe_bwd_dX_expanded_kernel` (gate half-GEMM + up half-GEMM into one acc). Per-config probe
(`probe_b300_configs.py`): 21 of 40 autotune configs compute wrong dx, 15 OOM-skip, only the
4 BK=128 configs happened to be correct — so autotune usually lands on a broken config.

**V15 fix**: compute dx = d_pre_act @ W1[e] as a SINGLE GEMM over K=2I (d_pre_act columns and
W1[e] rows share the same [gate; up] ordering — the two-half split was never mathematically
necessary, it just mirrored the forward kernel's structure). Architecture-neutral, TMA path
unchanged (same flattened descriptor, k now spans [0,2I)), and the K-loop gets twice as many
single-dot iterations → also pipelines better. v0_baseline has the same latent bug on B300
(its timings below are still valid; its dx is silently wrong on sm100/sm103).
`check_correctness(v15)`: PASS.

Baseline A/B on B300 (`results/b300_e19_baseline.txt`), Qwen3-MoE-30B shard, bf16:

| T | mode | v0 ms | v15 ms | speedup |
|---|------|-------|--------|---------|
| 128 | fwd / bwd / full / infer | 0.365 / 0.938 / 1.317 / 0.352 | 0.318 / 0.553 / 0.775 / 0.283 | 1.15 / 1.70 / 1.70 / 1.24 |
| 1024 | fwd / bwd / full / infer | 0.398 / 1.310 / 1.729 / 0.401 | 0.309 / 0.780 / 1.067 / 0.288 | 1.29 / 1.68 / 1.62 / 1.39 |
| 8192 | fwd / bwd / full / infer | 1.320 / 5.059 / 6.360 / 1.338 | 0.949 / 2.661 / 3.573 / 0.905 | 1.39 / 1.90 / 1.78 / 1.48 |
| 32768 | fwd / bwd / full / infer | 4.989 / 19.836 / 24.410 / 5.145 | 3.369 / 9.192 / 12.275 / 3.277 | 1.48 / 2.16 / 1.99 / 1.57 |

(The H100-shipped optimizations carry over: full-step 1.7-2.0× vs v0 on B300 vs 1.4-1.9× on
H100.) Peak memory identical to H100 runs (same allocation structure). During tuning,
ptxas-blackwell also hard-crashed on one up_proj config
("Register allocation failed with register count of '255'") — caught+skipped by the
autotuner like the smem OOMs; noted as toolchain fragility.

Per-kernel profile of v15 @T=8192 full (µs, % of 3057 total): dW1 652 (21%), bwd_down 614
(20%), up_proj 523 (17%), dX 442 (15%), dW2 392 (13%), down_proj 307 (10%), gather 98×2 (3%).
Effective TFLOPS: dX 932 (BN=256 single-dot — best), up_proj 788 (two-acc structure halves
usable TMEM tile width), down_proj 671, dW1 632, dW2 526, bwd_down 336 (epilogue-bound:
~1 GB of pre_act/d_pre_act/wact traffic + dS atomics after the MMA loop).

### Probe results that re-rank the E18 backlog (all on B300, triton 3.7.1)

- **TMA gather4 (`desc.gather`) ≈ pointer gathers, WS slower** (`probe_gather_b300.py`,
  up-proj-shaped gathered GEMM, M=8192): ptr 0.042 ms, gather 0.046 ms, gather+WS 0.046-0.081 ms.
  E18 #1 (WS + gather4) is a LOSS on this toolchain → dropped without a kernel-level variant.
- **num_ctas=2 loses ~10% and miscompiles with BM=256** (`probe_ctas_bigm.py`, PassManager
  crash) → E18 #3 (cluster multicast) dropped.
- **BM=256 loses to BM=128 + BN=256** (870 vs 1103 TFLOPS on the gathered-GEMM probe;
  M=65536, N=768, K=2048) → E18 #2's BLOCK_M raise dropped; the winning direction is
  wide-N tiles + deeper pipelines with BM=128 (TMEM holds the accumulator either way).
- Autotune-selected configs on B300 (v15, `probe_best_configs.py`): the GEMM kernels pick
  ns=4-5 and BN=128-256 — already at the edge of the H100-era config space → widen it.

### E20 — V16 `v16_bwtune`: Blackwell-gated autotune config spaces (retunes E5/E12/E13, L2 part of E18 #4)

Hypothesis: the H100-tuned config space underserves sm103 — TMEM accumulators make BN=256
"free", tcgen05 issue style makes nw=4 viable, HBM3e + 132 MB L2 favor deep-K dW tiles and
bigger GROUP_M. Gate: `cap[0] == 10` (sm100/sm103 only; sm120 consumer has no TMEM and would
waste tuning time), Hopper/Ampere spaces byte-identical. Adds: 10 GEMM configs (BN=256
nw=4/ns=3-5, BK=128 ns=3-4, 2× GROUP_M=16), 8 dW configs (BK=128, ns=4, GROUP_M=8),
BLOCK_H=1024 token-gather rows. Expected: 5-15% on the GEMM-bound modes.

Result (`results/b300_e20_v16_vs_v15.txt`): **~neutral** — +5% fwd @8192, +3% infer @32768,
+1% full @8192/@32768, −9% fwd @128 (see below), everything else ±1%. correctness PASS.
The H100 wide-tile extras (E12) already covered the configs sm103 wants; the NEW configs
rarely beat them (autotuner picked the same winners at the probed sizes). The T=128 fwd wobble
is tuning noise on a 0.3 ms op (do_bench re-picks between runs; v16@128 later re-measured
0.29 ms). Verdict: MIXED-KEEP — the config additions are free where neutral, help fwd at
large T, and the gate keeps other archs untouched. Learnings: on B300 the *shape* of the
kernel (single-dot vs two-dot, epilogue weight) limits throughput, not the tile config —
attack the epilogue next.

### E21 — TMA weight-load A/B on B300 (extends E6): KEEP TMA

Same v16 code, TMA on vs `_tma_eligibility` forced False (separate module instances so each
arm autotunes for its own USE_TMA constexpr). `results/b300_e21_tma_ab.txt`:
TMA wins at large T — fwd @32768 1.05×, infer @32768 1.03×, full @8192/@32768 1.02×; loses
slightly at T=1024 fwd (0.95×, weight-read-bound regime where descriptor setup overhead
shows). Net: keep the existing cap≥9 auto-enable unchanged. The E17 prediction ("B200
auto-enables, would want revalidation") holds up: TMA is worth less on B300 (2-5%) than on
H100 (6-9%) because the 132 MB L2 already absorbs most weight re-reads.

### E22 — V17 `v17_epilogue`: evict-first hints on cross-pass activation traffic

Hypothesis: `_moe_bwd_down_proj_kernel` runs at 336 effective TFLOPS (worst of all six) not
because of its MMA loop but its epilogue: per step at T=8192 it loads pre_act (402 MB,
read-once), stores d_pre_act (402 MB) + weighted_act (201 MB) — ~1 GB of dead-on-arrival L2
traffic that evicts the W2 tiles and gathered dO rows other resident CTAs still need.
Similarly the forward up-proj stores pre_act (402 MB) that nothing reads until backward,
polluting L2 for the immediately-following down-proj (which re-reads post_act). Change:
`eviction_policy="evict_first"` on (a) bwd_down epilogue pre_act loads + d_pre_act/wact
stores, (b) up-proj pre_act stores — both only when the (TK, 2I) buffer exceeds the device
L2 size (host-computed flag, new autotune key) and only on Blackwell-DC (unmeasured on
Hopper → other archs compile the exact old path). Expected: 5-15% on backward @T≥8192, small
fwd gain; neutral at small T (flag off below ~43K tokens·K on B300).

Result (`results/b300_e22_v17_vs_v16.txt`): **REJECTED** — backward/full ±1% everywhere
(0.99× bwd @8192), fwd +2% @8192 but −3% infer @8192, −11% fwd @1024 (noise band on a 0.3 ms
op, but no upside to justify it). Learnings: sm103's L2 replacement policy already handles
streaming stores well (the H100-era `evict_first` on gathered x rows survives in the shipped
kernel because it was measured there; adding more hints on B300 buys nothing). The
bwd_down 336-TFLOPS ceiling is NOT L2 displacement; it's plain HBM traffic + the serial
epilogue dependency chain. NB: between E20 and E22 the absolute numbers drifted (bwd @8192
2.66→2.15 ms) — the dW kernels tuned at a different first-seen T; see E24.

### E23 — LIGER_FUSED_MOE_MEMORY_EFFICIENT on B300 (extends E7/E8): tradeoff unchanged, still opt-in

`results/b300_e23_memeff.txt` (v15): full @8192 4.011 ms / 1665 MB vs default 3.032 ms /
1953 MB; @32768 13.757 ms / 3205 MB vs 10.411 ms / 4357 MB. Same −288 MB / −1152 MB savings
as H100, but the recompute now costs 32% / 32% on the full step (H100: 12-18%) — B300's
faster MMAs make the extra pre_act reads in dW2 relatively more expensive. Default-off stays
right on Blackwell; docs unchanged (the flag documents "~10-15% slower backward" per its
H100 measurement — now arch-dependent, noted here).

### E24 — V18 `v18_dwkey`: tokens-per-expert bucket in the dW autotune keys

Hypothesis: `_moe_bwd_dW{1,2}_kernel` key their autotune caches on (H_dim, I_dim) only, so
whichever T runs FIRST in a process pins the dW config for every later T. The optimum is
regime-dependent (TK/E=64: output-write-bound, small tiles/parallelism; TK/E=2048+:
K-loop-bound, deep-K tiles) — the E20↔E22 run-to-run drift (bwd @8192 2.66 vs 2.15 ms
depending on which T tuned first) is direct evidence of the contamination. Mirror E5's
per-BLOCK_M GEMM keys: add TPE_BUCKET = clamp(next_pow2(TK//E), 16, 4096) as a key-only
constexpr on both dW kernels. Architecture-neutral, config spaces unchanged; costs one
tuning pass per distinct bucket. Expected: recovers the ~0.5 ms/step backward drift at
T=8192 deterministically; helps any variable-sequence-length training run.

Result → **ACCEPTED**. Same-order A/B is neutral (±1%, `results/b300_e24_v18_vs_v16.txt`,
both orders) — correct, since single-shape tuning conditions are identical. The realistic
mixed-shape run (`results/b300_e24_v18_order128.txt`, T=128 tunes first as any warmup/unit
test would): v16 bwd 2.615 / 9.032 ms @8192/@32768 vs v18 2.150 / 7.171 ms = **1.22× / 1.26×
backward**. correctness PASS. Learnings: on B300 the dW1 kernel wants BK=128-deep tiles at
large TK/E (an E20-added config) but BK=32 at small TK/E — the single shared cache entry was
masking the E20 config-space win; E13 (deep-K dW) was "neutral" on H100 partly for this
exact reason.

v18 kernel breakdown @T=8192 full (µs of 2942): dW1 653 (22%), bwd_down 614 (21%), up_proj
524 (18%), dX 394 (13%), dW2 371 (13%), down_proj 261 (9%), gather 96×2 (3%). Remaining
structural ideas (split up-proj into single-dot GEMM + separate SwiGLU: ~2.5% expected but
regresses inference memory; de-atomic dS: µs-level) are below the noise-vs-risk bar →
stop the variant loop per the diminishing-returns rule. Ship = v15 fix + v16 config spaces
+ v18 keys (all three live in `variants/v18_dwkey.py`).

### E25 — Ship to src + validation on B300

Shipped to `src/liger_kernel/ops/fused_moe{,_kernels}.py`: (1) merged-K dX kernel (v15,
unconditional — it is a correctness fix on sm100/sm103 and a neutral-to-positive
simplification elsewhere); (2) Blackwell config-space extras (v16) as `_blackwell_config`-
tagged entries that the `_make_tile_prune` hooks drop at launch when
`infer_device_arch() not in ("blackwell", "blackwell_ultra")` — the arch check runs at
first launch, NOT at import (importing liger_kernel must not initialize CUDA), and other
archs keep byte-identical search spaces; (3) TPE_BUCKET dW autotune keys (v18, arch-neutral);
(4) BLOCK_H=1024 token-gather configs (universal).

Validation:
- `pytest test/transformers/test_fused_moe.py`: **22 passed** (19:00, autotune-dominated) —
  the suite that FAILED at E19 on the unmodified checkout.
- `make checkstyle`: green. CPU-only import smoke test: OK (no CUDA touch at import).
- `LIGER_FUSED_MOE_AUTOTUNE=0` pinned-config path: correctness PASS.
- Per-config dX sweep post-fix (`probe_src_dx_configs.py`): all runnable configs correct
  across {research shape, mini-model shape} × {fp32, bf16} × {TMA on/off}.
- fp32 convergence `test_mini_model[mini_qwen3_moe]`: PASS.
- Robustness (`results/b300_robustness.txt`): skew=1.5 full **1.65×** vs v0; Mixtral-8x7B
  shape fwd **1.45×** / full **1.62×**; CUDA-graph inference capture+replay and
  `make_graphed_callables` fwd+bwd both OK.
- `benchmark/data/all_benchmark_data.csv` refreshed with B300 rows for both sweep dims
  (T and num_experts); H100 rows retained (gpu_name column distinguishes).

**Known-issue note (pre-existing, NOT this kernel):** bf16 convergence
`test_mini_model[mini_qwen3_moe]` fails on this B300 box with 2 of 32 loss steps outside
rtol=0.2 (late steps 29-32) — identically on the pristine aeab3e3 checkout. The convergence
suite never exercises fused_moe (no `fused_moe` reference in `monkey_patch.py`; the MoE
block runs stock HF + LigerQwen3MoeSwiGLUMLP), the same test passes in fp32 and the dense
mini_qwen3 passes in bf16, so this is late-step bf16 training chaos under tcgen05
accumulation order vs the H100-calibrated tolerances — upstream tolerance work, out of
scope here.

**Environment gotcha (documented for the next box):** this AMI exports
`LD_LIBRARY_PATH=/usr/local/cuda/...` (system CUDA 13.2), which mid-session started
shadowing the pip `nvidia-cublas` (13.1) libraries and crashed every cuBLASLt user with
"Invalid handle. Cannot load symbol cublasLtGetVersion" (torch matmul → abort). All
validation above ran with `LD_LIBRARY_PATH=` cleared; the Triton kernels themselves are
unaffected (no cuBLAS dependency).

## FINAL RESULT (Phase B, B300)

PR-#1179 baseline (v0) → shipped src on B300 (`results/b300_final_vs_v0.txt`), Qwen3-MoE-30B
shard, bf16, same harness as the H100 table at the top of this file:

| T | fwd | bwd | full step | inference fwd |
|---|-----|-----|-----------|---------------|
| 128 | 0.353→0.304 (**1.16×**) | 0.938→0.553 (**1.70×**) | 1.314→0.774 (**1.70×**) | 0.352→0.285 (**1.24×**) |
| 1024 | 0.395→0.312 (**1.27×**) | 1.310→0.738 (**1.78×**) | 1.719→1.023 (**1.68×**) | 0.396→0.289 (**1.37×**) |
| 8192 | 1.323→0.924 (**1.43×**) | 5.060→2.150 (**2.35×**) | 6.351→3.066 (**2.07×**) | 1.337→0.923 (**1.45×**) |
| 32768 | 4.994→3.152 (**1.58×**) | 19.829→7.147 (**2.77×**) | 24.345→10.344 (**2.35×**) | 5.100→3.201 (**1.59×**) |

vs the H100 final (same code lineage): full step @32768 19.005→10.344 ms box-to-box, and
the *relative* win over v0 grew from 1.91× (H100) to 2.35× (B300). Peak memory unchanged.
And unlike the checkout this phase started from, **the gradients are right**: the headline
deliverable is the sm103 miscompile fix (E19) — everything on Blackwell silently trained on
wrong dx before it.

Phase B experiment ledger: E19 fix (shipped), E20 config spaces (shipped), E21 TMA A/B
(keep, no change), E22 eviction hints (REJECTED), E23 memeff cost re-measure (docs only),
E24 dW keys (shipped, 1.22-1.26× bwd in mixed-shape runs), E25 ship+validate. Base rate
this phase: 2 of 4 measured hypotheses shipped; the E18 backlog's top three ideas
(WS+gather4, BM=256, clusters) all died in probes before reaching a variant — cheap kills,
exactly what the probe-first ordering is for.
