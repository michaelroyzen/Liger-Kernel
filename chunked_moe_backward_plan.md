# Token-chunked MoE backward with deterministic dX reduction

Implementation plan for bounding the fused-MoE backward's transient memory. Self-contained:
no prior context needed beyond this repo (branch `triton-moe-tuning-clean`).

## 1. Motivation

### The failure that started this

GRPO training of Qwen3.5-35B-A3B (4 training ranks, B300 275 GiB, ZeRO-3, ~50k-token
prompts) crashed at `BATCH_SIZE=8` with:

```
File "src/liger_kernel/ops/fused_moe.py", line 534, in backward
    dx_expanded = torch.empty(TK, H, dtype=dO.dtype, device=dO.device)
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 17.18 GiB
(251.71 GiB already in use)
```

`dx_expanded` is one of a *family* of buffers in `LigerFusedMoEFunction` that scale with
`TK = tokens_per_microbatch x top_k`. At the crashing workload (~563k tokens x K=8 =
4.5M assignments; H=2048, I=512):

| buffer | shape | size | when |
|---|---|---|---|
| `dx_expanded` | TK x H bf16 | 17.2 GiB | backward transient |
| `d_pre_act` | TK x 2I bf16 | 9.2 GiB | backward transient |
| `pre_act` | TK x 2I bf16 | 9.2 GiB | stored fwd → bwd (training only) |
| `post_act` / `weighted_act` | TK x I bf16 | 4.6 GiB each | fwd transient / bwd input |

Under activation checkpointing only one layer's workspace is live at a time, but that
workspace + ZeRO-3 sharded states is exactly what sets the per-rank memory peak in
long-context training. These buffers are the peak.

### Why it gets worse

Per-token buffer cost is `K x dim x 2 bytes` and grows with model size
(configs from the HF model cards):

| model | H | I (moe) | routed K | `dx_expanded` @563k tokens | full TK-family @563k tokens |
|---|---|---|---|---|---|
| Qwen3.5-35B-A3B | 2048 | 512 | 8 | 17.2 GiB | ~45 GiB |
| Qwen3.5-122B-A10B | 3072 | 1024 | 8 | 25.8 GiB | ~80 GiB |
| Qwen3.5-397B-A17B | 4096 | 1024 | 10 | 43.0 GiB | ~130 GiB |

For the 35B, config levers absorb the problem (smaller micro-batch,
`LIGER_FUSED_MOE_MEMORY_EFFICIENT=1`, `PYTORCH_ALLOC_CONF=expandable_segments:True`).
For the 122B they are strained; for the 397B this redesign is a prerequisite.

### Goal

Cap every TK-proportional buffer at a fixed chunk size (config-independent, workload-
independent), while:

1. keeping expert-grouped GEMMs on tensor cores (no per-token degenerate GEMMs),
2. **not expanding the layer's non-deterministic surface** (see §4.1),
3. preserving the no-CPU-sync property of the routing metadata design,
4. costing ≤ ~15% on the layer backward (aggregation is GEMM-shadowed).

Precedent: `src/liger_kernel/ops/chunked_grpo_loss.py` applies the same philosophy to
the GRPO loss (never materialize the big intermediate; recompute per tile in backward;
fp32 cross-chunk accumulators; fixed launch configs; bitwise-reproducible). Read it
first — this plan is that pattern applied to the MoE block.

### Alternatives considered and rejected

- **fp32 `tl.atomic_add` epilogue into `dx`** (megablocks-style): fastest and smallest
  (kills `dx_expanded` in one pass), but each token's K=8 contributions arrive in
  scheduler order → `dx` becomes bitwise non-reproducible. Today the layer's only
  float atomic is `dS` (router-score grads, ~2–4 partials/entry, low stakes); an atomic
  `dx` would put non-determinism on the tensor that back-propagates into *everything*.
  Also solves only `dx_expanded`, not the buffer family.
- **Serializing experts** (per-expert scatter is race-free since a token appears at most
  once per expert): 256–512 sequential waves with heavy load imbalance; unacceptable.
- **Fixed-point (int64) atomic accumulation**: order-independent (integer adds are
  associative) hence deterministic *and* fast, but requires gradient dynamic-range /
  scale management with silent-saturation risk across training regimes. Held in
  reserve for `dS` if formal cross-rank bitwise reproducibility is ever required.

## 2. Current architecture (what you're modifying)

Files: `src/liger_kernel/ops/fused_moe.py` (orchestration + autograd Function),
`src/liger_kernel/ops/fused_moe_kernels.py` (Triton kernels). Line numbers as of
commit `0a0a852`.

### Routing metadata (`fused_moe.py:96` `compute_routing_metadata`)

Device-side only (no `.item()` anywhere):

- `x_gather_idx (TK,)` — for sorted assignment row r, which token to read
- `s_scatter_idx / s_reverse_scatter_idx (TK,)` — maps (token t, slot k) ↔ sorted row
- `tile_row_start (num_m_tiles,)`, `tile_expert (num_m_tiles,)` — each BLOCK_M-row GEMM
  tile's start row and owning expert; **a tile never spans two experts**
- `expert_tile_offset (E+1,)` — cumsum of per-expert tile counts;
  `total_tiles_dev = expert_tile_offset[E:]` (`fused_moe.py:287`) is a 1-element device
  scalar; kernels launch on a host-side upper bound and early-exit past
  `tl.load(total_tiles_ptr)` (`fused_moe_kernels.py:536` and analogues). Preserve this.

### Forward

- `_fused_up_proj_swiglu_kernel` (`fused_moe_kernels.py:499`): gathers x rows in-kernel
  via `x_gather_idx`, grouped GEMM vs `W1[expert]`, SwiGLU; stores `post_act` always and
  `pre_act` only when training (inference skips the store).
- `_fused_down_proj_kernel` (`:639`): grouped GEMM vs `W2[expert]` → per-assignment
  outputs.
- `_token_gather_weighted_sum_kernel` (`:748`): token-gridded; each program gathers its
  K rows (via reverse-scatter metadata) and computes the weighted sum → `out (T, H)`.
  **Deterministic single-writer-per-token pattern — the template for the new dX apply.**

### Backward (`fused_moe.py:380`)

1. `_moe_bwd_down_proj_kernel` (`:813`): computes `d_post_act`, applies SwiGLU backward
   → `d_pre_act`, writes `weighted_act`, and accumulates `dS` via **fp32 `tl.atomic_add`**
   (`:928`; buffer allocated fp32 at `fused_moe.py:465` — the PR comment explains bf16
   atomics would round every partial).
2. dW1/dW2 kernels (autotune configs at `:403`): program-local split-K accumulation, **no
   atomics**; `reset_to_zero` annotations exist because autotune re-runs candidate configs.
3. dX: `dx_expanded = torch.empty(TK, H)` (`fused_moe.py:534`), expert-grouped GEMM
   `d_pre_act @ W1[expert]` writes per-assignment rows, then a token-gather reduction
   (same pattern as forward) produces `dx (T, H)`.

### Atomics inventory (important for §4.1)

- int32 router histogram (`fused_moe_kernels.py:159`) — integer, order-independent,
  deterministic. Fine.
- fp32 `dS` (`:928`) — the **only** float atomic in the layer. Leave unchanged.
- dW kernels: no atomics (despite `reset_to_zero`, which is autotune-related).

### Existing memory knob

`LIGER_FUSED_MOE_MEMORY_EFFICIENT=1`: in-place SwiGLU backward + `weighted_act`
recompute (~TK x 3I bytes saved, 10–15% slower backward, breaks `retain_graph`).
Chunked mode supersedes it — define precedence explicitly (suggest: chunked mode
implies it / ignores it; document either way).

## 3. Design: chunked backward + deterministic segmented reduce

### 3.1 Chunk axis: GEMM tiles, not raw rows

Chunk over **tile index ranges** `[tile_lo, tile_hi)` of the expert-grouped M-tiles.
Rationale:

- tiles are BLOCK_M-aligned and single-expert, so chunking never splits a GEMM tile and
  never splits a (tile → expert) mapping; existing kernels need only a `tile_base`
  offset added to their `pid → tile` computation,
- the host knows the tile-count **upper bound** (`tile_row_start.shape[0]`) without any
  sync; per-chunk grids use `min(CHUNK_TILES, upper_bound - tile_lo)` and programs keep
  the existing early-exit against `total_tiles_dev`,
- chunk row capacity is `CHUNK_TILES x BLOCK_M` rows → staging buffers have a fixed,
  host-known allocation size.

An expert *may* span multiple chunks (its tiles split across the boundary). That is
fine for every consumer (see dW and dX handling below).

### 3.2 Backward chunk loop (python-level, serial)

Allocate once, before the loop (sizes are chunk-capacity, not TK):

- `pre_act_chunk (C, 2I)` bf16 (recomputed; see 3.4)
- `d_pre_act_chunk (C, 2I)` bf16
- `weighted_act_chunk (C, I)` bf16
- `dx_staging (C, H)` bf16
- `dx (T, H)` **fp32** accumulator (final cast to `dO.dtype`)
- `dW1 (E, 2I, H)`, `dW2 (E, H, I)` **fp32** accumulators (final cast)
- `dS (TK,)` fp32 — unchanged, full-length (it is 1-D and small: 9 MB at 4.5M)

Per chunk `c`:

1. **Recompute** `pre_act_chunk` from `x` + `W1` (up-proj kernel with `tile_base`,
   writing to the staging buffer at `row - chunk_row_lo`).
2. Down-proj backward kernel (with `tile_base`): `d_post_act` → SwiGLU backward →
   `d_pre_act_chunk`, `weighted_act_chunk`, `dS` (atomic, global indices — unchanged).
3. **dW accumulation**: dW kernels over the chunk's tiles, *adding* into the fp32
   accumulators. Serial chunk loop ⇒ deterministic order. See §4.2 for the autotune
   interaction (this is the subtlest correctness point in the plan).
4. **dX GEMM**: `dx_staging[0:rows_c] = d_pre_act_chunk @ W1[expert]` (existing
   expert-grouped kernel + `tile_base`, writing chunk-local rows).
5. **dX apply (the deterministic segmented reduce)**: extend
   `_token_gather_weighted_sum_kernel` into a `_token_gather_add_kernel`:
   - token-gridded (grid over T x H-tiles), single writer per `dx[t]` → no atomics;
   - each program looks up its K global sorted-row positions (existing reverse-scatter
     metadata), keeps those with `chunk_row_lo <= pos < chunk_row_hi`, gathers them
     from `dx_staging[pos - chunk_row_lo]`, and **adds** into the fp32 `dx[t]`;
   - tokens with zero in-chunk rows exit after K comparisons (cheap; at T=563k and
     ~9 chunks that is ~5M mostly-idle lightweight programs — measure, and if it shows
     up, build per-chunk token lists from the metadata as a follow-up optimization).

Chunks processed in fixed ascending order ⇒ every accumulation (`dx`, `dW*`) has a
fixed association order ⇒ bitwise reproducible run-to-run and rank-to-rank.

### 3.3 What this eliminates

- `dx_expanded (TK, H)` → `dx_staging (C, H)`
- `d_pre_act (TK, 2I)` → chunk-sized
- `pre_act (TK, 2I)` **stored activation from forward** → gone entirely (recomputed
  per chunk in backward). Forward then never stores it → forward memory drops too and
  the training/inference forward paths converge.
- `weighted_act (TK, I)` → chunk-sized.

New costs: fp32 `dx` (T x H x 4; 4.3 GiB at the crashing workload — vs 17.2 saved on
`dx_expanded` alone), fp32 dW accumulators (E x 2I x H + E x H x I; ~4.3 GiB at 35B dims
— constant in workload), one extra up-proj GEMM pass (the recompute), and one extra
read/write of chunk staging per chunk.

Net at the crashing workload: ~45 GiB of TK-scaled buffers → ~10–12 GiB of fixed-size
accumulators + ~2–4 GiB of chunk staging. At 397B dims the delta is larger.

### 3.4 Chunk size

`LIGER_FUSED_MOE_CHUNK_TILES` env var (or module constant), expressed in tiles.
Suggested default: whatever yields ~512k–1M assignment rows (~2–4 GiB of staging at
H=2048–4096). `0` = disabled → current unchunked path, byte-for-byte untouched.
**Ship with default off**; flip after A/B validation (§5.4).

## 4. Things to keep in mind

### 4.1 Determinism rules (this codebase takes them seriously)

- No new float atomics. `dS` stays as-is (already fp32, already atomic, small surface).
- All new accumulation must be single-writer or serialized-by-chunk.
- Add a run-twice bitwise test (template:
  `test/transformers/test_chunked_grpo_loss.py::test_bitwise_determinism`).
- Note: chunked results will NOT be bitwise-equal to the *unchunked* path (different
  summation order). Compare against unchunked with tolerances (see §5.2) and against
  itself bitwise.

### 4.2 Autotune x in-place accumulation — the biggest trap

Triton autotune benchmarks each candidate config by *re-running the kernel*. For a
kernel that **adds into** a live accumulator (the chunked dW kernels, the dX apply
kernel), config trials double-add garbage. The existing kernels handle this with
`reset_to_zero=["dW2_ptr"]`, which zeroes the buffer between trials — but with
cross-chunk accumulation, `reset_to_zero` on chunk N would wipe chunks 0..N-1.

Options (pick one, document it):

1. **Fixed launch configs for accumulating kernels** (no autotune) — simplest, matches
   `chunked_grpo_loss.py` practice. Recommended for v1.
2. Autotune only on a warmup call against scratch buffers before the real loop.
3. Per-chunk partial buffer + separate add (extra memory + pass — defeats the purpose).

Also ensure every chunk presents **identical constexpr shapes** (pad the last chunk's
grid rather than specializing it) so no re-tuning triggers mid-loop even if autotune is
kept anywhere.

### 4.3 Triton-on-B300 (sm_103) landmines — both already documented in-repo

- **triton-lang/triton#10821**: two `tl.dot` chained through ONE accumulator in a K-loop
  miscompiles on tcgen05 tiles (BLOCK_M ≥ 64). The down-proj backward was restructured
  to avoid it (see comments near `fused_moe_kernels.py:764`); keep the
  one-dot-per-accumulator rule in any new/modified K-loop.
  `ops/chunked_grpo_loss.py` documents the same rule.
- **num_warps ≥ 4 raciness history** (FLA kernels, `triton_sm103_nondeterminism_bug_report.md`):
  if adding configs, stay within the `_blackwell_config`-gated families
  (`fused_moe_kernels.py:51`) and verify bitwise stability on B300.

### 4.4 No CPU syncs

Never `.item()` routing counts. Host-side chunk bounds must come from host-known upper
bounds (tile-array length, `TK`), with device-side early-exit handling the actual
counts (`total_tiles_dev` pattern). If you find yourself needing the real per-chunk row
count on the host, restructure instead.

### 4.5 Scope boundaries

- **TMA paths**: `_tma_eligibility` / `_ensure_tma_allocator` (`fused_moe.py:62–71`)
  feed device-descriptor paths in the GEMM kernels. Chunk `tile_base` offsets must not
  break descriptor assumptions; if it gets hairy, force the non-TMA path when chunking
  is enabled for v1 and note the perf gap.
- **Ascend backend** (`src/liger_kernel/ops/backends/_ascend/ops/fused_moe.py`) is a
  parallel implementation with the same `dx_expanded` allocation — out of scope; leave
  a TODO cross-reference.
- **Shared expert** is handled outside `LigerFusedMoEFunction` (dense path) — unaffected.
- Forward stays unchunked in v1 (its transients are smaller: no `dx_expanded`, and
  `pre_act` storage disappears as a side effect of backward recompute). Chunking the
  forward is a possible v2.
- `retain_graph` / double-backward: recompute-based backward makes re-backward invalid
  (same limitation as `LIGER_FUSED_MOE_MEMORY_EFFICIENT=1` — see its docstring). Guard
  or document.

### 4.6 dtype conventions (match existing practice)

Staging buffers in input dtype (bf16); cross-chunk accumulators (`dx`, `dW1`, `dW2`)
fp32, single cast to param/input dtype at the end. Rationale precedents:
the `dS` fp32-atomics comment, and `fused_linear_ppo.py` / `chunked_grpo_loss.py`
(GEMM operands in input dtype, fp32 cross-chunk buffers).

## 5. Validation plan

### 5.1 Existing assets

- `test/transformers/test_fused_moe.py` — reference Python-loop implementation
  (`_reference_moe_forward`), forward/backward parity tests, routing invariants,
  edge cases (empty experts, single expert, all-tokens-one-expert), dtypes. 22 tests.
- `benchmark/scripts/benchmark_fused_moe.py`.
- `chunked_triton_grpo_head_to_head.md` + `benchmark/scripts/benchmark_chunked_grpo_loss_head_to_head.py`
  as the model for how to present before/after numbers.

### 5.2 Correctness matrix (parametrize the existing suite over chunked mode)

- vs fp32 reference loop: `out`, `dx`, `dw1`, `dw2`, `dtkw` with existing tolerances,
  chunked and unchunked.
- chunked vs unchunked: tolerance-based (summation order differs — do NOT expect bitwise).
- Chunk-boundary cases: chunk = 1 tile; chunk > total tiles; chunk exactly total;
  expert spanning a chunk boundary; token whose K assignments span chunks; last-chunk
  ragged tail; E with zero-token experts.
- Determinism: chunked run twice → bitwise-equal `dx/dW/out` (dS/dtkw excluded — it
  keeps its pre-existing atomic).
- Dims: 35B (H2048/I512/E256/K8), 122B (H3072/I1024/E256/K8), 397B
  (H4096/I1024/E512/K10), plus a tiny config for fast CI.
- Memory assertion: `torch.cuda.max_memory_allocated` during backward capped at
  expected staging+accumulator budget at a large TK (prove the point of the project).

### 5.3 Performance bar

At 35B dims, T=256k–563k tokens: layer backward regression ≤ ~15% vs unchunked
(aggregation extra passes are GEMM-shadowed; the recompute GEMM is the main real cost —
it replaces the `pre_act` store/load, so measure net). Report a table like the GRPO
head-to-head (time + peak memory, chunked/unchunked, several TK).

### 5.4 Integration validation

In the monorepo training setup (`training/scripts/run_train_grpo_multi.sh`,
Qwen3.5-35B): 5–10 GRPO steps chunked vs unchunked on the same data/seed — loss curves
overlap within run-to-run noise, per-step `max_memory_allocated` drops in the chunked
run at long-completion batches. Then flip the default.

## 6. Suggested sequencing

1. Fixed-config (no-autotune) chunked dW + chunked dX GEMM + `_token_gather_add_kernel`,
   `pre_act` still stored (skip recompute) — smallest diff proving the reduce design.
2. Add per-chunk `pre_act` recompute; delete the stored activation.
3. Memory/perf benchmark sweep; tune `CHUNK_TILES` default; decide autotune strategy
   (§4.2) if fixed configs leave perf on the table.
4. Full test matrix (§5.2) + bitwise determinism test.
5. Training A/B (§5.4), flip default, update README/docstrings and
   `LIGER_FUSED_MOE_MEMORY_EFFICIENT` interplay note.
