"""
Tests for LigerFusedMoEFunction.

Tests cover:
1. Routing metadata correctness (permutation invariants)
2. Forward correctness vs. reference Python loop
3. Backward / gradient correctness
4. Edge cases (empty experts, single expert, all tokens to one expert)
5. Multiple dtypes
6. Token-chunked backward (LIGER_FUSED_MOE_CHUNK_TILES): correctness vs the
   unchunked path and the reference, chunk-boundary cases, bitwise determinism,
   and the backward-memory cap that is the point of the mode
"""

import pytest
import torch
import torch.nn as nn

from liger_kernel.ops import LigerFusedMoEFunction
from liger_kernel.utils import infer_device

device = infer_device()

if device == "npu":
    from liger_kernel.ops.backends._ascend.ops.fused_moe import compute_routing_metadata
else:
    from liger_kernel.ops.fused_moe import compute_routing_metadata


# ---------------------------------------------------------------------------
# Reference implementation (original Python loop)
# ---------------------------------------------------------------------------


def _reference_moe_forward(x, gate_up_proj, down_proj, top_k_index, top_k_weights):
    """Reference: Python loop over active experts (original LigerExperts logic)."""
    T, H = x.shape
    E = gate_up_proj.shape[0]
    final = torch.zeros_like(x)

    with torch.no_grad():
        expert_mask = torch.nn.functional.one_hot(top_k_index.long(), num_classes=E)
        # top_k_index (T, K) → one_hot (T, K, E) → permute (E, K, T)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for eh in expert_hit:
        eidx = eh[0]
        top_k_pos, token_idx = torch.where(expert_mask[eidx])
        curr = x[token_idx]
        gate, up = nn.functional.linear(curr, gate_up_proj[eidx]).chunk(2, dim=-1)
        curr = nn.functional.silu(gate) * up
        curr = nn.functional.linear(curr, down_proj[eidx])
        curr = curr * top_k_weights[token_idx, top_k_pos, None]
        final.index_add_(0, token_idx, curr.to(final.dtype))

    return final


def _make_inputs(T, E, H, intermediate_dim, K, dtype, device, seed=42):
    torch.manual_seed(seed)
    x = torch.randn(T, H, dtype=dtype, device=device)
    gate_up_proj = torch.randn(E, 2 * intermediate_dim, H, dtype=dtype, device=device) * 0.02
    down_proj = torch.randn(E, H, intermediate_dim, dtype=dtype, device=device) * 0.02
    # Random top-k routing (uniform distribution)
    logits = torch.randn(T, E, device=device)
    top_k_index = torch.topk(logits, K, dim=-1).indices.to(torch.int32)
    top_k_weights = torch.softmax(torch.gather(logits, 1, top_k_index.long()), dim=-1).to(dtype)
    return x, gate_up_proj, down_proj, top_k_index, top_k_weights


# ---------------------------------------------------------------------------
# Test 1: Routing metadata invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "T, E, K",
    [
        (64, 8, 2),
        (128, 16, 4),
        (100, 8, 2),  # non-power-of-2 T
        (256, 32, 8),
    ],
)
def test_routing_metadata_invariants(T, E, K):
    torch.manual_seed(0)
    logits = torch.randn(T, E, device=device)
    top_k_index = torch.topk(logits, K, dim=-1).indices.to(torch.int32)

    expert_freq, expert_freq_offset, x_gather_idx, s_scatter_idx, s_rev_scatter_idx, *_ = compute_routing_metadata(
        top_k_index, E
    )

    TK = T * K

    # Inverse permutation: s_rev[s_scatter[i]] == i
    reconstructed = s_rev_scatter_idx[s_scatter_idx.long()]
    assert torch.all(reconstructed == torch.arange(TK, device=device, dtype=torch.int32)), (
        "s_reverse_scatter_idx is not the inverse of s_scatter_idx"
    )

    # expert_freq_offset[e+1] - expert_freq_offset[e] == expert_frequency[e]
    freq_from_offset = expert_freq_offset[1:] - expert_freq_offset[:-1]
    assert torch.all(freq_from_offset == expert_freq), "expert_freq_offset does not match expert_frequency"

    # Total tokens: offset[E] == TK
    assert int(expert_freq_offset[-1]) == TK

    # Expert frequencies sum to TK
    assert int(expert_freq.sum()) == TK

    # x_gather_idx values are in [0, T)
    assert x_gather_idx.min() >= 0 and x_gather_idx.max() < T


# ---------------------------------------------------------------------------
# Test 2: Forward + backward correctness vs. reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "T, E, H, intermediate_dim, K",
    [
        (7, 4, 64, 32, 2),  # T < BLOCK_M_TOKEN: tile row-mask is mostly padding (unique sub-tile edge)
        (512, 8, 256, 128, 2),  # multi-tile baseline: T*K/E=128 → 2 tiles/expert
        (512, 8, 97, 47, 2),  # multi-tile + odd H/I: tail masking across tile boundaries
        (512, 7, 128, 64, 3),  # multi-tile + prime E: non-pow2 grid decomposition
        (512, 8, 256, 64, 1),  # multi-tile + K=1: no weighted sum in token aggregation
        (128, 8, 256, 64, 8),  # multi-tile + K=E: maximum routing density
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-3, 1e-4),
        (torch.bfloat16, 1e-1, 1e-2),
    ],
)
def test_correctness(T, E, H, intermediate_dim, K, dtype, atol, rtol):
    x, gate_up_proj, down_proj, top_k_index, top_k_weights = _make_inputs(T, E, H, intermediate_dim, K, dtype, device)

    ref = _reference_moe_forward(x, gate_up_proj, down_proj, top_k_index, top_k_weights)
    out = LigerFusedMoEFunction.apply(x, gate_up_proj, down_proj, top_k_index, top_k_weights)

    assert out.shape == ref.shape, f"Shape mismatch: {out.shape} vs {ref.shape}"
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)

    if dtype == torch.float32:
        x1 = x.clone().requires_grad_(True)
        gup1 = gate_up_proj.clone().requires_grad_(True)
        dn1 = down_proj.clone().requires_grad_(True)
        wts1 = top_k_weights.clone().requires_grad_(True)
        x2 = x.clone().requires_grad_(True)
        gup2 = gate_up_proj.clone().requires_grad_(True)
        dn2 = down_proj.clone().requires_grad_(True)
        wts2 = top_k_weights.clone().requires_grad_(True)

        out_ref = _reference_moe_forward(x1, gup1, dn1, top_k_index, wts1)
        out_ref.sum().backward()
        out_fused = LigerFusedMoEFunction.apply(x2, gup2, dn2, top_k_index, wts2)
        out_fused.sum().backward()

        b_atol, b_rtol = 3e-3, 1e-2
        torch.testing.assert_close(wts2.grad, wts1.grad, atol=b_atol, rtol=b_rtol)
        torch.testing.assert_close(dn2.grad, dn1.grad, atol=b_atol, rtol=b_rtol)
        torch.testing.assert_close(x2.grad, x1.grad, atol=b_atol, rtol=b_rtol)
        torch.testing.assert_close(gup2.grad, gup1.grad, atol=b_atol, rtol=b_rtol)


# ---------------------------------------------------------------------------
# Test 3: Edge cases (forward correctness)
# ---------------------------------------------------------------------------


def test_all_tokens_to_one_expert():
    """All tokens route to expert 0; all others empty."""
    T, E, H, intermediate_dim, K = 32, 8, 64, 32, 2
    dtype = torch.float32
    torch.manual_seed(0)

    x = torch.randn(T, H, dtype=dtype, device=device)
    gate_up_proj = torch.randn(E, 2 * intermediate_dim, H, dtype=dtype, device=device) * 0.02
    down_proj = torch.randn(E, H, intermediate_dim, dtype=dtype, device=device) * 0.02
    # Force all tokens to expert 0 and 1
    top_k_index = torch.zeros(T, K, dtype=torch.int32, device=device)
    top_k_weights = torch.ones(T, K, dtype=dtype, device=device) / K

    # Should not crash
    out = LigerFusedMoEFunction.apply(x, gate_up_proj, down_proj, top_k_index, top_k_weights)
    ref = _reference_moe_forward(x, gate_up_proj, down_proj, top_k_index, top_k_weights)

    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-4)


def test_single_token():
    """T=1 edge case."""
    T, E, H, intermediate_dim, K = 1, 4, 32, 16, 2
    dtype = torch.float32
    x, gate_up_proj, down_proj, top_k_index, top_k_weights = _make_inputs(T, E, H, intermediate_dim, K, dtype, device)
    out = LigerFusedMoEFunction.apply(x, gate_up_proj, down_proj, top_k_index, top_k_weights)
    ref = _reference_moe_forward(x, gate_up_proj, down_proj, top_k_index, top_k_weights)
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-4)


def test_K_equals_E():
    """K == E: every token routes to every expert."""
    T, E, H, intermediate_dim, K = 16, 4, 32, 16, 4
    dtype = torch.float32
    x, gate_up_proj, down_proj, top_k_index, top_k_weights = _make_inputs(T, E, H, intermediate_dim, K, dtype, device)
    out = LigerFusedMoEFunction.apply(x, gate_up_proj, down_proj, top_k_index, top_k_weights)
    ref = _reference_moe_forward(x, gate_up_proj, down_proj, top_k_index, top_k_weights)
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-4)


def test_inference_no_grad_path():
    """No input requires grad → the fast inference path (no pre_act store) must
    produce the same output as the training path."""
    T, E, H, intermediate_dim, K = 256, 8, 128, 64, 2
    dtype = torch.float32
    x, gate_up_proj, down_proj, top_k_index, top_k_weights = _make_inputs(T, E, H, intermediate_dim, K, dtype, device)

    with torch.no_grad():
        out_infer = LigerFusedMoEFunction.apply(x, gate_up_proj, down_proj, top_k_index, top_k_weights)

    x2 = x.clone().requires_grad_(True)
    out_train = LigerFusedMoEFunction.apply(x2, gate_up_proj, down_proj, top_k_index, top_k_weights)
    torch.testing.assert_close(out_infer, out_train, atol=1e-5, rtol=1e-5)


def test_retain_graph_second_backward():
    """Default mode: backward twice over the same graph (retain_graph) must work
    and produce identical gradients (no in-place clobbering of saved tensors)."""
    T, E, H, intermediate_dim, K = 64, 4, 64, 32, 2
    dtype = torch.float32
    x, gate_up_proj, down_proj, top_k_index, top_k_weights = _make_inputs(T, E, H, intermediate_dim, K, dtype, device)
    x = x.requires_grad_(True)
    out = LigerFusedMoEFunction.apply(x, gate_up_proj, down_proj, top_k_index, top_k_weights)
    out.sum().backward(retain_graph=True)
    g1 = x.grad.clone()
    x.grad = None
    out.sum().backward()
    torch.testing.assert_close(x.grad, g1)


_MEM_EFFICIENT_SUBPROCESS_SCRIPT = """
import torch
import torch.nn as nn

from liger_kernel.ops import LigerFusedMoEFunction

from test.transformers.test_fused_moe import _make_inputs, _reference_moe_forward

device = "cuda"
T, E, H, intermediate_dim, K = 128, 4, 64, 32, 2
x, gup, dn, idx, wts = _make_inputs(T, E, H, intermediate_dim, K, torch.float32, device)

x1, gup1, dn1, wts1 = (t.detach().clone().requires_grad_(True) for t in (x, gup, dn, wts))
x2, gup2, dn2, wts2 = (t.detach().clone().requires_grad_(True) for t in (x, gup, dn, wts))

ref = _reference_moe_forward(x1, gup1, dn1, idx, wts1)
ref.sum().backward()
out = LigerFusedMoEFunction.apply(x2, gup2, dn2, idx, wts2)
out.sum().backward(retain_graph=True)

torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-4)
for a, b in [(x2, x1), (gup2, gup1), (dn2, dn1), (wts2, wts1)]:
    torch.testing.assert_close(a.grad, b.grad, atol=3e-3, rtol=1e-2)

# in-place mode: a second backward over the same graph must raise, not corrupt
try:
    out.sum().backward()
except RuntimeError as e:
    assert "modified by an inplace operation" in str(e), e
else:
    raise AssertionError("second backward did not raise in memory-efficient mode")
print("MEM_EFFICIENT_OK")
"""


@pytest.mark.skipif(device != "cuda", reason="subprocess script assumes cuda")
def test_memory_efficient_mode():
    """LIGER_FUSED_MOE_MEMORY_EFFICIENT=1 (import-time flag → subprocess):
    gradients must match the reference, and a second backward must raise (SwiGLU
    backward runs in place over the saved pre-activations, guarded by a version
    bump)."""
    import os
    import subprocess
    import sys

    env = dict(os.environ, LIGER_FUSED_MOE_MEMORY_EFFICIENT="1")
    result = subprocess.run(
        [sys.executable, "-c", _MEM_EFFICIENT_SUBPROCESS_SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "MEM_EFFICIENT_OK" in result.stdout


# ---------------------------------------------------------------------------
# Test 4: Token-chunked backward (LIGER_FUSED_MOE_CHUNK_TILES)
#
# The env var is read at forward-call time and captured in ctx, so chunking
# can be toggled per test via monkeypatch (no subprocess needed).
# ---------------------------------------------------------------------------


def _run_fwd_bwd(x, gate_up_proj, down_proj, top_k_index, top_k_weights, retain_graph=False):
    """One forward+backward; returns (out, dx, dW1, dW2, dS) detached."""
    x1 = x.detach().clone().requires_grad_(True)
    gup1 = gate_up_proj.detach().clone().requires_grad_(True)
    dn1 = down_proj.detach().clone().requires_grad_(True)
    wts1 = top_k_weights.detach().clone().requires_grad_(True)
    out = LigerFusedMoEFunction.apply(x1, gup1, dn1, top_k_index, wts1)
    out.sum().backward(retain_graph=retain_graph)
    return out.detach(), x1.grad, gup1.grad, dn1.grad, wts1.grad


_GRAD_NAMES = ("out", "dx", "dgate_up_proj", "ddown_proj", "dtopk_weights")


@pytest.mark.parametrize(
    "T, E, H, intermediate_dim, K",
    [
        (512, 8, 256, 128, 2),  # T*K/E=128, BLOCK_M=128 → 1 tile/expert: windows split experts exactly
        (512, 8, 97, 47, 2),  # odd H/I: tail masking inside the staging buffers
        (512, 7, 128, 64, 3),  # prime E + K=3 (non-pow2 K in the dx-apply kernel)
        (128, 8, 256, 64, 8),  # K=E: every token's assignments span many windows
        (7, 4, 64, 32, 2),  # T < BLOCK_M: single ragged tile
    ],
)
@pytest.mark.parametrize("chunk_tiles", [1, 3, 10_000])  # 1 tile, ragged tail, chunk > total tiles
def test_chunked_backward_matches_unchunked(T, E, H, intermediate_dim, K, chunk_tiles, monkeypatch):
    """Chunked backward must match the unchunked path (tolerance-based: the fp32
    cross-chunk accumulators change the summation order, so bitwise equality
    with the unchunked path is NOT expected) and the fp32 reference loop."""
    dtype = torch.float32
    x, gate_up_proj, down_proj, top_k_index, top_k_weights = _make_inputs(T, E, H, intermediate_dim, K, dtype, device)

    monkeypatch.delenv("LIGER_FUSED_MOE_CHUNK_TILES", raising=False)
    unchunked = _run_fwd_bwd(x, gate_up_proj, down_proj, top_k_index, top_k_weights)

    monkeypatch.setenv("LIGER_FUSED_MOE_CHUNK_TILES", str(chunk_tiles))
    chunked = _run_fwd_bwd(x, gate_up_proj, down_proj, top_k_index, top_k_weights)

    for name, got, want in zip(_GRAD_NAMES, chunked, unchunked):
        torch.testing.assert_close(got, want, atol=3e-3, rtol=1e-2, msg=lambda m, n=name: f"{n}: {m}")

    # And against the reference implementation's autograd.
    x1 = x.detach().clone().requires_grad_(True)
    gup1 = gate_up_proj.detach().clone().requires_grad_(True)
    dn1 = down_proj.detach().clone().requires_grad_(True)
    wts1 = top_k_weights.detach().clone().requires_grad_(True)
    out_ref = _reference_moe_forward(x1, gup1, dn1, top_k_index, wts1)
    out_ref.sum().backward()
    ref = (out_ref.detach(), x1.grad, gup1.grad, dn1.grad, wts1.grad)
    for name, got, want in zip(_GRAD_NAMES, chunked, ref):
        torch.testing.assert_close(got, want, atol=3e-3, rtol=1e-2, msg=lambda m, n=name: f"{n} vs ref: {m}")


def test_chunked_backward_bf16(monkeypatch):
    """bf16 chunked grads track the unchunked bf16 grads within bf16 tolerances."""
    T, E, H, intermediate_dim, K = 512, 8, 256, 128, 4
    x, gate_up_proj, down_proj, top_k_index, top_k_weights = _make_inputs(
        T, E, H, intermediate_dim, K, torch.bfloat16, device
    )

    monkeypatch.delenv("LIGER_FUSED_MOE_CHUNK_TILES", raising=False)
    unchunked = _run_fwd_bwd(x, gate_up_proj, down_proj, top_k_index, top_k_weights)
    monkeypatch.setenv("LIGER_FUSED_MOE_CHUNK_TILES", "2")
    chunked = _run_fwd_bwd(x, gate_up_proj, down_proj, top_k_index, top_k_weights)

    for name, got, want in zip(_GRAD_NAMES, chunked, unchunked):
        torch.testing.assert_close(got, want, atol=1e-1, rtol=1e-2, msg=lambda m, n=name: f"{n}: {m}")


def test_chunked_backward_zero_token_experts(monkeypatch):
    """All tokens on expert 0 → every other expert is empty in every window;
    their dW rows must stay exactly zero and grads must match the reference."""
    T, E, H, intermediate_dim, K = 32, 8, 64, 32, 2
    dtype = torch.float32
    torch.manual_seed(0)
    x = torch.randn(T, H, dtype=dtype, device=device)
    gate_up_proj = torch.randn(E, 2 * intermediate_dim, H, dtype=dtype, device=device) * 0.02
    down_proj = torch.randn(E, H, intermediate_dim, dtype=dtype, device=device) * 0.02
    top_k_index = torch.zeros(T, K, dtype=torch.int32, device=device)
    top_k_weights = torch.ones(T, K, dtype=dtype, device=device) / K

    monkeypatch.setenv("LIGER_FUSED_MOE_CHUNK_TILES", "1")
    out, dx, dW1, dW2, dS = _run_fwd_bwd(x, gate_up_proj, down_proj, top_k_index, top_k_weights)

    assert torch.all(dW1[1:] == 0), "empty experts must have exactly zero dW1"
    assert torch.all(dW2[1:] == 0), "empty experts must have exactly zero dW2"

    x1 = x.clone().requires_grad_(True)
    gup1 = gate_up_proj.clone().requires_grad_(True)
    dn1 = down_proj.clone().requires_grad_(True)
    wts1 = top_k_weights.clone().requires_grad_(True)
    out_ref = _reference_moe_forward(x1, gup1, dn1, top_k_index, wts1)
    out_ref.sum().backward()
    for name, got, want in zip(_GRAD_NAMES, (out, dx, dW1, dW2, dS), (out_ref, x1.grad, gup1.grad, dn1.grad, wts1.grad)):
        torch.testing.assert_close(got, want, atol=3e-3, rtol=1e-2, msg=lambda m, n=name: f"{n}: {m}")


@pytest.mark.parametrize("chunk_tiles", [1, 2])
def test_chunked_backward_bitwise_determinism(chunk_tiles, monkeypatch):
    """Chunked runs must be bitwise identical run-to-run: the chunk loop is
    serial and every cross-chunk accumulation is single-writer fp32 (no float
    atomics). dS is excluded — it keeps the pre-existing fp32 atomics of the
    unchunked path."""
    T, E, H, intermediate_dim, K = 512, 8, 128, 64, 4
    x, gate_up_proj, down_proj, top_k_index, top_k_weights = _make_inputs(
        T, E, H, intermediate_dim, K, torch.bfloat16, device
    )
    monkeypatch.setenv("LIGER_FUSED_MOE_CHUNK_TILES", str(chunk_tiles))
    a = _run_fwd_bwd(x, gate_up_proj, down_proj, top_k_index, top_k_weights)
    b = _run_fwd_bwd(x, gate_up_proj, down_proj, top_k_index, top_k_weights)
    for name, run1, run2 in zip(_GRAD_NAMES[:4], a[:4], b[:4]):
        assert torch.equal(run1, run2), f"{name} is not bitwise deterministic across runs"


def test_chunked_retain_graph_second_backward(monkeypatch):
    """Chunked mode never writes over saved tensors (pre_act is recomputed into
    per-backward staging), so retain_graph re-backward must work and reproduce
    the same gradients bitwise."""
    T, E, H, intermediate_dim, K = 64, 4, 64, 32, 2
    x, gate_up_proj, down_proj, top_k_index, top_k_weights = _make_inputs(
        T, E, H, intermediate_dim, K, torch.float32, device
    )
    monkeypatch.setenv("LIGER_FUSED_MOE_CHUNK_TILES", "2")
    x1 = x.clone().requires_grad_(True)
    out = LigerFusedMoEFunction.apply(x1, gate_up_proj, down_proj, top_k_index, top_k_weights)
    out.sum().backward(retain_graph=True)
    g1 = x1.grad.clone()
    x1.grad = None
    out.sum().backward()
    torch.testing.assert_close(x1.grad, g1)


@pytest.mark.skipif(device != "cuda", reason="uses torch.cuda memory stats")
def test_chunked_backward_memory_cap(monkeypatch):
    """The point of the mode: backward transients drop from TK-proportional to
    window-proportional. At TK=65536 the unchunked backward materializes
    ~(TK×2I + TK×I + TK×H) of transients; chunked staging is a few MiB plus the
    fixed fp32 accumulators."""
    T, E, H, intermediate_dim, K = 8192, 16, 512, 256, 8
    x, gate_up_proj, down_proj, top_k_index, top_k_weights = _make_inputs(
        T, E, H, intermediate_dim, K, torch.bfloat16, device
    )

    def peak_backward_bytes():
        x1 = x.detach().clone().requires_grad_(True)
        gup1 = gate_up_proj.detach().clone().requires_grad_(True)
        dn1 = down_proj.detach().clone().requires_grad_(True)
        wts1 = top_k_weights.detach().clone().requires_grad_(True)
        out = LigerFusedMoEFunction.apply(x1, gup1, dn1, top_k_index, wts1)
        loss = out.sum()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        loss.backward()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() - base

    # Warm up both paths first so autotune/compile scratch doesn't pollute peaks.
    monkeypatch.delenv("LIGER_FUSED_MOE_CHUNK_TILES", raising=False)
    peak_backward_bytes()
    unchunked_peak = peak_backward_bytes()
    monkeypatch.setenv("LIGER_FUSED_MOE_CHUNK_TILES", "8")
    peak_backward_bytes()
    chunked_peak = peak_backward_bytes()

    assert chunked_peak < 0.5 * unchunked_peak, (
        f"chunked backward peak {chunked_peak / 2**20:.1f} MiB not < 50% of "
        f"unchunked {unchunked_peak / 2**20:.1f} MiB"
    )
