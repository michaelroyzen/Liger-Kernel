"""Head-to-head at Qwen3.5-35B-A3B lm_head shapes on B300:

  A. TRL-built-in-style logps path: bf16 logits kept in the autograd graph +
     trl.trainer.utils.selective_log_softmax, GRPO-DAPO loss math on top.
     (What their training runs today with use_liger_loss unset.)
  B. Liger chunked GRPO loss, post-E30 fix (bf16 tensor-core backward GEMMs).

Same inputs, same loss family (dapo, sequence-level IS, beta=0, temp=1, eps 0.2),
same measurement (CUDA events, 3 iters, peak memory above inputs).
"""

import sys

import torch

from trl.trainer.utils import selective_log_softmax

from liger_kernel.chunked_loss.grpo_loss import LigerFusedLinearGRPOLoss

H, V = 2048, 248320
DEV = "cuda"


def make_case(B, T, seed=7):
    g = torch.Generator(device=DEV).manual_seed(seed)
    hidden = torch.randn(B, T, H, device=DEV, dtype=torch.bfloat16, generator=g) * 0.5
    hidden.requires_grad_(True)
    weight = (torch.randn(V, H, device=DEV, dtype=torch.bfloat16, generator=g) * 0.02).requires_grad_(True)
    ids = torch.randint(0, V, (B, T), device=DEV, generator=g)
    mask = torch.ones(B, T, device=DEV, dtype=torch.long)
    adv = torch.randn(B, device=DEV, generator=g)
    old_logps = (-torch.rand(B, T, device=DEV, generator=g) * 3).to(torch.float32)
    return hidden, weight, ids, mask, adv, old_logps


def grpo_dapo_loss_from_logps(logps, old_logps, mask, adv, eps=0.2):
    """Sequence-level IS + clip, dapo (global token-mean) normalization —
    mirrors the math both TRL and Liger apply after logps."""
    seq_logp = (logps * mask).sum(-1)
    seq_old = (old_logps * mask).sum(-1)
    log_ratio = (seq_logp - seq_old) / mask.sum(-1).clamp(min=1)
    ratio = torch.exp(log_ratio).unsqueeze(-1)  # (B, 1) broadcast over tokens
    a = adv.unsqueeze(-1)
    unclipped = ratio * a
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * a
    per_token = -torch.minimum(unclipped, clipped) * mask
    return per_token.sum() / mask.sum().clamp(min=1)


def run_trl_style(case):
    hidden, weight, ids, mask, adv, old = case
    hidden.grad = None
    weight.grad = None
    logits = hidden @ weight.t()  # (B, T, V) bf16, kept in graph
    logps = selective_log_softmax(logits, ids)  # TRL's util (fp32 math inside)
    loss = grpo_dapo_loss_from_logps(logps, old, mask, adv)
    loss.backward()
    return loss


def run_liger(case, loss_fn):
    hidden, weight, ids, mask, adv, old = case
    hidden.grad = None
    weight.grad = None
    out = loss_fn(hidden, weight, ids, mask, adv, old_per_token_logps=old)
    loss = out[0] if isinstance(out, tuple) else out
    loss.backward()
    return loss


def bench(fn, iters=3):
    fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.max_memory_allocated()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters, (torch.cuda.max_memory_allocated() - base) / 1024**2


def main():
    print(f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}, H={H} V={V}")
    for B, T in [(4, 4096), (4, 16384)]:
        case = make_case(B, T)
        loss_fn = LigerFusedLinearGRPOLoss(
            beta=0.0, use_ref_model=False, chunk_size=1, compiled=False,
            temperature=1.0, epsilon_low=0.2, epsilon_high=0.2,
            loss_type="dapo", importance_sampling_level="sequence",
        )
        try:
            ms_t, mem_t = bench(lambda: run_trl_style(case))
            trl_row = f"TRL-style: {ms_t:8.1f} ms  peak +{mem_t:9.0f} MB"
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            trl_row = "TRL-style: OOM"
        ms_l, mem_l = bench(lambda: run_liger(case, loss_fn))
        print(f"B={B} T={T:>6} ({B * T:>6} tokens):  {trl_row}   ||   "
              f"liger(fixed): {ms_l:8.1f} ms  peak +{mem_l:7.0f} MB")
        del case, loss_fn
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
