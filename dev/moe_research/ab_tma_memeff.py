"""Two cheap A/Bs on B300 for v15:
1. TMA weight loads on vs off (E6 gave +6-9% on H100; B300 L2 is 2.6x bigger).
2. LIGER_FUSED_MOE_MEMORY_EFFICIENT=1 cost (was -12-18% bwd on H100).

Run:  python ab_tma_memeff.py tma
      LIGER_FUSED_MOE_MEMORY_EFFICIENT=1 python ab_tma_memeff.py memeff
"""

import sys

sys.path.insert(0, ".")

import torch

from common import bench_memory, bench_variant, load_variant

QW = dict(E=128, H=2048, I=768, K=8)
mode = sys.argv[1] if len(sys.argv) > 1 else "tma"

if mode == "tma":
    # Two separately-named module instances of the SAME variant → independent
    # autotune caches, so each arm tunes for its own USE_TMA constexpr value.
    v_on = load_variant("variants/v16_bwtune.py", name="tma_on")
    v_off = load_variant("variants/v16_bwtune.py", name="tma_off")
    v_off._tma_eligibility = lambda *a, **k: (False, False)
    for T in [1024, 8192, 32768]:
        for m in ["forward", "full", "infer"]:
            a = bench_variant(v_on, T, **QW, mode=m)
            b = bench_variant(v_off, T, **QW, mode=m)
            print(f"T={T:>6} {m:>8}: TMA={a:7.3f}  noTMA={b:7.3f}  (TMA speedup {b / a:4.2f}x)")
else:
    import os

    assert os.environ.get("LIGER_FUSED_MOE_MEMORY_EFFICIENT") == "1"
    # "backward" mode is invalid here: it re-backwards with retain_graph=True,
    # which the in-place alias intentionally rejects. Use "full" (fresh graph).
    v = load_variant("variants/v15_fixdx.py", name="memeff")
    for T in [8192, 32768]:
        b = bench_variant(v, T, **QW, mode="full")
        mb = bench_memory(v, T, **QW, mode="full")
        print(f"T={T:>6} full: memeff={b:7.3f} ms   peak-mem={mb:7.0f} MB")
