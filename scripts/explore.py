#!/usr/bin/env python3
"""Four small experiments that test the central claims of the project (deck slides 3-5).
Run this FIRST: it is the fastest way to build intuition for the tool.

  E1  Which model is KV-bound where?   (KV share of HBM bytes vs context; crossover S*)
  E2  CUDA-core-roof hypothesis        (does INT4 attention hit the 20 FLOP/B CUDA roof before the memory roof?)
  E3  SRAM hit-rate sensitivity        (does a bigger on-chip cache help, and what does it depend on?)
  E4  Decode roofline picture          (figure)

  python scripts/explore.py [--hw h100]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pandas as pd  # noqa: E402

from kvroofline import MODELS, Compression, SramPolicy, evaluate, get_hw, get_model  # noqa: E402
from kvroofline.plots import plot_crossover, plot_roofline  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda x: f"{x:,.3g}")


def e1(hw, out):
    print("\n=== E1: KV share of HBM traffic at 128k (B=4) and crossover S* ===")
    rows = []
    for m in MODELS.values():
        h = hw.with_(n_chips=8 if m.params_total > 5e10 else 1)
        r = evaluate(m, h, 131072, 4)
        rows.append({"model": m.name, "attn": m.attn, "KV KiB/token": m.kv_bytes_per_token() / 1024,
                     "AI_attn(BF16)": m.attn_ai(), "KV share @128k": r["kv_share_of_hbm"],
                     "S*(B=4) tokens": r["s_star"], "dominant": r["dominant"], "tok/s": r["tokens_per_s"]})
    print(pd.DataFrame(rows).to_string(index=False))
    plot_crossover([get_model(n) for n in ["llama2-7b", "llama3.1-8b", "qwen2.5-7b", "deepseek-v2-lite"]],
                   hw, 4, os.path.join(out, "E1_crossover.png"))


def e2(hw, out):
    print("\n=== E2: Does INT4 KV hit the CUDA-core roof? (Llama-3.1-8B, S=128k, B=3) ===")
    print("Ideal byte reduction BF16 -> INT4+metadata(g64) = %.2fx; compare to achieved speedup.\n"
          % (2.0 / Compression(kv_bits=4, group=64).bytes_per_elem_quant()))
    m = get_model("llama3.1-8b")
    base = evaluate(m, hw, 131072, 3, Compression())
    rows = []
    for unit in ("tensor", "cuda"):
        for c in (2, 4, 6, 8, 12, 16):
            comp = Compression(kv_bits=4, group=64, residual=128, dequant_flops=c)
            r = evaluate(m, hw, 131072, 3, comp, attn_unit=unit)
            rows.append({"QK/PV on": unit, "dequant FLOP/elem": c, "attn AI": r["ai_attn"],
                         "CUDA-time/HBM-time": r["cuda_time_over_hbm_time"], "attn bound": r["bind_attn"],
                         "speedup vs BF16": base["tpot_ms"] / r["tpot_ms"]})
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nReading: if 'attn bound' flips from hbm to cuda, the CUDA-core roof explains sub-linear quantisation speedups.")


def e3(hw, out):
    print("\n=== E3: SRAM hit-rate sensitivity (Llama-3.1-8B, S=128k, B=3, KV16) ===")
    m = get_model("llama3.1-8b")
    base = evaluate(m, hw, 131072, 3)
    rows = []
    for label, comp in (("full KV", Compression()), ("Quest 4096", Compression(evict_policy="quest", evict_budget=4096)),
                        ("Streaming 4096", Compression(evict_policy="streaming", evict_budget=4096))):
        for theta in (1.0, 0.5):
            for extra in (0, 128, 512, 2048):
                h = hw.with_(sram_pin_mb=hw.sram_pin_mb + extra)
                r = evaluate(m, h, 131072, 3, comp, SramPolicy(1.0, theta, 0.0))
                rows.append({"KV policy": label, "theta": theta, "SRAM MB": h.sram_pin_mb, "h_kv": r["h_kv"],
                             "tok/s": r["tokens_per_s"], "speedup vs full-KV no SRAM": r["tokens_per_s"] / base["tokens_per_s"]})
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nReading: with the full 128k KV (~50 GB working set) even 2 GB of SRAM covers ~4% of reads, so SRAM alone "
          "barely helps. SRAM pays off after eviction shrinks the working set, or if access is skewed (theta<1).")


def e4(hw, out):
    m = get_model("llama3.1-8b")
    plot_roofline(m, hw, 131072, 3, [Compression(), Compression(kv_bits=8, group=0),
                                     Compression(kv_bits=4, group=64, residual=128),
                                     Compression(kv_bits=2, group=32, residual=128)],
                  os.path.join(out, "E4_roofline.png"))
    print(f"\nWrote figures to {out}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hw", default="h100")
    ap.add_argument("--out", default="results/explore")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    hw = get_hw(a.hw)
    e1(hw, a.out)
    e2(hw, a.out)
    e3(hw, a.out)
    e4(hw, a.out)
