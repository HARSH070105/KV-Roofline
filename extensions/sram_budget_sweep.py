#!/usr/bin/env python3
"""
Workstream 2 follow-up: how small must the eviction budget be for pinned SRAM
to matter?  Sweeps eviction budget x pinned SRAM size, with BF16 and INT4 g64 KV.

Run from the repo root:   python extensions/sram_budget_sweep.py
Outputs -> extensions/outputs/: ws2_sram_budget_sweep.csv, .png, .pdf
Read-only use of kvroofline.

NOTE: quality at small budgets (<= 4k) is NOT in the quality table; treat tiny
budgets as speed-only what-ifs, not as gate-passing designs.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kvroofline import Compression, evaluate, get_hw, get_model, max_batch  # noqa: E402

OUT = Path(__file__).resolve().parent

MODEL, HW, S, CAP = "llama3.1-8b", "h100", 131072, 8
BUDGETS = [512, 1024, 2048, 4096, 8192, 16384]
SRAMS = [0, 128, 256, 512, 1024, 2048, 4096]
KV = {"BF16 KV": dict(kv_bits=16, group=0, residual=0, dequant_flops=0),
      "INT4 g64 KV": dict(kv_bits=4, group=64, residual=128, dequant_flops=6)}


def mb(m, hw, comp):
    try:
        return int(max_batch(m, hw, S=S, comp=comp))
    except TypeError:
        return int(max_batch(m, hw, S, comp))


def main():
    m = get_model(MODEL)
    hw0 = get_hw(HW)
    base = Compression(kv_bits=16, group=0, residual=0, dequant_flops=0)
    b_base = max(1, min(CAP, mb(m, hw0, base)))
    t_base = evaluate(m, hw0, S=S, B=b_base, comp=base)["tokens_per_s"]
    rows = []
    for kname, kv in KV.items():
        for bud in BUDGETS:
            comp = Compression(evict_policy="streaming", evict_budget=bud, **kv)
            b_eff = max(1, min(CAP, mb(m, hw0, comp)))
            for sram in SRAMS:
                hw = hw0.with_(sram_pin_mb=sram)
                for label, B in (("B=1", 1), ("B=B_eff", b_eff)):
                    r = evaluate(m, hw, S=S, B=B, comp=comp)
                    rows.append(dict(kv=kname, budget=bud, sram_mb=sram, batch_mode=label,
                                     B=B, h_kv=r["h_kv"], tokens_per_s=r["tokens_per_s"],
                                     speedup_vs_bf16_base=r["tokens_per_s"] / t_base))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "ws2_sram_budget_sweep.csv", index=False)
    print("saved ws2_sram_budget_sweep.csv")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for i, kname in enumerate(KV):
        for j, (col, title) in enumerate((("h_kv", "on-chip hit fraction h_kv"),
                                          ("speedup_vs_bf16_base", "speedup vs BF16 baseline"))):
            ax = axes[i, j]
            g = df[(df.kv == kname) & (df.batch_mode == "B=B_eff")]
            for sram in SRAMS:
                s = g[g.sram_mb == sram].sort_values("budget")
                ax.plot(s.budget, s[col], "o-", label=f"{sram} MB")
            ax.set_xscale("log", base=2)
            ax.set_xlabel("eviction budget (tokens)")
            ax.set_ylabel(title)
            ax.set_title(f"{kname}, {MODEL}, S={S}, {HW.upper()}, B=B_eff")
            if i == 0 and j == 0:
                ax.legend(title="pinned SRAM", fontsize=6)
    fig.tight_layout()
    fig.savefig(OUT / "ws2_sram_budget_sweep.png", bbox_inches="tight")
    print("saved ws2_sram_budget_sweep.png")

    # smallest budget (largest working set) at which h_kv >= 0.9 for each SRAM size
    g = df[df.batch_mode == "B=B_eff"]
    fits = (g[g.h_kv >= 0.9].groupby(["kv", "sram_mb"]).budget.max()
            .unstack(0))
    print("\nlargest budget (tokens) that reaches h_kv >= 0.9, by SRAM size:")
    print(fits.to_string())


if __name__ == "__main__":
    main()