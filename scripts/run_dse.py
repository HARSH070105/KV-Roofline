#!/usr/bin/env python3
"""Run the full design-space exploration for one model/hardware pair over the context regimes.

Example:
  python scripts/run_dse.py --model llama3.1-8b --hw h100
  python scripts/run_dse.py --model llama3-70b --hw h100 --chips 8 --theta 0.5
  python scripts/run_dse.py --model llama3.1-8b --hw h100 --quality-csv data/quality_table.csv --gate 1.0

Outputs in results/<model>_<hw>/ :
  <regime>_sweep.csv      every evaluated config (all metrics)
  <regime>_pareto.csv     quality-gated Pareto front (tokens/s up, area down, energy/token down)
  <regime>_*.png          figures
  RECOMMENDATIONS.md      per-regime summary: baseline, best knob per family, iso-area winners
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pandas as pd  # noqa: E402

from kvroofline import get_hw, get_model  # noqa: E402
from kvroofline.compression import Compression  # noqa: E402
from kvroofline.cost import CostParams  # noqa: E402
from kvroofline.dse import (DEFAULT_REGIMES, Grid, Regime, best_by_family, iso_area_best, pareto_front,  # noqa: E402
                            single_knob_curves, sweep)
from kvroofline.engine import Calibration, crossover_S, evaluate, max_batch  # noqa: E402
from kvroofline.plots import plot_heatmap, plot_knob_curves, plot_pareto, plot_roofline  # noqa: E402
from kvroofline.quality import QualityTable  # noqa: E402


def describe(r) -> str:
    parts = []
    if r["sram_extra_mb"] > 0:
        kvf = "KV" if r["sram_kv_frac"] >= 1 else ("weights" if r["sram_kv_frac"] <= 0 else f"{r['sram_kv_frac']:.0%} KV")
        parts.append(f"+{int(r['sram_extra_mb'])} MB SRAM (pins {kvf})")
    if r["hbm_extra_stacks"] > 0:
        parts.append(f"+{int(r['hbm_extra_stacks'])} HBM stack")
    if r["hbm_speed"] > 1:
        parts.append(f"HBM x{r['hbm_speed']:.2f} pin speed")
    if r["comp"] != "KV16":
        parts.append(r["comp"])
    return ", ".join(parts) if parts else "baseline"


def md_table(df: pd.DataFrame, cols: dict) -> str:
    out = ["| " + " | ".join(cols.values()) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        row = []
        for k in cols:
            v = r[k]
            if isinstance(v, float):
                v = "-" if pd.isna(v) else f"{v:.3g}"
            row.append(str(v))
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def parse_regime(s: str, default_chips: int) -> Regime:
    p = s.split(":")
    name, S, B = p[0], int(p[1]), int(p[2])
    return Regime(name, S, B, ""), (int(p[3]) if len(p) > 3 else default_chips)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="llama3.1-8b")
    ap.add_argument("--hw", default="h100")
    ap.add_argument("--chips", type=int, default=1, help="default chip count (tensor parallel)")
    ap.add_argument("--regime", action="append",
                    help="name:S:B[:chips]  (repeatable). Default: A:8k:8, B:128k:8, C:512k:8 (C uses 2 chips for 8B)")
    ap.add_argument("--batch-mode", choices=["max", "fixed"], default="max",
                    help="max: largest feasible batch up to B (compression frees capacity); fixed: exactly B")
    ap.add_argument("--gate", type=float, default=1.0, help="quality gate, %% degradation vs BF16")
    ap.add_argument("--quality-csv", default=None, help="measured quality table (see kvroofline/quality.py)")
    ap.add_argument("--theta", type=float, default=1.0, help="KV access skew (1 uniform, 0.5 skewed)")
    ap.add_argument("--rho", type=float, default=0.0, help="KV shared across batch (prefix reuse)")
    ap.add_argument("--dequant-flops", type=float, default=6.0)
    ap.add_argument("--attn-unit", choices=["tensor", "cuda"], default="tensor")
    ap.add_argument("--hbm-speeds", default="1.0", help="comma list of HBM pin-speed scales, e.g. 1.0,1.25 (zero area cost in model: use with care)")
    ap.add_argument("--calib", default=None, help="calibration json from scripts/validate.py --fit")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    model = get_model(a.model)
    out = a.out or os.path.join("results", f"{a.model}_{a.hw}")
    os.makedirs(out, exist_ok=True)
    qt = QualityTable.from_csv(a.quality_csv) if a.quality_csv else QualityTable.placeholder()
    calib = Calibration()
    if a.calib:
        import json
        calib = Calibration(**json.load(open(a.calib)))

    if a.regime:
        regs = [parse_regime(s, a.chips) for s in a.regime]
    else:
        regs = [(r, a.chips) for r in DEFAULT_REGIMES]
        if a.chips == 1 and model.name == "llama3.1-8b":
            regs[2] = (regs[2][0], 2)          # 512k x 64 GB KV needs 2 chips for BF16 baseline
        if a.chips == 1 and model.name in ("llama3-70b", "deepseek-v2"):
            regs = [(r, 8) for r, _ in regs]

    md = [f"# DSE results: {model.name} on {a.hw}\n",
          f"Quality source: **{'measured table ' + a.quality_csv if a.quality_csv else 'PLACEHOLDER heuristic (not measured)'}**; "
          f"gate = {a.gate}%; theta = {a.theta}; rho = {a.rho}; batch mode = {a.batch_mode}; "
          f"dequant FLOPs/elem = {a.dequant_flops}; attention unit = {a.attn_unit}.\n"]
    if not a.quality_csv:
        md.append("> WARNING: quality numbers are assumed placeholders. Rankings that depend on the gate are NOT reportable "
                  "until `data/quality_table.csv` is filled with measured values (week 5).\n")

    for reg, chips in regs:
        base_hw = get_hw(a.hw, n_chips=chips)
        print(f"\n=== {reg.name}: S={reg.S:,} B<={reg.B} chips={chips} ===")
        try:
            df = sweep(model, base_hw, reg, Grid(hbm_speed=[float(x) for x in a.hbm_speeds.split(',')]), qt, a.gate, a.batch_mode, a.theta, a.rho, a.dequant_flops,
                       a.attn_unit, CostParams(), calib)
        except RuntimeError as e:
            print("  SKIPPED:", e)
            md.append(f"## {reg.name}\nSkipped: {e}\n")
            continue
        df["config"] = df.apply(describe, axis=1)
        df.to_csv(os.path.join(out, f"{reg.name}_sweep.csv"), index=False)
        front = pareto_front(df)
        front.to_csv(os.path.join(out, f"{reg.name}_pareto.csv"), index=False)
        curves = single_knob_curves(df)

        plot_pareto(df, front, os.path.join(out, f"{reg.name}_pareto.png"),
                    f"{model.name} {reg.name} (S={reg.S:,}): gated Pareto")
        plot_knob_curves(curves, os.path.join(out, f"{reg.name}_knobs.png"),
                         f"{model.name} {reg.name}: single-knob speedup vs added area")
        plot_heatmap(df, os.path.join(out, f"{reg.name}_sram_hbm_heatmap.png"),
                     f"{reg.name}: speedup from SRAM x HBM stacks (BF16 KV)")
        plot_roofline(model, base_hw, reg.S, int(df["base_B"].iloc[0]),
                      [Compression(), Compression(kv_bits=4, group=64, residual=128)],
                      os.path.join(out, f"{reg.name}_roofline.png"), a.attn_unit)

        b = df[(df["sram_extra_mb"] == 0) & (df["hbm_extra_stacks"] == 0) & (df["hbm_speed"] == 1.0) &
               (df["comp"] == "KV16") & (df["sram_kv_frac"] == 1.0)].iloc[0]
        md.append(f"## {reg.name}  (S = {reg.S:,}, chips = {chips})\n")
        md.append(f"**Baseline** (BF16 KV, no extras): B = {int(b['B'])} (max feasible {int(b['max_batch'])}), "
                  f"TPOT = {b['tpot_ms']:.2f} ms, {b['tokens_per_s']:.0f} tok/s, dominant = `{b['dominant']}`, "
                  f"AI = {b['ai_step']:.1f} FLOP/B (ridge {b['ridge_tensor']:.0f}), KV share of HBM bytes = {b['kv_share_of_hbm']:.0%}, "
                  f"S* = {b['s_star']:,.0f} tokens.\n")

        fam = best_by_family(df)
        md.append("### Best gated config per knob family\n")
        md.append(md_table(fam, {"family": "family", "config": "config", "B": "B", "speedup": "speedup",
                                 "area_added_mm2": "+mm2", "d_energy_mj": "dE mJ/tok", "d_ai": "dAI",
                                 "d_ai_over_ridge": "d(AI/ridge)", "quality_delta_pct": "quality %",
                                 "dominant": "bound"}) + "\n")

        md.append("### Iso-area winners (best tokens/s under an added-area budget, quality-gated)\n")
        iso = iso_area_best(df)
        if len(iso):
            md.append(md_table(iso, {"area_budget_mm2": "budget mm2", "config": "config", "B": "B",
                                     "speedup": "speedup", "area_added_mm2": "used mm2", "energy_mj_per_token": "mJ/token",
                                     "quality_delta_pct": "quality %", "quality_measured": "measured?"}) + "\n")

        for fam_name, cur in curves.items():
            if fam_name in ("SRAM", "HBM") and len(cur):
                g = cur[cur["passes_gate"] & (cur["area_added_mm2"] > 0)]
                if len(g):
                    r = g.loc[g["speedup_gain_per_mm2"].idxmax()]
                    md.append(f"* Best single-knob **{fam_name}** efficiency: {describe(r)} -> speedup {r['speedup']:.3f}x for "
                              f"{r['area_added_mm2']:.0f} mm2 ({r['speedup_gain_per_mm2']:.2e} speedup/mm2; "
                              f"dAI/mm2 = {r['d_ai_per_mm2']:.2e}, d(AI/ridge)/mm2 = {r['d_ai_ridge_per_mm2']:.2e}).")
        md.append("")

        print(f"  baseline: B={int(b['B'])} {b['tokens_per_s']:.0f} tok/s dominant={b['dominant']} KVshare={b['kv_share_of_hbm']:.0%}")
        print(best_by_family(df)[["family", "config", "B", "speedup", "area_added_mm2", "quality_delta_pct"]]
              .head(6).to_string(index=False))

    with open(os.path.join(out, "RECOMMENDATIONS.md"), "w") as f:
        f.write("\n".join(md))
    print(f"\nWrote {out}/ (CSV, PNG, RECOMMENDATIONS.md)")


if __name__ == "__main__":
    main()
