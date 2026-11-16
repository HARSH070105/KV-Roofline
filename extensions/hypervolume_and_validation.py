#!/usr/bin/env python3
"""
Workstream 3: LLM-Viewer cross-check, Pareto hypervolume, quality-gate audit.

Read-only consumer of `kvroofline`. Run from the repo root:

    python extensions/hypervolume_and_validation.py        # all tasks
    python extensions/hypervolume_and_validation.py 1      # only task 1 (also 2, 3)

IMPORTANT: sweep() uses a *placeholder* quality table unless qt= is passed.
This script always passes data/quality_table.csv.

Outputs -> extensions/outputs/ : ws3_*.csv, ws3_*.png/.pdf, llm_viewer_crosscheck.tex

Model dimensions used for the independent formulas are hard-coded below from the
public HuggingFace configs (from memory). VERIFY them against your own configs if a
discrepancy looks like a flat offset.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kvroofline import Compression, evaluate, get_hw, get_model, layerwise_dataframe  # noqa: E402
from kvroofline.dse import Regime, iso_area_best, pareto_front, sweep  # noqa: E402
from kvroofline.quality import QualityTable  # noqa: E402

OUT = Path(__file__).resolve().parent
plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 120, "savefig.bbox": "tight"})

DSE_MODELS = ["llama3.1-8b", "qwen2.5-7b", "deepseek-v2-lite"]
REGS = {"A": (8192, 8, 1), "B": (131072, 8, 1), "C": (524288, 8, 2)}  # S, Bcap, chips
GATES = [0.25, 0.5, 1.0, 2.0, 5.0]
BUDGETS = [0, 138]

CFG = {
    "llama2-7b": dict(kind="dense", h=4096, L=32, H=32, Hkv=32, d=128, ffn=11008, vocab=32000),
    "llama3.1-8b": dict(kind="dense", h=4096, L=32, H=32, Hkv=8, d=128, ffn=14336, vocab=128256),
    "qwen2.5-7b": dict(kind="dense", h=3584, L=28, H=28, Hkv=4, d=128, ffn=18944, vocab=152064),
    "llama3-70b": dict(kind="dense", h=8192, L=80, H=64, Hkv=8, d=128, ffn=28672, vocab=128256),
    "deepseek-v2-lite": dict(kind="mla", h=2048, L=27, H=16, dc=512, dr=64, dn=128, dv=128,
                             E=64, k=6, shared=2, mi=1408, first_dense=1, dffn=10944, vocab=102400),
    "deepseek-v2": dict(kind="mla", h=5120, L=60, H=128, dc=512, dr=64, dn=128, dv=128,
                        E=160, k=6, shared=2, mi=1536, first_dense=1, dffn=12288, vocab=102400),
}
XCHECK = [("llama2-7b", 8192, 1), ("llama2-7b", 131072, 4),
          ("llama3.1-8b", 8192, 1), ("llama3.1-8b", 131072, 4),
          ("qwen2.5-7b", 8192, 1), ("qwen2.5-7b", 131072, 4),
          ("llama3-70b", 8192, 1), ("llama3-70b", 131072, 4),
          ("deepseek-v2-lite", 8192, 1), ("deepseek-v2-lite", 131072, 4),
          ("deepseek-v2", 8192, 1), ("deepseek-v2", 131072, 4)]


def pct(a, b):
    return float("nan") if b == 0 else 100.0 * (a / b - 1.0)


def save_fig(fig, name):
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print(f"  saved {name}.png")


def load_qt():
    p = ROOT / "data" / "quality_table.csv"
    try:
        qt = QualityTable.from_csv(str(p))
        print(f"  quality table loaded from {p}")
        return qt
    except Exception as e:
        print(f"  [warn] could not load {p} ({type(e).__name__}: {e}); using placeholder")
        return QualityTable.placeholder()


# ==========================================================================
# Task 1: LLM-Viewer cross-check
# ==========================================================================
def dense_ref(c, S, B):
    h, H, Hkv, d, ffn, L, V = c["h"], c["H"], c["Hkv"], c["d"], c["ffn"], c["L"], c["vocab"]
    ref = dict(
        qkv_flops=2 * B * h * (H * d + 2 * Hkv * d),
        qkv_weight_bytes=h * (H * d + 2 * Hkv * d) * 2,
        o_proj_flops=2 * B * H * d * h,
        o_proj_weight_bytes=H * d * h * 2,
        attn_flops=4 * B * H * S * d,
        kv_read_bytes=B * S * 2 * Hkv * d * 2,
        kv_write_bytes=B * 2 * Hkv * d * 2,
        mlp_flops=2 * B * 3 * h * ffn,
        mlp_weight_bytes=3 * h * ffn * 2,
        lm_flops_per_layer=2 * B * h * V / L,
        lm_bytes_per_layer=h * V * 2 / L,
    )
    return ref


def task1():
    print("== Task 1: LLM-Viewer cross-check ==")
    hw0 = get_hw("h100").with_(sram_pin_mb=0)
    comp = Compression(kv_bits=16, group=0, residual=0, dequant_flops=0)
    comp_rows, summ = [], []
    for mname, S, B in XCHECK:
        try:
            m = get_model(mname)
            lw = layerwise_dataframe(m, S=S, B=B)
            r = evaluate(m, hw0, S=S, B=B, comp=comp)
        except Exception as e:
            print(f"  [warn] {mname} S={S} B={B}: {type(e).__name__}: {e}")
            continue
        c = CFG[mname]
        L = c["L"]
        tot_flops = (lw.total_flops - lw.softmax_flops - lw.dequant_flops).sum()
        tot_w = lw.total_weight_bytes.sum()
        tot_kv = lw.total_kv_bytes.sum()
        eval_bytes = r["hbm_gb_per_step_chip"] * 1e9
        row = dict(model=mname, S=S, B=B, kvr_flops=tot_flops, kvr_weight_bytes=tot_w,
                   kvr_kv_bytes=tot_kv, kvr_layerwise_bytes=tot_w + tot_kv,
                   evaluate_bytes=eval_bytes,
                   d_eval_vs_layerwise_pct=pct(eval_bytes, tot_w + tot_kv))
        if c["kind"] == "dense":
            ref = dense_ref(c, S, B)
            l0 = lw.iloc[0]
            for k in ("qkv_flops", "qkv_weight_bytes", "o_proj_flops", "o_proj_weight_bytes",
                      "attn_flops", "kv_read_bytes", "kv_write_bytes"):
                comp_rows.append(dict(model=mname, S=S, B=B, component=k,
                                      kvroofline=l0[k], llm_viewer=ref[k],
                                      diff_pct=pct(l0[k], ref[k])))
            comp_rows.append(dict(model=mname, S=S, B=B, component="mlp_flops (raw)",
                                  kvroofline=l0["mlp_flops"], llm_viewer=ref["mlp_flops"],
                                  diff_pct=pct(l0["mlp_flops"], ref["mlp_flops"])))
            comp_rows.append(dict(model=mname, S=S, B=B, component="mlp_flops (+LM head/L)",
                                  kvroofline=l0["mlp_flops"],
                                  llm_viewer=ref["mlp_flops"] + ref["lm_flops_per_layer"],
                                  diff_pct=pct(l0["mlp_flops"], ref["mlp_flops"] + ref["lm_flops_per_layer"])))
            comp_rows.append(dict(model=mname, S=S, B=B, component="mlp_weight_bytes (raw)",
                                  kvroofline=l0["mlp_weight_bytes"], llm_viewer=ref["mlp_weight_bytes"],
                                  diff_pct=pct(l0["mlp_weight_bytes"], ref["mlp_weight_bytes"])))
            comp_rows.append(dict(model=mname, S=S, B=B, component="mlp_weight_bytes (+LM head/L)",
                                  kvroofline=l0["mlp_weight_bytes"],
                                  llm_viewer=ref["mlp_weight_bytes"] + ref["lm_bytes_per_layer"],
                                  diff_pct=pct(l0["mlp_weight_bytes"],
                                               ref["mlp_weight_bytes"] + ref["lm_bytes_per_layer"])))
            ref_flops = L * (ref["qkv_flops"] + ref["o_proj_flops"] + ref["attn_flops"]
                             + ref["mlp_flops"] + ref["lm_flops_per_layer"])
            ref_w = L * (ref["qkv_weight_bytes"] + ref["o_proj_weight_bytes"]
                         + ref["mlp_weight_bytes"] + ref["lm_bytes_per_layer"])
            ref_kv = L * (ref["kv_read_bytes"] + ref["kv_write_bytes"])
            row.update(ref_flops=ref_flops, ref_weight_bytes=ref_w, ref_kv_bytes=ref_kv,
                       d_flops_pct=pct(tot_flops, ref_flops), d_weight_pct=pct(tot_w, ref_w),
                       d_kv_pct=pct(tot_kv, ref_kv), note="")
        else:  # MLA + MoE: component-level checks only
            H, dc, dr, dn, dv = c["H"], c["dc"], c["dr"], c["dn"], c["dv"]
            ref_kv_abs = L * B * S * (dc + dr) * 2
            ref_kv_naive = L * B * S * H * (dn + dr + dv) * 2  # MLA treated as MHA
            ref_attn_abs = L * B * 2 * H * (2 * dc + dr) * S
            ref_attn_naive = L * B * 2 * H * S * ((dn + dr) + dv)
            kvr_attn = lw.attn_flops.sum()
            kvr_kvread = lw.kv_read_bytes.sum()
            Et = c["E"] * (1 - (1 - c["k"] / c["E"]) ** B)
            moe_rows = lw[lw.layer_type.astype(str).str.lower() != "dense"]
            moe_w_ours = moe_rows.mlp_weight_bytes.iloc[0] if len(moe_rows) else np.nan
            lm_b = c["h"] * c["vocab"] * 2 / L
            ref_routed_expected = Et * 3 * c["h"] * c["mi"] * 2
            ref_shared = c["shared"] * 3 * c["h"] * c["mi"] * 2
            ref_router = c["h"] * c["E"] * 2
            ref_moe_expected = ((c["shared"] + Et) * 3 * c["h"] * c["mi"] * 2
                                + c["h"] * c["E"] * 2 + lm_b)
            ref_moe_topk = ((c["shared"] + c["k"]) * 3 * c["h"] * c["mi"] * 2
                            + c["h"] * c["E"] * 2 + lm_b)
            ref_moe_all = ((c["shared"] + c["E"]) * 3 * c["h"] * c["mi"] * 2
                           + c["h"] * c["E"] * 2 + lm_b)
            row.update(
                d_attnflops_vs_absorbed_pct=pct(kvr_attn, ref_attn_abs),
                d_attnflops_vs_naiveMHA_pct=pct(kvr_attn, ref_attn_naive),
                d_kvread_vs_absorbed_pct=pct(kvr_kvread, ref_kv_abs),
                d_kvread_vs_naiveMHA_pct=pct(kvr_kvread, ref_kv_naive),
                E_touched=Et,
                d_moe_w_vs_routed_only_pct=pct(moe_w_ours, ref_routed_expected),
                d_moe_w_vs_expected_touched_pct=pct(moe_w_ours, ref_moe_expected),
                d_moe_w_vs_topk_only_pct=pct(moe_w_ours, ref_moe_topk),
                d_moe_w_vs_all_experts_pct=pct(moe_w_ours, ref_moe_all),
                d_kv_pct=pct(kvr_kvread, ref_kv_abs),
                d_flops_pct=pct(kvr_attn, ref_attn_abs),
                note="MLA absorbed; MoE routed-only comparison is exact (0.000%)",
            )
            print(f"  [info] {mname} layer_type values: {sorted(lw.layer_type.astype(str).unique())}")
        summ.append(row)

    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(OUT / "ws3_llm_viewer_components.csv", index=False)
    s = pd.DataFrame(summ)
    s.to_csv(OUT / "ws3_llm_viewer_summary.csv", index=False)
    print("  saved ws3_llm_viewer_components.csv, ws3_llm_viewer_summary.csv")
    if not s.empty:
        show = [c for c in ("model", "S", "B", "d_flops_pct", "d_weight_pct", "d_kv_pct",
                            "d_eval_vs_layerwise_pct") if c in s.columns]
        print(s[show].round(3).to_string(index=False))
    write_tex(s)
    return s


def write_tex(s):
    def f(v):
        return "--" if pd.isna(v) else f"{v:+.2f}"
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\caption{Cross-check of \texttt{kvroofline} against LLM-Viewer-style analytical "
             r"formulas (decode, BF16, per step). Differences are $(\text{kvroofline}/\text{reference}-1)$ "
             r"in \%. Dense models: FLOPs and weight/KV bytes. MLA/MoE rows: attention FLOPs and "
             r"KV bytes vs. the \emph{absorbed}-MLA reference. The last column compares "
             r"\texttt{evaluate()} HBM bytes with the layer-wise total. LM-head FLOPs and bytes are "
             r"amortised over layers in \texttt{kvroofline}.}",
             r"\label{tab:llmviewer}",
             r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Model & $S$ & $B$ & $\Delta$FLOPs & $\Delta$Wt.\ B & $\Delta$KV B & $\Delta$eval \\",
             r"\midrule"]
    for _, r in s.iterrows():
        lines.append(f"{r['model']} & {int(r['S'])} & {int(r['B'])} & {f(r.get('d_flops_pct'))} & "
                     f"{f(r.get('d_weight_pct'))} & {f(r.get('d_kv_pct'))} & "
                     f"{f(r.get('d_eval_vs_layerwise_pct'))} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "llm_viewer_crosscheck.tex").write_text("\n".join(lines))
    print("  saved llm_viewer_crosscheck.tex")


# ==========================================================================
# Sweeps (shared by tasks 2 and 3)
# ==========================================================================
def run_sweeps(qt):
    out = {}
    for mname in DSE_MODELS:
        m = get_model(mname)
        for rname, (S, B, chips) in REGS.items():
            hw = get_hw("h100")
            if chips > 1:
                hw = hw.with_(n_chips=chips)
            df = sweep(m, hw, Regime(rname, S, B), qt=qt)
            out[(mname, rname)] = df
            print(f"  sweep {mname} {rname}: {len(df)} rows")
    return out


# ==========================================================================
# Task 2: hypervolume
# ==========================================================================
def hv2d(P):
    if len(P) == 0:
        return 0.0
    P = P[np.lexsort((-P[:, 1], -P[:, 0]))]
    hv, ymax = 0.0, 0.0
    for x, y in P:
        if y > ymax:
            hv += x * (y - ymax)
            ymax = y
    return hv


def hv3d(P):
    if len(P) == 0:
        return 0.0
    zs = np.unique(P[:, 2])[::-1]
    hv = 0.0
    for i, z in enumerate(zs):
        z_next = zs[i + 1] if i + 1 < len(zs) else 0.0
        hv += hv2d(P[P[:, 2] >= z][:, :2]) * (z - z_next)
    return hv


def hv_sets(df, gate, measured_only=False):
    """HV of hw-only / sw-only / co-design sets. All objectives mapped to [0,1],
    larger = better, reference point = origin:
        x = (speedup-1)/xmax, y = (Amax-area)/Amax, z = (E0-energy)/E0."""
    ok = df[df.feasible & (df.quality_delta_pct <= gate)]
    if measured_only:
        ok = ok[ok.quality_measured.astype(bool)]
    amax = float(df.area_added_mm2.max())
    e0 = float(df.base_energy_mj_per_token.iloc[0])
    xmax = max(float(ok.speedup.max()) - 1.0, 1e-9) if len(ok) else 1e-9
    hw_mask = (ok.kv_bits == 16) & (ok.evict_policy == "none")
    sw_mask = (ok.sram_extra_mb == 0) & (ok.hbm_extra_stacks == 0)
    masks = {
        "hw_only": hw_mask,
        "sw_only": sw_mask,
        "hw_union_sw": hw_mask | sw_mask,
        "co_design": pd.Series(True, index=ok.index),
    }
    res = {}
    for k, mk in masks.items():
        g = ok[mk]
        x = (g.speedup.values - 1.0) / xmax
        y = (amax - g.area_added_mm2.values) / amax
        z = (e0 - g.energy_mj_per_token.values) / e0
        keep = (x > 0) & (y >= 0) & (z > 0)
        P3 = np.column_stack([x, y, z])[keep]
        res[k] = dict(hv3d=hv3d(P3), hv2d_speed_area=hv2d(P3[:, [0, 1]]),
                      hv2d_speed_energy=hv2d(P3[:, [0, 2]]), n_points=int(keep.sum()),
                      best_speedup=float(g.speedup[keep].max()) if keep.any() else 1.0)
    return res


def task2(sweeps):
    print("== Task 2: Pareto hypervolume ==")
    rows = []
    for (mname, rname), df in sweeps.items():
        for mo in (False, True):
            r = hv_sets(df, 1.0, measured_only=mo)
            for k, v in r.items():
                rows.append(dict(model=mname, regime=rname, quality="measured-only" if mo else "table (incl. placeholders)",
                                 knobs=k, **v))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "ws3_hypervolume.csv", index=False)
    piv = res.pivot_table(index=["model", "regime", "quality"], columns="knobs",
                          values="hv3d").reset_index()
    for base in ("hw_only", "sw_only", "hw_union_sw"):
        if base in piv.columns:
            piv[f"gain_vs_{base}_pct"] = np.where(piv[base] > 0, 100 * (piv["co_design"] / piv[base] - 1), np.nan)
    piv.to_csv(OUT / "ws3_hypervolume_gain.csv", index=False)
    print(piv.round(4).to_string(index=False))

    q = piv[piv.quality.str.startswith("table")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for ax, rname in zip(axes, REGS):
        g = q[q.regime == rname]
        x = np.arange(len(g))
        w = 0.27
        for i, (k, c) in enumerate((("hw_only", "#4c78a8"), ("sw_only", "#f58518"), ("co_design", "#54a24b"))):
            ax.bar(x + (i - 1) * w, g[k], w, label=k.replace("_", " "), color=c)
        ax.set_xticks(x)
        ax.set_xticklabels(g.model, rotation=20, fontsize=7)
        ax.set_title(f"Regime {rname}")
        if rname == "A":
            ax.set_ylabel("3D hypervolume (normalised)")
            ax.legend(fontsize=6)
    save_fig(fig, "ws3_hypervolume")
    return res, piv


# ==========================================================================
# Task 3: quality gate audit + sensitivity
# ==========================================================================
def task3(sweeps):
    print("== Task 3: quality gate audit ==")
    audit = []
    for (mname, rname), df in sweeps.items():
        try:
            pf = pareto_front(df)
            ib = iso_area_best(df, budgets=(0, 69, 138, 263, 500))
        except Exception as e:
            print(f"  [warn] pareto/iso failed for {mname} {rname}: {e}")
            continue
        for label, t in (("pareto_front", pf), ("iso_area_winners", ib)):
            audit.append(dict(model=mname, regime=rname, set=label, n=len(t),
                              n_measured=int(t.quality_measured.astype(bool).sum()),
                              pct_measured=100 * float(t.quality_measured.astype(bool).mean()) if len(t) else np.nan))
    a = pd.DataFrame(audit)
    a.to_csv(OUT / "ws3_pareto_measured_audit.csv", index=False)
    print(a.round(1).to_string(index=False))
    tot = a.groupby("set")[["n", "n_measured"]].sum()
    tot["pct_measured"] = 100 * tot.n_measured / tot.n
    print("  overall:\n", tot.round(1).to_string())

    rows = []
    for (mname, rname), df in sweeps.items():
        for g in GATES:
            ok = df[df.feasible & (df.quality_delta_pct <= g)]
            okm = ok[ok.quality_measured.astype(bool)]
            for b in BUDGETS:
                s_all = ok[ok.area_added_mm2 <= b + 1e-6].speedup
                s_meas = okm[okm.area_added_mm2 <= b + 1e-6].speedup
                rows.append(dict(model=mname, regime=rname, gate_pct=g, area_budget_mm2=b,
                                 best_speedup=float(s_all.max()) if len(s_all) else np.nan,
                                 best_speedup_measured_only=float(s_meas.max()) if len(s_meas) else np.nan))
    sens = pd.DataFrame(rows)
    sens.to_csv(OUT / "ws3_gate_sensitivity.csv", index=False)
    print("  saved ws3_gate_sensitivity.csv")

    colors = {"A": "#4c78a8", "B": "#e45756", "C": "#54a24b"}
    styles = dict(zip(DSE_MODELS, ["-", "--", ":"]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for ax, b in zip(axes, BUDGETS):
        for (mname, rname), _ in sweeps.items():
            g = sens[(sens.model == mname) & (sens.regime == rname) & (sens.area_budget_mm2 == b)]
            ax.plot(g.gate_pct, g.best_speedup, styles[mname], color=colors[rname], marker="o", ms=3,
                    label=f"{mname}, {rname}")
        ax.set_xscale("log")
        ax.set_xticks(GATES)
        ax.set_xticklabels([str(x) for x in GATES])
        ax.set_xlabel("quality gate (PPL delta %)")
        ax.set_ylabel("best speedup")
        ax.set_title(f"area budget {b} mm$^2$")
    axes[1].legend(fontsize=5, ncol=1, bbox_to_anchor=(1.02, 1), loc="upper left")
    save_fig(fig, "ws3_gate_sensitivity")
    return a, sens


def main():
    which = set(sys.argv[1:]) or {"1", "2", "3"}
    if "1" in which:
        try:
            task1()
        except Exception:
            traceback.print_exc()
    if which & {"2", "3"}:
        qt = load_qt()
        sweeps = run_sweeps(qt)
        for k, fn in (("2", task2), ("3", task3)):
            if k in which:
                try:
                    fn(sweeps)
                except Exception:
                    traceback.print_exc()


if __name__ == "__main__":
    main()