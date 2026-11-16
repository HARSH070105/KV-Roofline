#!/usr/bin/env python3
"""
Workstream 2: frontier KV-compression schemes vs. baselines across 3 regimes.

Read-only consumer of `kvroofline`. Run from the repo root:

    python extensions/frontier_schemes.py

Outputs -> extensions/outputs/ :
    ws2_full_comparison.csv      every (hw, model, regime, variant, scheme) row
    ws2_decomposition.csv        fixed-batch vs batch-expansion speedup (base HW)
    ws2_profiles_check.csv       stored-KV compression ratio of each profile
    ws2_summary.md               best scheme per regime/model/hardware
    ws2_fig1_decomposition_<hw>.png/.pdf
    ws2_fig2_tput_energy_<model>.png/.pdf
    ws2_fig3_eviction_sram_<model>.png/.pdf

MODELLING ASSUMPTIONS (edit PROFILES below to change them):
  * dequant_flops values, effective bit-widths, and group sizes for the
    2025-2026 schemes are *modelling choices*, not numbers read from the papers.
  * Expected Attention and StreamingLLM map to the same `evict_policy` in the
    engine, so their speed results are identical; they differ only in quality.
  * CSA2 / cross-layer KV sharing is NOT modelled: `Compression` has no field
    for it, and faking it by scaling bits would be misleading.
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

from kvroofline import Compression, evaluate, get_hw, get_model, max_batch  # noqa: E402

try:
    from kvroofline.quality import QualityTable  # noqa: E402
except Exception:  # quality audit is optional
    QualityTable = None

OUT = Path(__file__).resolve().parent

plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 120, "savefig.bbox": "tight"})

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
HWS = ["h100", "b200"]
MODELS = ["llama3.1-8b", "qwen2.5-7b", "deepseek-v2-lite"]
REGIMES = {  # name: (S, batch cap, n_chips)
    "A": (8192, 8, 1),
    "B": (131072, 8, 1),
    "C": (524288, 8, 2),
}
VARIANTS = ["base", "+1HBM", "+2HBM", "+SRAM512"]
BUDGET = 16384


def C(**kw):
    return Compression(**kw)


PROFILES = {
    "BF16_Baseline": C(kv_bits=16, group=0, residual=0, dequant_flops=0),
    "FP8_KV": C(kv_bits=8, group=0, residual=0, dequant_flops=1),
    "KIVI_INT4_g32": C(kv_bits=4, group=32, residual=128, dequant_flops=6),
    "KVQuant_INT4_g64": C(kv_bits=4, group=64, residual=128, dequant_flops=8),
    "KIVI_INT2_g32": C(kv_bits=2, group=32, residual=128, dequant_flops=6),
    # PM-KVQ: avg ~2.5 bits for old tokens, big high-precision recent window
    "PM_KVQ_Mixed": C(kv_bits=2.5, group=64, residual=512, dequant_flops=6),
    # AATC: ~5.8x total compression = 2 B / (2.5/8 + 4/128 B) with transform cost
    "AATC_Transform": C(kv_bits=2.5, group=128, residual=128, dequant_flops=12),
    "StreamingLLM_16k": C(kv_bits=16, group=0, residual=0, dequant_flops=0,
                          evict_policy="streaming", evict_budget=BUDGET),
    "Quest_16k": C(kv_bits=16, group=0, residual=0, dequant_flops=0,
                   evict_policy="quest", evict_budget=BUDGET),
    "ExpectedAttn_16k": C(kv_bits=16, group=0, residual=0, dequant_flops=0,
                          evict_policy="streaming", evict_budget=BUDGET),
    "Hybrid_INT4_Quest16k": C(kv_bits=4, group=64, residual=128, dequant_flops=6,
                              evict_policy="quest", evict_budget=BUDGET),
    "Hybrid_AATC_ExpectedAttn16k": C(kv_bits=2.5, group=128, residual=128,
                                     dequant_flops=12, evict_policy="streaming",
                                     evict_budget=BUDGET),
}
BASE = "BF16_Baseline"
EVICTION_SCHEMES = ["Quest_16k", "StreamingLLM_16k", "ExpectedAttn_16k",
                    "Hybrid_INT4_Quest16k", "Hybrid_AATC_ExpectedAttn16k"]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
_warned = set()


def warn_once(msg):
    if msg not in _warned:
        _warned.add(msg)
        print(f"  [warn] {msg}")


def make_hw(hname, chips, variant):
    hw = get_hw(hname)
    if chips > 1:
        hw = hw.with_(n_chips=chips)
    if variant in ("+1HBM", "+2HBM"):
        stacks = getattr(hw, "hbm_stacks", None)
        if stacks is None:
            warn_once("hw has no hbm_stacks attribute; HBM variants skipped")
            return None
        hw = hw.with_(hbm_stacks=int(stacks) + int(variant[1]))
    elif variant == "+SRAM512":
        hw = hw.with_(sram_pin_mb=512)
    return hw


def ev(m, hw, S, B, comp):
    try:
        return evaluate(m, hw, S=S, B=B, comp=comp)
    except Exception as e:
        warn_once(f"evaluate failed ({type(e).__name__}: {e})")
        return None


def mb(m, hw, S, comp):
    try:
        return int(max_batch(m, hw, S=S, comp=comp))
    except TypeError:
        try:
            return int(max_batch(m, hw, S, comp))
        except Exception as e:
            warn_once(f"max_batch failed ({type(e).__name__}: {e})")
            return 0
    except Exception as e:
        warn_once(f"max_batch failed ({type(e).__name__}: {e})")
        return 0


def save_fig(fig, name):
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print(f"  saved {name}.png")


def load_quality():
    if QualityTable is None:
        return None
    p = ROOT / "data" / "quality_table.csv"
    try:
        return QualityTable.from_csv(str(p))
    except Exception as e:
        warn_once(f"quality table not loaded ({type(e).__name__}); skipping gate columns")
        return None


# --------------------------------------------------------------------------
# Main sweep
# --------------------------------------------------------------------------
def run_sweep():
    qt = load_quality()
    rows = []
    for hname in HWS:
        for mname in MODELS:
            m = get_model(mname)
            for rname, (S, cap, chips) in REGIMES.items():
                for variant in VARIANTS:
                    hw = make_hw(hname, chips, variant)
                    if hw is None:
                        continue
                    bcomp = PROFILES[BASE]
                    mb_base = mb(m, hw, S, bcomp)
                    B_base = max(1, min(cap, mb_base))
                    base_n = ev(m, hw, S, B_base, bcomp)
                    base_1 = ev(m, hw, S, 1, bcomp)
                    if base_n is None or base_1 is None:
                        continue
                    for pname, comp in PROFILES.items():
                        mbx = mb(m, hw, S, comp)
                        B_eff = min(cap, mbx)
                        row = dict(hw=hname, model=mname, regime=rname, S=S,
                                   chips=chips, variant=variant, scheme=pname,
                                   max_batch=mbx, B_eff=B_eff, B_base=B_base,
                                   base_feasible=mb_base >= 1)
                        if qt is not None:
                            try:
                                d, meas = qt.lookup(comp, S)
                                status = ("pass_measured" if meas and d <= 1.0 else
                                          "fail_measured" if meas else "unverified")
                                q_val = float(d) if meas else np.nan
                                row.update(quality_delta_pct=q_val, quality_measured=bool(meas),
                                           passes_1pct=bool(meas and d <= 1.0),
                                           gate_status=status)
                            except Exception:
                                pass
                        if B_eff < 1:
                            row["feasible"] = False
                            rows.append(row)
                            continue
                        r1 = ev(m, hw, S, 1, comp)
                        rn = ev(m, hw, S, B_eff, comp)
                        rf = ev(m, hw, S, B_base, comp)
                        if r1 is None or rn is None or rf is None:
                            continue
                        total = rn["tokens_per_s"] / base_n["tokens_per_s"]
                        fixed = rf["tokens_per_s"] / base_n["tokens_per_s"]
                        row.update(
                            feasible=True,
                            tpot_ms_B1=r1["tpot_ms"],
                            latency_speedup_B1=base_1["tpot_ms"] / r1["tpot_ms"],
                            tokens_per_s=rn["tokens_per_s"],
                            energy_mj_per_token=rn["energy_mj_per_token"],
                            tokens_per_joule=rn["tokens_per_joule"],
                            ai_attn=rn["ai_attn"], bind_attn=rn["bind_attn"],
                            dominant=rn["dominant"], h_kv=rn["h_kv"],
                            kv_stored_gb_total=rn["kv_stored_gb_total"],
                            kv_hbm_gb=rn["kv_hbm_gb"],
                            speedup_total=total,
                            speedup_fixed_batch=fixed,
                            speedup_batch_expansion=total / fixed if fixed else np.nan,
                            base_tokens_per_s=base_n["tokens_per_s"],
                        )
                        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "ws2_full_comparison.csv", index=False)
    print(f"  saved ws2_full_comparison.csv ({len(df)} rows)")
    return df


def profile_check():
    """Sanity table: stored-KV compression ratio of each profile (B=1, regime B)."""
    m = get_model("llama3.1-8b")
    hw = get_hw("h100")
    S = 131072
    base = ev(m, hw, S, 1, PROFILES[BASE])
    rows = []
    for p, comp in PROFILES.items():
        r = ev(m, hw, S, 1, comp)
        if r is None or base is None:
            continue
        rows.append(dict(scheme=p,
                         stored_kv_ratio=base["kv_stored_gb_total"] / r["kv_stored_gb_total"],
                         read_kv_ratio=base["kv_hbm_gb"] / r["kv_hbm_gb"],
                         ai_attn=r["ai_attn"]))
    pc = pd.DataFrame(rows)
    pc.to_csv(OUT / "ws2_profiles_check.csv", index=False)
    print("  profile sanity (llama3.1-8b, S=131072, H100, B=1):")
    print(pc.round(2).to_string(index=False))
    print("  expected: INT4 g64 ~3.56x, AATC ~5.7x. Quest/streaming change READ "
          "bytes (read_kv_ratio), not stored bytes.")
    return pc


# --------------------------------------------------------------------------
# Decomposition table + figures
# --------------------------------------------------------------------------
def decomposition(df):
    d = df[(df.variant == "base") & (df.feasible == True)].copy()  # noqa: E712
    cols = ["hw", "model", "regime", "scheme", "B_base", "B_eff", "speedup_fixed_batch",
            "speedup_batch_expansion", "speedup_total", "latency_speedup_B1"]
    d = d[cols]
    d.to_csv(OUT / "ws2_decomposition.csv", index=False)
    print("  saved ws2_decomposition.csv")
    return d


def fig1(df):
    for hname in HWS:
        fig, axes = plt.subplots(2, len(MODELS), figsize=(5.2 * len(MODELS), 7),
                                 sharey="row")
        for i, rname in enumerate(["B", "C"]):
            for j, mname in enumerate(MODELS):
                ax = axes[i, j]
                g = df[(df.hw == hname) & (df.model == mname) & (df.regime == rname)
                       & (df.variant == "base")].set_index("scheme")
                xs = np.arange(len(PROFILES))
                fixed, extra, beff = [], [], []
                for p in PROFILES:
                    if p in g.index and g.loc[p, "feasible"] == True:  # noqa: E712
                        f = g.loc[p, "speedup_fixed_batch"]
                        t = g.loc[p, "speedup_total"]
                        fixed.append(f); extra.append(max(t - f, 0)); beff.append(int(g.loc[p, "B_eff"]))
                    else:
                        fixed.append(0); extra.append(0); beff.append(0)
                ax.bar(xs, fixed, color="#4c78a8", label="fixed-batch (traffic cut)")
                ax.bar(xs, extra, bottom=fixed, color="#f58518",
                       label="batch expansion (capacity)")
                for x, f, e, b in zip(xs, fixed, extra, beff):
                    ax.text(x, f + e, "n/a" if b == 0 else f"B={b}", ha="center",
                            va="bottom", fontsize=6)
                ax.set_xticks(xs)
                ax.set_xticklabels(list(PROFILES), rotation=70, ha="right", fontsize=6)
                ax.set_title(f"{mname}, Regime {rname}")
                if j == 0:
                    ax.set_ylabel("throughput speedup vs BF16")
                if i == 0 and j == 0:
                    ax.legend(fontsize=6)
        fig.suptitle(f"Speedup decomposition on {hname.upper()} "
                     "(additive stack: total = fixed + expansion)", y=1.0)
        fig.tight_layout()
        save_fig(fig, f"ws2_fig1_decomposition_{hname}")


def fig2(df):
    cmap = plt.get_cmap("tab20")
    colors = {p: cmap(i) for i, p in enumerate(PROFILES)}
    for mname in MODELS:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, rname in zip(axes, REGIMES):
            for hname, mk in (("h100", "o"), ("b200", "^")):
                g = df[(df.hw == hname) & (df.model == mname) & (df.regime == rname)
                       & (df.variant == "base") & (df.feasible == True)]  # noqa: E712
                for _, r in g.iterrows():
                    ax.scatter(r.tokens_per_s, r.energy_mj_per_token, marker=mk,
                               color=colors[r.scheme], s=40, edgecolor="k", lw=.4)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel("system throughput (tokens/s)")
            ax.set_ylabel("energy (mJ/token)")
            ax.set_title(f"{mname}, Regime {rname}  (o = H100, ^ = B200)")
        handles = [plt.Line2D([], [], marker="s", ls="", color=colors[p], label=p)
                   for p in PROFILES]
        fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, .5), fontsize=6)
        fig.tight_layout()
        save_fig(fig, f"ws2_fig2_tput_energy_{mname}")


def fig3(df):
    for mname in MODELS:
        fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
        for i, rname in enumerate(["B", "C"]):
            g = df[(df.hw == "h100") & (df.model == mname) & (df.regime == rname)
                   & (df.feasible == True)]  # noqa: E712
            xs = np.arange(len(EVICTION_SCHEMES))
            w = 0.38
            for k, (variant, col) in enumerate((("base", "#4c78a8"), ("+SRAM512", "#54a24b"))):
                sp, hk = [], []
                for p in EVICTION_SCHEMES:
                    v = g[(g.scheme == p) & (g.variant == variant)]
                    sp.append(v.speedup_total.iloc[0] if len(v) else 0)
                    hk.append(v.h_kv.iloc[0] if len(v) else 0)
                axes[i, 0].bar(xs + (k - .5) * w, sp, w, color=col,
                               label="no pinned SRAM" if variant == "base" else "+512 MB SRAM")
                axes[i, 1].bar(xs + (k - .5) * w, hk, w, color=col)
            for a, yl in ((axes[i, 0], "throughput speedup vs BF16"),
                          (axes[i, 1], "on-chip hit fraction h_kv")):
                a.set_xticks(xs)
                a.set_xticklabels(EVICTION_SCHEMES, rotation=30, ha="right", fontsize=6)
                a.set_ylabel(yl)
            axes[i, 0].set_title(f"{mname}, Regime {rname} (H100)")
            axes[i, 1].set_ylim(0, 1.05)
            axes[i, 1].set_title("does the working set fit in SRAM? (h_kv -> 1)")
            if i == 0:
                axes[i, 0].legend(fontsize=6)
        fig.tight_layout()
        save_fig(fig, f"ws2_fig3_eviction_sram_{mname}")


def write_summary(df, dec):
    L = ["# Workstream 2 summary (auto-generated)\n",
         "Assumptions: see header of frontier_schemes.py. Expected Attention and "
         "StreamingLLM are identical in the engine (quality differs only).\n"]
    ok = df[(df.variant == "base") & (df.feasible == True)]  # noqa: E712
    best = (ok.sort_values("tokens_per_s", ascending=False)
            .groupby(["hw", "model", "regime"]).head(3))
    best = best[["hw", "model", "regime", "scheme", "B_eff", "tokens_per_s",
                 "speedup_total", "speedup_fixed_batch", "speedup_batch_expansion",
                 "energy_mj_per_token"]]
    cols = ["hw", "model", "regime", "scheme", "B_eff", "tokens_per_s",
            "speedup_total", "speedup_fixed_batch", "speedup_batch_expansion",
            "energy_mj_per_token"]
    tables = [("Top 3 schemes by system throughput (quality NOT considered; "
               "unmeasured quality entries are placeholders)", best)]
    if "gate_status" in ok.columns:
        gated = (ok[ok.gate_status == "pass_measured"]
                 .sort_values("tokens_per_s", ascending=False)
                 .groupby(["hw", "model", "regime"]).head(1))[cols]
        tables.append(("Best scheme that PASSES the 1% gate with MEASURED quality", gated))
        unv = ok[ok.gate_status == "unverified"][["scheme", "regime"]].drop_duplicates()
        L.append("Schemes with unmeasured quality (placeholder deltas, do not quote as "
                 "gate-passing): " + ", ".join(sorted(unv.scheme.unique())) + "\n")
    for title, t in tables:
        L.append(f"## {title}\n")
        try:
            L.append(t.to_markdown(index=False, floatfmt=".2f"))
        except Exception:
            L.append(t.to_string(index=False))
        L.append("")
    (OUT / "ws2_summary.md").write_text("\n".join(L))
    print("  saved ws2_summary.md")


def main():
    print("== Workstream 2: frontier compression schemes ==")
    try:
        profile_check()
    except Exception:
        traceback.print_exc()
    df = run_sweep()
    if df.empty or "feasible" not in df:
        print("  no results; aborting")
        return
    dec = decomposition(df)
    for fn in (fig1, fig2, fig3):
        try:
            fn(df)
        except Exception:
            traceback.print_exc()
    write_summary(df, dec)


if __name__ == "__main__":
    main()