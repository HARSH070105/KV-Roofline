"""Matplotlib figures for the report. Palette = validated categorical slots (blue, orange, aqua, magenta),
thin marks, recessive grid, and a second encoding (marker / linestyle) so colour is never the only cue."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .compression import Compression
from .engine import evaluate, crossover_S

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#e87ba4", "#eda100", "#6250d6"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
FAMILY_STYLE = {"baseline": (INK2, "o"), "SRAM": (SERIES[0], "s"), "HBM": (SERIES[1], "^"),
                "COMP": (SERIES[2], "D"), "SRAM+COMP": (SERIES[3], "P"), "HBM+COMP": (SERIES[4], "v"),
                "SRAM+HBM": (SERIES[5], "X"), "SRAM+HBM+COMP": ("#8d8c85", "*")}


def _style(ax, title=None, xl=None, yl=None):
    ax.set_facecolor(SURF)
    ax.figure.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left", fontweight="bold")
    if xl:
        ax.set_xlabel(xl, color=INK2, fontsize=9)
    if yl:
        ax.set_ylabel(yl, color=INK2, fontsize=9)


def plot_roofline(model, hw, S, B, comps, path, attn_unit="tensor"):
    """Log-log roofline with the tensor roof, CUDA roof, and one point per compression config."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    ai = np.logspace(-1, 4, 400)
    bw = hw.hbm_bw
    ax.loglog(ai, np.minimum(hw.p_tensor, bw * ai) / 1e12, color=INK, lw=2, label="tensor-core roof")
    ax.loglog(ai, np.minimum(hw.p_cuda, bw * ai) / 1e12, color=INK2, lw=1.5, ls="--", label="CUDA-core roof")
    ax.axvline(hw.ridge_tensor, color=GRID, lw=1)
    ax.text(hw.ridge_tensor * 1.05, hw.p_tensor / 1e12 * 0.3, f"ridge {hw.ridge_tensor:.0f}", color=INK2, fontsize=8)
    ax.text(hw.ridge_cuda * 1.05, hw.p_cuda / 1e12 * 0.3, f"CUDA ridge {hw.ridge_cuda:.0f}", color=INK2, fontsize=8)
    for i, c in enumerate(comps):
        r = evaluate(model, hw, S, B, c, attn_unit=attn_unit)
        ax.scatter([r["ai_step"]], [r["achieved_tflops"]], color=SERIES[i % 6], marker=MARKERS[i % 6], s=46,
                   zorder=3, edgecolor=SURF, linewidth=1.5, label=f"{c.label()}  (attn AI {r['ai_attn']:.0f})")
    _style(ax, f"{model.name} decode on {hw.name}: S={S:,}, B={B}", "arithmetic intensity (FLOP/B, HBM-level)",
           "achieved (TFLOP/s)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_crossover(models, hw, B, path, comp=Compression(), s_range=(512, 2_000_000)):
    """Share of HBM traffic that is KV vs context length; marks S*."""
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=160)
    Ss = np.unique(np.logspace(np.log10(s_range[0]), np.log10(s_range[1]), 60).astype(int))
    for i, m in enumerate(models):
        sh = [evaluate(m, hw.with_(n_chips=1), int(S), B, comp)["kv_share_of_hbm"] for S in Ss]
        ax.semilogx(Ss, sh, color=SERIES[i % 6], lw=2, marker=MARKERS[i % 6], markevery=8, ms=5, label=m.name)
        sstar = crossover_S(m, B, comp)
        if s_range[0] < sstar < s_range[1]:
            ax.scatter([sstar], [0.5], color=SERIES[i % 6], s=60, zorder=4, edgecolor=SURF, linewidth=1.5)
    ax.axhline(0.5, color=GRID, lw=1)
    _style(ax, f"KV share of HBM traffic vs context (B={B}, {comp.label()}); dots = S*", "context S (tokens)",
           "KV bytes / total HBM bytes")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_pareto(df, front, path, title):
    d = df[df["feasible"] & df["passes_gate"]]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    for fam, g in d.groupby("family"):
        col, mk = FAMILY_STYLE.get(fam, ("#8d8c85", "o"))
        ax.scatter(g["area_added_mm2"], g["speedup"], s=14, color=col, marker=mk, alpha=0.55, label=fam, linewidths=0)
    ax.step(front["area_added_mm2"], front["speedup"], where="post", color=INK, lw=1.4)
    ax.scatter(front["area_added_mm2"], front["speedup"], s=36, color=INK, zorder=4, edgecolor=SURF, linewidth=1.2,
               label="Pareto (tokens/s, area, energy)")
    _style(ax, title, "added area (mm$^2$, die + 0.5 x package)", "tokens/s speedup vs baseline")
    ax.legend(frameon=False, fontsize=7, labelcolor=INK2, ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_knob_curves(curves: dict, path, title):
    """Speedup vs added area for each single-knob family."""
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=160)
    for i, key in enumerate(["SRAM", "HBM"]):
        d = curves[key]
        d = d[d["passes_gate"]]
        if d.empty:
            continue
        g = d.groupby("area_added_mm2")["speedup"].max().reset_index()
        ax.plot(g["area_added_mm2"], g["speedup"], color=SERIES[i], marker=MARKERS[i], lw=2, ms=5, label=key)
    c = curves["COMP"]
    c = c[c["passes_gate"]]
    if not c.empty:
        best = c["speedup"].max()
        ax.axhline(best, color=SERIES[2], lw=2, ls="--", label=f"best gated compression: {best:.2f}x at ~0 mm$^2$")
    ax.axhline(1.0, color=GRID, lw=1)
    _style(ax, title, "added area (mm$^2$)", "tokens/s speedup")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_heatmap(df, path, title):
    """SRAM extra x HBM extra stacks: best gated speedup with no compression."""
    d = df[(df["kv_bits"] == 16) & (df["evict_policy"] == "none") & (df["hbm_speed"] == 1.0) & df["feasible"]]
    if d.empty:
        return
    pv = d.groupby(["hbm_extra_stacks", "sram_extra_mb"])["speedup"].max().unstack()
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=160)
    im = ax.imshow(pv.values, cmap="Blues", aspect="auto", origin="lower")
    ax.set_xticks(range(pv.shape[1]), [f"{int(c)}" for c in pv.columns])
    ax.set_yticks(range(pv.shape[0]), [f"+{int(r)}" for r in pv.index])
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            v = pv.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if v > pv.values.max() * 0.6 else INK)
    ax.set_xlabel("extra SRAM (MB)", color=INK2, fontsize=9)
    ax.set_ylabel("extra HBM stacks", color=INK2, fontsize=9)
    ax.set_title(title, color=INK, fontsize=11, loc="left", fontweight="bold")
    ax.tick_params(colors=INK2, labelsize=9)
    fig.set_facecolor(SURF)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
