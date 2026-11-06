"""Design-space exploration: grid sweep, quality gate, Pareto fronts, iso-area recommendations.

Efficiency currency (deck slide 7B)
  d_ai            : change in step-level HBM arithmetic intensity vs the baseline
  d_ai_over_ridge : change in AI/ridge. NOTE: extra HBM bandwidth leaves AI unchanged (d_ai = 0 by
                    construction) but moves the ridge; AI/ridge and tokens/s capture that, d_ai alone does not.
  speedup         : tokens/s relative to baseline at the same regime (batch mode applies to both)
  per_mm2 / per_mj: the above divided by added area (mm^2) or energy change (mJ/token)
Compression knobs add ~0 area, so area-normalised ratios are undefined for them; use the iso-area
table (best config under an area budget, quality-gated) to compare knob families fairly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Optional

import numpy as np
import pandas as pd

from .compression import Compression
from .cost import CostParams, DEFAULT_COST
from .engine import Calibration, SramPolicy, evaluate, resolve_batch
from .hardware import Hardware
from .models import ModelConfig
from .quality import QualityTable


@dataclass(frozen=True)
class Regime:
    name: str
    S: int
    B: int            # target (fixed mode) or cap (max mode)
    note: str = ""


DEFAULT_REGIMES = [
    Regime("A_short_medium", 8_192, 8, "weight-dominated: 8k-32k (S* ~ 14k at B=8)"),
    Regime("B_long", 131_072, 8, "KV-dominated: 128k"),
    Regime("C_ultra_long", 524_288, 8, "capacity-bound: 512k+"),
]


@dataclass
class Grid:
    sram_extra_mb: list = field(default_factory=lambda: [0, 64, 128, 256, 512, 1024])
    hbm_extra_stacks: list = field(default_factory=lambda: [0, 1, 2])
    hbm_speed: list = field(default_factory=lambda: [1.0])   # opt-in: pin-speed has no area cost in the model
    sram_kv_frac: list = field(default_factory=lambda: [1.0, 0.0])
    kv_bits: list = field(default_factory=lambda: [16, 8, 4, 2])
    groups: list = field(default_factory=lambda: [32, 64, 128])
    residual: int = 128
    evict: list = field(default_factory=lambda: [("none", None), ("streaming", 4096), ("quest", 4096),
                                                  ("quest", 16384)])


def _comp_list(grid: Grid, S: int, dequant_flops: float, w_bits: int = 16) -> list[Compression]:
    out = []
    for bits in grid.kv_bits:
        if bits >= 16:
            gl = [0]
        elif bits == 8:
            gl = [0]                    # FP8, per-tensor scale -> no per-group metadata
        else:
            gl = grid.groups
        for g in gl:
            for pol, bud in grid.evict:
                if pol != "none" and (bud is None or bud >= S):
                    continue            # budget no smaller than S is the same as no eviction
                out.append(Compression(kv_bits=bits, group=g, residual=grid.residual if bits < 16 else 0,
                                       evict_policy=pol, evict_budget=bud, w_bits=w_bits,
                                       dequant_flops=dequant_flops))
    return out


def _family(sram_extra, stacks_extra, speed, comp: Compression) -> str:
    parts = []
    if sram_extra > 0:
        parts.append("SRAM")
    if stacks_extra > 0 or speed > 1.0:
        parts.append("HBM")
    if comp.kv_bits < 16 or comp.evict_policy != "none":
        parts.append("COMP")
    return "+".join(parts) if parts else "baseline"


def sweep(model: ModelConfig, base_hw: Hardware, regime: Regime, grid: Grid = Grid(),
          qt: Optional[QualityTable] = None, gate_pct: float = 1.0, batch_mode: str = "max",
          theta: float = 1.0, rho: float = 0.0, dequant_flops: float = 6.0,
          attn_unit: str = "tensor", cost: CostParams = DEFAULT_COST,
          calib: Calibration = Calibration()) -> pd.DataFrame:
    qt = qt or QualityTable.placeholder()
    S = regime.S
    comps = _comp_list(grid, S, dequant_flops)

    def _eval(hw, comp, frac):
        B = resolve_batch(model, hw, S, comp, batch_mode, regime.B)
        if B < 1:
            return None
        return evaluate(model, hw, S, B, comp, SramPolicy(frac, theta, rho), calib, cost,
                        base_hw=base_hw, attn_unit=attn_unit)

    rows = []
    base_row = _eval(base_hw, Compression(dequant_flops=dequant_flops), 1.0)
    for sx, st, sp, frac in product(grid.sram_extra_mb, grid.hbm_extra_stacks, grid.hbm_speed, grid.sram_kv_frac):
        hw = base_hw.with_(sram_pin_mb=base_hw.sram_pin_mb + sx,
                           hbm_stacks=base_hw.hbm_stacks + st, hbm_speed_scale=sp)
        for comp in comps:
            r = _eval(hw, comp, frac)
            if r is None:
                continue
            ok, dq, measured = qt.passes(comp, S, gate_pct)
            r.update(regime=regime.name, sram_extra_mb=sx, hbm_extra_stacks=st, hbm_speed=sp, sram_kv_frac=frac,
                     kv_bits=comp.kv_bits, group=comp.group, evict_policy=comp.evict_policy,
                     evict_budget=comp.evict_budget or 0, quality_delta_pct=dq, quality_measured=measured,
                     passes_gate=ok, family=_family(sx, st, sp, comp))
            rows.append(r)
    df = pd.DataFrame(rows)
    if base_row is None:
        raise RuntimeError("baseline does not fit in memory for this regime; use more chips or a smaller model")
    for col in ("tokens_per_s", "ai_step", "ai_over_ridge", "energy_mj_per_token", "tokens_per_joule", "B"):
        df[f"base_{col}"] = base_row[col]
    df["speedup"] = df["tokens_per_s"] / base_row["tokens_per_s"]
    df["d_ai"] = df["ai_step"] - base_row["ai_step"]
    df["d_ai_over_ridge"] = df["ai_over_ridge"] - base_row["ai_over_ridge"]
    df["d_energy_mj"] = df["energy_mj_per_token"] - base_row["energy_mj_per_token"]
    a = df["area_added_mm2"]
    df["d_ai_per_mm2"] = np.where(a > 1e-9, df["d_ai"] / a.where(a > 1e-9), np.nan)
    df["d_ai_ridge_per_mm2"] = np.where(a > 1e-9, df["d_ai_over_ridge"] / a.where(a > 1e-9), np.nan)
    df["speedup_gain_per_mm2"] = np.where(a > 1e-9, (df["speedup"] - 1) / a.where(a > 1e-9), np.nan)
    df["tok_per_s_gain_per_mm2"] = np.where(a > 1e-9, (df["tokens_per_s"] - base_row["tokens_per_s"]) / a.where(a > 1e-9), np.nan)
    df["d_ai_per_mj_saved"] = np.where(df["d_energy_mj"] < -1e-9, df["d_ai"] / -df["d_energy_mj"], np.nan)
    return df


# --------------------------------------------------------------------------------------
def pareto_mask(df: pd.DataFrame, maximize: list[str], minimize: list[str]) -> np.ndarray:
    """True for non-dominated rows. O(n^2) but n is a few thousand."""
    X = np.column_stack([df[c].to_numpy(float) for c in maximize] +
                        [-df[c].to_numpy(float) for c in minimize])
    n = len(X)
    keep = np.ones(n, bool)
    order = np.argsort(-X[:, 0])
    for i in order:
        if not keep[i]:
            continue
        dom = np.all(X >= X[i], axis=1) & np.any(X > X[i], axis=1)
        if dom.any():
            keep[i] = False
    return keep


def pareto_front(df: pd.DataFrame, gated: bool = True) -> pd.DataFrame:
    d = df[df["feasible"]]
    if gated:
        d = d[d["passes_gate"]]
    d = d.reset_index(drop=True)
    m = pareto_mask(d, ["tokens_per_s"], ["area_added_mm2", "energy_mj_per_token"])
    return d[m].sort_values("area_added_mm2").reset_index(drop=True)


def iso_area_best(df: pd.DataFrame, budgets=(0, 25, 50, 100, 200, 400), metric="tokens_per_s") -> pd.DataFrame:
    d = df[df["feasible"] & df["passes_gate"]]
    out = []
    for b in budgets:
        s = d[d["area_added_mm2"] <= b + 1e-9]
        if s.empty:
            continue
        r = s.loc[s[metric].idxmax()]
        out.append(r.to_frame().T.assign(area_budget_mm2=b))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def best_by_family(df: pd.DataFrame, metric="tokens_per_s") -> pd.DataFrame:
    d = df[df["feasible"] & df["passes_gate"]]
    idx = d.groupby("family")[metric].idxmax()
    return d.loc[idx].sort_values(metric, ascending=False).reset_index(drop=True)


def single_knob_curves(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """One-knob-at-a-time marginal curves (all other knobs at baseline) for the headline efficiency table."""
    base = df[(df["sram_extra_mb"] == 0) & (df["hbm_extra_stacks"] == 0) & (df["hbm_speed"] == 1.0) &
              (df["kv_bits"] == 16) & (df["evict_policy"] == "none")]
    sram = df[(df["family"].isin(["SRAM", "baseline"])) & (df["hbm_extra_stacks"] == 0)]
    hbm = df[(df["family"].isin(["HBM", "baseline"])) & (df["sram_extra_mb"] == 0)]
    comp = df[(df["family"].isin(["COMP", "baseline"])) & (df["sram_extra_mb"] == 0) &
              (df["hbm_extra_stacks"] == 0) & (df["hbm_speed"] == 1.0)]
    return {"SRAM": sram[sram["feasible"]], "HBM": hbm[hbm["feasible"]], "COMP": comp[comp["feasible"]],
            "baseline": base}
