"""Quality lookup table Q and the quality gate (deck slide 6).

Quality-affecting knobs (KV bits/group, eviction budget/policy, weight bits) are measured ONCE per
(method, bits, budget, context) and stored in a CSV. Hardware sweeps then reuse the table for free.
Quality is a CONSTRAINT (gate), not an objective: configs worse than `gate_pct` are dropped before ranking.

CSV schema (data/quality_table.csv):
  kv_bits,group,evict_policy,evict_budget,w_bits,context,metric,delta_pct,source
  - delta_pct: relative degradation vs BF16 in percent (e.g. perplexity increase 100*(ppl/ppl_bf16-1)).
  - evict_budget: empty / 0 = none.   group: 0 if not applicable.
  - source: "measured" for real runs.  Anything else is treated as unmeasured.

Until week 5 the table does not exist, so lookup falls back to `placeholder_delta_pct`, which is an
ASSUMED monotone heuristic. Every DSE row carries quality_measured=False in that case; do NOT quote
rankings from placeholder quality in the report.
"""
from __future__ import annotations

import math
from typing import Optional

import pandas as pd

from .compression import Compression

KEY_COLS = ["kv_bits", "group", "evict_policy", "evict_budget", "w_bits"]


def placeholder_delta_pct(comp: Compression, S: int) -> float:
    """ASSUMED degradation heuristic (percent). Replace by measured data."""
    d = 0.0
    kvq = 0.0
    if comp.kv_bits == 8:
        kvq = 0.05
    elif comp.kv_bits == 4:
        kvq = 0.4 if 0 < comp.group <= 64 else (0.8 if comp.group > 64 else 1.5)
    elif comp.kv_bits == 2:
        kvq = 3.0 if 0 < comp.group <= 32 else 4.5
    # deck: compressions degrade with context
    kvq *= 1.0 + 0.15 * max(0.0, math.log2(max(S, 8192) / 8192))
    d += kvq
    ev = 0.0
    if comp.evict_policy != "none" and comp.evict_budget and comp.evict_budget < S:
        r = comp.evict_budget / S
        ev = 0.5 * math.sqrt(1.0 / r - 1.0)
        if comp.evict_policy == "streaming":
            ev *= 1.5
    d += ev
    if kvq > 0 and ev > 0:
        d *= 1.2   # deck: compressions are not additive
    if comp.w_bits == 8:
        d += 0.1
    elif comp.w_bits == 4:
        d += 1.0
    return d


class QualityTable:
    def __init__(self, df: Optional[pd.DataFrame] = None):
        self.df = df
        self._idx = None
        if df is not None and len(df):
            d = df.copy()
            d["evict_budget"] = d["evict_budget"].fillna(0).astype(int)
            d["group"] = d["group"].fillna(0).astype(int)
            d["evict_policy"] = d["evict_policy"].fillna("none")
            self.df = d
            self._idx = {k: g for k, g in d.groupby(KEY_COLS)}

    @classmethod
    def from_csv(cls, path: str) -> "QualityTable":
        return cls(pd.read_csv(path))

    @classmethod
    def placeholder(cls) -> "QualityTable":
        return cls(None)

    @staticmethod
    def _key(comp: Compression) -> tuple:
        g = comp.group if comp.kv_bits < 16 else 0
        b = int(comp.evict_budget or 0) if comp.evict_policy != "none" else 0
        return (comp.kv_bits, g, comp.evict_policy, b, comp.w_bits)

    def lookup(self, comp: Compression, S: int) -> tuple[float, bool]:
        """Returns (delta_pct, measured?). Nearest measured context in log2 distance."""
        if comp.kv_bits == 16 and comp.evict_policy == "none" and comp.w_bits == 16:
            return 0.0, True
        if self._idx is not None:
            g = self._idx.get(self._key(comp))
            if g is not None:
                g = g[g["source"].astype(str).str.lower() == "measured"]
                if len(g):
                    j = (g["context"].map(lambda c: abs(math.log2(max(c, 1)) - math.log2(max(S, 1))))).idxmin()
                    return float(g.loc[j, "delta_pct"]), True
        return placeholder_delta_pct(comp, S), False

    def passes(self, comp: Compression, S: int, gate_pct: float = 1.0) -> tuple[bool, float, bool]:
        d, measured = self.lookup(comp, S)
        return d <= gate_pct, d, measured
