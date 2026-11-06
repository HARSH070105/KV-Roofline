"""On-chip SRAM hit-rate model h(C).

Key physical fact the model captures (and the deck flags as "the key input"):
in decode every KV byte is read exactly ONCE per step. A cache that must hold the whole
working set W gets h = C/W at best (pinning), and an LRU cache gets ~0 on a cyclic scan.
So a plain bigger-cache only pays off when
   (a) the working set is small  (eviction / sparse attention / small B),
   (b) accesses are skewed       (sinks + recent window; theta < 1),
   (c) sequences share data      (prefix sharing across the batch; rho > 0).

    h = [ rho*W * min(1, C/Ws)  +  (1-rho)*W * min(1, C_left/Wp)^theta ] / W
        Ws = rho*W/B (stored size of the shared part), Wp = (1-rho)*W, C_left = C - min(C, Ws)

theta = 1 : uniform access (pessimistic).   theta ~ 0.5 : strongly skewed (sink + recent window).
THETA IS AN ASSUMPTION until you measure it (see README section 4).
"""
from __future__ import annotations


def kv_hit_rate(C: float, W: float, B: int = 1, theta: float = 1.0, rho: float = 0.0,
                foot_scale: float = 1.0) -> float:
    """Fraction of KV *read* bytes served on-chip.

    C: capacity (bytes) dedicated to KV pinning.  W: total KV bytes read per step (all B sequences).
    foot_scale: stored footprint / bytes read (>1 for Quest-style selection: the pages read change every
                step, so what must be pinned is the whole stored cache, not just the bytes read).
    """
    C = C / max(foot_scale, 1.0)
    if W <= 0:
        return 1.0
    if C <= 0:
        return 0.0
    B = max(int(B), 1)
    rho = min(max(rho, 0.0), 1.0)
    Ws = rho * W / B
    Wp = (1.0 - rho) * W
    covered = 0.0
    used = min(C, Ws)
    if Ws > 0:
        covered += rho * W * (used / Ws)
    c_left = C - used
    if Wp > 0 and c_left > 0:
        covered += Wp * min(1.0, c_left / Wp) ** theta
    return min(1.0, covered / W)


def weight_hit_rate(C: float, W_read: float) -> float:
    """Weights are read uniformly once per step: pinning C bytes covers C/W of the reads."""
    if W_read <= 0:
        return 1.0
    return min(1.0, max(C, 0.0) / W_read)
