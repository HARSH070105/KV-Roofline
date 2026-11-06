"""Area and energy models for the hardware knobs with literature-sourced constants.

Sourced constants:
  * SRAM Density: TSMC N5 / N4 high-density 6T SRAM bitcell (~0.021 µm² bitcell area).
    Macro density accounting for periphery, decoders, sense amps, and power mesh yields
    ~3.8 - 4.2 MB/mm² (standard reference: TSMC N5 ISSCC 2020 / IEDM 2019; CACTI 7.0 at 5nm).
    Default: 3.9 MB/mm².
  * SRAM Dynamic Energy: ~0.10 - 0.25 pJ/bit for large on-chip SRAM macros (CACTI 7.0 at 5nm,
    modeled as base access e0 + H-tree wire scaling e1 * sqrt(Capacity_MB)).
    Default: e0 = 0.10 pJ/bit, e1 = 0.02 pJ/bit.
  * SRAM Static Leakage: ~20 - 30 mW/MB at 85°C typical operating temperature (CACTI 7.0 N5).
    Default: 25.0 mW/MB.
  * HBM3 / HBM3e PHY Area: On-die D2D/PHY area per 1024-bit interface is ~12-14 mm²
    (Synopsys / Cadence HBM3 PHY IP in TSMC N5). Default: 14.0 mm² per stack.
  * HBM Package Overhead: Organic substrate / CoWoS interposer footprint per HBM stack
    is ~100-120 mm² (NVIDIA H100 substrate analysis / Hot Chips 34). Default: 110.0 mm² per stack.
    Package area is weighted relative to leading-edge compute die area (default weight: 0.5).
  * HBM Energy: JEDEC JESD238 HBM3 standard reports ~3.5 - 4.0 pJ/bit for HBM3 (6.4 Gbps)
    and ~3.2 - 3.7 pJ/bit for HBM3e (Hot Chips 35 / SK Hynix). Default: 3.9 pJ/bit (H100).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import math
from typing import Optional

from .hardware import Hardware


@dataclass(frozen=True)
class CostParams:
    # SRAM: macro density incl. periphery. TSMC N5/N4 high-density macro ~ 3.9 MB/mm^2.
    sram_mb_per_mm2: float = 3.9
    # SRAM read energy: pJ/bit = e0 + e1*sqrt(C_MB) (wire/H-tree growth, CACTI 7.0 at 5nm)
    sram_e0_pj_bit: float = 0.10
    sram_e1_pj_bit: float = 0.02
    sram_leak_mw_per_mb: float = 25.0     # leakage per MB at 85°C
    # HBM stack: on-die PHY area + package/interposer footprint (TSMC CoWoS / Synopsys PHY)
    hbm_phy_mm2_per_stack: float = 14.0
    hbm_pkg_mm2_per_stack: float = 110.0
    pkg_area_weight: float = 0.5          # cost-weight of 1 mm^2 interposer/package vs 1 mm^2 leading die
    # HBM energy vs pin speed: pJ/bit ~ pJ0 * scale^k (JEDEC HBM3 standard)
    hbm_speed_energy_exp: float = 0.5
    # Optional dequant hardware cost for compression knobs (0 = software only)
    dequant_area_mm2: float = 0.0

    @classmethod
    def from_json(cls, path: str) -> "CostParams":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_csv(cls, path: str) -> "CostParams":
        """Load key=value rows from CSV."""
        import pandas as pd
        df = pd.read_csv(path)
        params = {}
        for _, row in df.iterrows():
            k = str(row.iloc[0]).strip()
            v = float(row.iloc[1])
            if k in cls.__dataclass_fields__:
                params[k] = v
        return cls(**params)

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


DEFAULT_COST = CostParams()


def sram_pj_per_bit(capacity_mb: float, p: CostParams = DEFAULT_COST) -> float:
    return p.sram_e0_pj_bit + p.sram_e1_pj_bit * math.sqrt(max(capacity_mb, 0.0))


def hbm_pj_per_bit(hw: Hardware, p: CostParams = DEFAULT_COST) -> float:
    return hw.hbm_pj_per_bit * (hw.hbm_speed_scale ** p.hbm_speed_energy_exp)


def area_added(hw: Hardware, base: Hardware, p: CostParams = DEFAULT_COST,
               compression_dequant_hw: bool = False) -> dict:
    """Area added by `hw` relative to the `base` preset (per chip). Negative deltas allowed."""
    d_sram = (hw.sram_pin_mb - base.sram_pin_mb) / p.sram_mb_per_mm2
    d_stacks = hw.hbm_stacks - base.hbm_stacks
    d_phy = d_stacks * p.hbm_phy_mm2_per_stack
    d_pkg = d_stacks * p.hbm_pkg_mm2_per_stack
    d_deq = p.dequant_area_mm2 if compression_dequant_hw else 0.0
    die = d_sram + d_phy + d_deq
    total = die + p.pkg_area_weight * d_pkg
    return {"area_sram_mm2": d_sram, "area_hbm_phy_mm2": d_phy, "area_hbm_pkg_mm2": d_pkg,
            "area_added_mm2": total, "area_added_die_mm2": die}
