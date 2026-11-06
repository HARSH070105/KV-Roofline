"""Hardware presets and the three hardware knobs of the DSE.

Units (SI throughout, to match the deck): FLOP/s, bytes/s, bytes.
  989 TFLOP/s = 989e12 ;  3.35 TB/s = 3.35e12 ;  1 GB = 1e9 B ;  1 MB = 1e6 B

The three hardware knobs the project explores:
  1. sram_pin_mb     : on-chip capacity usable to *pin* KV (or weights)    -> raises hit rate h
  2. hbm_stacks      : number of HBM stacks (adds bandwidth AND capacity)  -> steeper memory roof
     hbm_speed_scale : pin-speed scale (adds bandwidth, costs pJ/bit)
  3. (KV compression lives in compression.py - it is a software/format knob)

!!! Every number below is from public datasheets / common literature values and
!!! MUST be re-verified before it goes into the report. Fields marked ASSUMED are
!!! modelling assumptions with no datasheet value (replace with CACTI / vendor data).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

TB = 1e12
GB = 1e9
MB = 1e6


@dataclass(frozen=True)
class Hardware:
    name: str
    p_tensor: float            # FLOP/s, dense BF16 tensor-core roof
    p_cuda: float              # FLOP/s, FP32 CUDA-core roof (softmax, dequant run here)
    hbm_stacks: int
    hbm_bw_per_stack: float    # B/s per stack at speed_scale = 1
    hbm_cap_per_stack: float   # bytes per stack
    sram_pin_mb: float         # MB usable for pinning (ASSUMED ~75% of L2 on NVIDIA parts)
    sram_bw: float             # B/s of the on-chip cache (approx L2 read BW)
    link_bw: float             # B/s per direction, chip-to-chip (NVLink / PCIe)
    die_mm2: float             # compute die area
    tdp_w: float
    static_w: float            # ASSUMED idle/leakage power per chip
    hbm_pj_per_bit: float      # ASSUMED ~3.5-4 pJ/bit for HBM3-class
    tensor_pj_per_flop: float  # ASSUMED
    cuda_pj_per_flop: float    # ASSUMED
    hbm_speed_scale: float = 1.0
    n_chips: int = 1
    notes: str = ""

    # ---- derived ------------------------------------------------------------
    @property
    def hbm_bw(self) -> float:
        return self.hbm_stacks * self.hbm_bw_per_stack * self.hbm_speed_scale

    @property
    def hbm_cap(self) -> float:
        return self.hbm_stacks * self.hbm_cap_per_stack

    @property
    def ridge_tensor(self) -> float:
        """FLOP/B where the tensor-core roof meets the memory roof."""
        return self.p_tensor / self.hbm_bw

    @property
    def ridge_cuda(self) -> float:
        """FLOP/B where the CUDA-core roof meets the memory roof (~20 on H100)."""
        return self.p_cuda / self.hbm_bw

    def with_(self, **kw) -> "Hardware":
        return replace(self, **kw)


PRESETS: dict[str, Hardware] = {
    # NVIDIA H100 SXM5 80GB: 989 TFLOP/s BF16 dense, 67 TFLOP/s FP32, 3.35 TB/s, 50 MB L2 (NVIDIA Hopper Architecture Whitepaper)
    "h100": Hardware(
        "H100-SXM-80GB", 989 * TB, 67 * TB, 5, 0.67 * TB, 16 * GB, 37.5, 7.0 * TB, 450 * GB,
        814.0, 700.0, 100.0, 3.9, 0.35, 1.5,
        notes="H100 SXM5: 5 active HBM3 stacks (16GB, 670 GB/s each = 3.35 TB/s). 50MB L2 with 75% pin set-aside (37.5MB).",
    ),
    # H200: same compute, 6 x HBM3e 24GB = 141 GB, 4.8 TB/s (NVIDIA H200 Datasheet)
    "h200": Hardware(
        "H200-SXM-141GB", 989 * TB, 67 * TB, 6, 0.8 * TB, 23.5 * GB, 37.5, 7.0 * TB, 450 * GB,
        814.0, 700.0, 100.0, 3.5, 0.35, 1.5,
        notes="H200 SXM: 6 active HBM3e stacks (23.5GB effective, 800 GB/s each = 4.8 TB/s). 3.5 pJ/bit HBM3e.",
    ),
    # A100 SXM 80GB: 312 TFLOP/s BF16, 19.5 FP32, 2.039 TB/s, 40 MB L2 (NVIDIA Ampere Architecture Whitepaper)
    "a100-80": Hardware(
        "A100-SXM-80GB", 312 * TB, 19.5 * TB, 5, 0.4078 * TB, 16 * GB, 30.0, 4.5 * TB, 300 * GB,
        826.0, 400.0, 60.0, 3.9, 0.5, 2.0,
        notes="A100 SXM 80GB: 5 active HBM2e stacks (16GB, 407.8 GB/s each = 2.039 TB/s). 40MB L2 (30MB pin limit).",
    ),
    # A100 40GB: 1.555 TB/s, 5 stacks x 8 GB
    "a100-40": Hardware(
        "A100-SXM-40GB", 312 * TB, 19.5 * TB, 5, 0.311 * TB, 8 * GB, 30.0, 4.5 * TB, 300 * GB,
        826.0, 400.0, 60.0, 3.9, 0.5, 2.0,
        notes="A100 SXM 40GB: 5 active HBM2 stacks (8GB, 311 GB/s each = 1.555 TB/s). 40MB L2.",
    ),
    # B200: NVIDIA Blackwell B200 Architecture Whitepaper: ~2.25 PFLOP/s dense BF16, 8.0 TB/s HBM3e, 192 GB
    "b200": Hardware(
        "B200-approx", 2250 * TB, 80 * TB, 8, 1.0 * TB, 24 * GB, 96.0, 12.0 * TB, 900 * GB,
        1600.0, 1000.0, 150.0, 3.5, 0.25, 1.2,
        notes="Blackwell B200: dual-die package (1600 mm²), 8 HBM3e stacks (24GB, 1.0 TB/s each = 8 TB/s), 128MB L2 (96MB pin).",
    ),
    # L4 24GB: NVIDIA Ada Lovelace L4 Datasheet (GDDR6 300 GB/s)
    "l4": Hardware(
        "L4-24GB", 121 * TB, 30.3 * TB, 1, 0.3 * TB, 24 * GB, 36.0, 2.0 * TB, 32 * GB,
        295.0, 72.0, 15.0, 7.0, 0.5, 1.5,
        notes="L4: AD104 die, 24GB GDDR6 (300 GB/s). L2 48MB (75% pin = 36MB). 7.0 pJ/bit GDDR6.",
    ),
    # RTX 4090 24GB: AD102 die, 24GB GDDR6X @ 1.008 TB/s
    "rtx4090": Hardware(
        "RTX4090-24GB", 165 * TB, 82.6 * TB, 1, 1.008 * TB, 24 * GB, 54.0, 5.0 * TB, 32 * GB,
        608.0, 450.0, 40.0, 7.5, 0.4, 1.2,
        notes="RTX 4090: AD102 die, 24GB GDDR6X (1.008 TB/s). L2 72MB (75% pin = 54MB). 7.5 pJ/bit GDDR6X.",
    ),
}


def get_hw(name: str, **overrides) -> Hardware:
    if name not in PRESETS:
        raise KeyError(f"unknown hardware '{name}'. Options: {sorted(PRESETS)}")
    hw = PRESETS[name]
    return hw.with_(**overrides) if overrides else hw
