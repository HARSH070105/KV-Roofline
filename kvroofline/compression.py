"""KV-cache compression knobs and their *overhead terms* (the novelty in Gap 2 of the deck).

Three overheads that plain "bits/16" accounting misses:
  1. Metadata: per-group scale+zero (2 x FP16 = 4 B per group) inflates stored/read bytes.
        bytes/elem = bits/8 + 4/group        (4-bit, g=32 -> +25%, matches Interface-Aware 2609.05764)
  2. Dequantization: c_deq FLOPs per KV element on the CUDA-core roof (unpack, cvt, fma scale+zero).
  3. Eviction indexing: Quest-style page min/max metadata must be read for ALL S tokens.
Also a full-precision residual window (KIVI keeps the newest R tokens in FP16).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

SCALE_ZERO_BYTES = 4.0  # FP16 scale + FP16 zero-point per group


@dataclass(frozen=True)
class Compression:
    kv_bits: int = 16                 # 16 = BF16 baseline, 8, 4, 2
    group: int = 64                   # quantization group size (0 = no metadata, e.g. FP8 per-tensor)
    residual: int = 0                 # newest tokens kept in BF16 (KIVI R)
    w_bits: int = 16                  # weight precision (quality-affecting, optional knob)
    evict_policy: str = "none"        # "none" | "streaming" | "quest"
    evict_budget: Optional[int] = None  # tokens attended (streaming: also stored)
    quest_page: int = 16              # page size for Quest metadata
    dequant_flops: float = 6.0        # FLOPs per dequantized element (ASSUMED; sweep 2..10)
    softmax_flops: float = 5.0        # FLOPs per attention score (ASSUMED)

    # ------------------------------------------------------------------
    def label(self) -> str:
        parts = [f"KV{self.kv_bits}" + (f"g{self.group}" if self.kv_bits < 16 and self.group else "")]
        if self.residual and self.kv_bits < 16:
            parts.append(f"R{self.residual}")
        if self.evict_policy != "none":
            parts.append(f"{self.evict_policy}{self.evict_budget}")
        if self.w_bits != 16:
            parts.append(f"W{self.w_bits}")
        return "+".join(parts)

    @property
    def quantized(self) -> bool:
        return self.kv_bits < 16

    def bytes_per_elem_quant(self) -> float:
        """Bytes per cached scalar for the quantized part (incl. metadata)."""
        if not self.quantized:
            return 2.0
        meta = SCALE_ZERO_BYTES / self.group if self.group else 0.0
        return self.kv_bits / 8.0 + meta

    def metadata_overhead(self) -> float:
        """Metadata as a fraction of payload bytes (deck: ~25% for KIVI-style)."""
        if not self.quantized or not self.group:
            return 0.0
        return (SCALE_ZERO_BYTES / self.group) / (self.kv_bits / 8.0)

    # ---- token accounting ---------------------------------------------------
    def tokens_read(self, S: int) -> float:
        """Tokens whose KV is actually attended to per decode step."""
        if self.evict_policy == "none" or self.evict_budget is None:
            return float(S)
        return float(min(S, self.evict_budget))

    def tokens_stored(self, S: int) -> float:
        """Tokens kept in memory (capacity). Streaming evicts; Quest keeps everything."""
        if self.evict_policy == "streaming" and self.evict_budget is not None:
            return float(min(S, self.evict_budget))
        return float(S)

    def avg_bytes_per_elem(self, n_tokens: float) -> tuple[float, float]:
        """(average bytes/elem, number of tokens that are quantized) over n_tokens cached tokens."""
        if n_tokens <= 0:
            return 2.0, 0.0
        if not self.quantized:
            return 2.0, 0.0
        r = min(n_tokens, float(self.residual))
        q = n_tokens - r
        avg = (q * self.bytes_per_elem_quant() + r * 2.0) / n_tokens
        return avg, q

    def index_bytes_per_seq_layer(self, S: int, n_kv_heads: int, head_dim: int) -> float:
        """Quest page metadata (FP16 min+max of K per page) that is read for all S tokens."""
        if self.evict_policy != "quest" or self.evict_budget is None or S <= self.evict_budget:
            return 0.0
        pages = S / self.quest_page
        return pages * n_kv_heads * 2 * head_dim * 2.0
