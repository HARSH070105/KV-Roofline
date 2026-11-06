"""Model configurations (from public HuggingFace config.json files - re-confirm in week 1).

Only what the roofline needs is stored:
  * KV cache layout  (per token, per layer: number of cached elements)
  * attention FLOPs per cached token per layer (decode, one query token)
  * weight bytes read per decode step (dense, or MoE with expected-expert-touch model)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class MoE:
    n_routed: int
    top_k: int
    n_moe_layers: int
    routed_expert_params: float  # params in ONE routed expert (gate+up+down)


@dataclass(frozen=True)
class ModelConfig:
    name: str
    attn: str                    # "mha" | "gqa" | "mla"
    n_layers: int
    hidden: int
    n_q_heads: int
    n_kv_heads: int              # = n_q_heads for MHA; unused for MLA
    head_dim: int
    params_total: float          # all stored parameters (incl. embeddings, all experts)
    vocab: int
    # MLA only
    kv_lora_rank: int = 0
    rope_dim: int = 0
    moe: Optional[MoE] = None
    role: str = ""

    # ---- KV layout ----------------------------------------------------------
    @property
    def g(self) -> int:
        """GQA group size Hq/Hkv (for MLA: all heads share one latent)."""
        if self.attn == "mla":
            return self.n_q_heads
        return self.n_q_heads // self.n_kv_heads

    def kv_elems_per_token_layer(self) -> int:
        """Cached scalars per token per layer."""
        if self.attn == "mla":
            return self.kv_lora_rank + self.rope_dim
        return 2 * self.n_kv_heads * self.head_dim

    def kv_bytes_per_token(self, bytes_per_elem: float = 2.0) -> float:
        """Whole-model KV bytes per token (BF16 -> 2 bytes). Deck: Llama-3-70B = 320 KiB."""
        return self.n_layers * self.kv_elems_per_token_layer() * bytes_per_elem

    def attn_flops_per_token_layer(self) -> float:
        """FLOPs to attend ONE query over ONE cached token in ONE layer (QK^T + PV).

        MHA/GQA: 2*d (QK) + 2*d (PV) per query head  -> 4*Hq*d.
        MLA (weight-absorbed decode): QK over (d_c + d_r), PV over d_c, per head
                                      -> 2*Hq*(2*d_c + d_r).
        """
        if self.attn == "mla":
            return 2.0 * self.n_q_heads * (2 * self.kv_lora_rank + self.rope_dim)
        return 4.0 * self.n_q_heads * self.head_dim

    def attn_ai(self, bytes_per_elem: float = 2.0) -> float:
        """Attention arithmetic intensity. GQA: 2g/b (deck slide 3)."""
        return self.attn_flops_per_token_layer() / (self.kv_elems_per_token_layer() * bytes_per_elem)

    # ---- weights ------------------------------------------------------------
    @property
    def embed_params(self) -> float:
        return float(self.vocab * self.hidden)

    def _nonrouted_params(self) -> float:
        """Params that are always touched for every token (excluding the input embedding table)."""
        p = self.params_total - self.embed_params
        if self.moe:
            p -= self.moe.n_moe_layers * self.moe.n_routed * self.moe.routed_expert_params
        return p

    def active_params_per_token(self) -> float:
        """Params that contribute FLOPs for one token (2 FLOP per param)."""
        p = self._nonrouted_params()
        if self.moe:
            p += self.moe.n_moe_layers * self.moe.top_k * self.moe.routed_expert_params
        return p

    def weight_params_read(self, batch: int) -> float:
        """Params fetched from memory in one decode step with `batch` sequences.

        Dense: every weight once per step (amortized over the batch).
        MoE:   expected number of distinct routed experts touched per layer, assuming
               uniform independent routing:  E * (1 - (1 - k/E)^B).
        """
        p = self._nonrouted_params()
        if self.moe:
            m = self.moe
            touched = m.n_routed * (1.0 - (1.0 - m.top_k / m.n_routed) ** max(batch, 1))
            p += m.n_moe_layers * touched * m.routed_expert_params
        return p


def _moe_dsv2lite() -> MoE:
    return MoE(n_routed=64, top_k=6, n_moe_layers=26, routed_expert_params=3 * 2048 * 1408)


def _moe_dsv2() -> MoE:
    return MoE(n_routed=160, top_k=6, n_moe_layers=59, routed_expert_params=3 * 5120 * 1536)


MODELS: dict[str, ModelConfig] = {m.name: m for m in [
    ModelConfig("llama2-7b", "mha", 32, 4096, 32, 32, 128, 6.74e9, 32000,
                role="MHA baseline (KIVI, KVQuant)"),
    ModelConfig("llama3.1-8b", "gqa", 32, 4096, 32, 8, 128, 8.03e9, 128256,
                role="Main measured model, 128k context"),
    ModelConfig("qwen2.5-7b", "gqa", 28, 3584, 28, 4, 128, 7.62e9, 152064,
                role="Quantization-sensitive second model"),
    ModelConfig("deepseek-v2-lite", "mla", 27, 2048, 16, 16, 128, 15.7e9, 102400,
                kv_lora_rank=512, rope_dim=64, moe=_moe_dsv2lite(),
                role="Small MLA model, measured"),
    ModelConfig("llama3-70b", "gqa", 80, 8192, 64, 8, 128, 70.6e9, 128256,
                role="Analytical: multi-GPU, capacity-bound"),
    ModelConfig("deepseek-v2", "mla", 60, 5120, 128, 128, 128, 236e9, 102400,
                kv_lora_rank=512, rope_dim=64, moe=_moe_dsv2(),
                role="Analytical: near-compute-bound MLA"),
]}


def get_model(name: str) -> ModelConfig:
    if name not in MODELS:
        raise KeyError(f"unknown model '{name}'. Options: {sorted(MODELS)}")
    return MODELS[name]
