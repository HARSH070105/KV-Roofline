"""Layer-wise roofline analysis reproducing and extending LLM-Viewer (Yuan et al., arXiv:2402.16363).

Computes per-layer arithmetic intensity, FLOPs, and memory traffic for:
  1. Attention Projections (Q, K, V, O projections)
  2. Attention Core (QK^T, Softmax, PV context matmul, KV cache read/write)
  3. Feed-Forward Network / MLP (Gate, Up, Down projections; dense or routed MoE)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import pandas as pd

from .compression import Compression
from .hardware import Hardware
from .models import ModelConfig


@dataclass(frozen=True)
class LayerMetrics:
    layer_idx: int
    layer_type: str            # "dense" or "moe"
    # Projections
    qkv_flops: float
    qkv_weight_bytes: float
    o_proj_flops: float
    o_proj_weight_bytes: float
    # Attention Core
    attn_flops: float          # QK^T + PV
    softmax_flops: float
    dequant_flops: float
    kv_read_bytes: float
    kv_write_bytes: float
    # MLP / MoE
    mlp_flops: float
    mlp_weight_bytes: float
    # Totals
    total_flops: float
    total_weight_bytes: float
    total_kv_bytes: float
    total_bytes: float
    ai_layer: float
    ai_attn_core: float
    ai_mlp: float
    kv_fraction: float


def analyze_layerwise(model: ModelConfig, S: int, B: int,
                      comp: Compression = Compression(),
                      bytes_per_weight: float = 2.0) -> list[LayerMetrics]:
    """Calculate per-layer breakdown for model at context S and batch B."""
    H = model.hidden
    d = model.head_dim
    Hq = model.n_q_heads
    Hkv = model.n_kv_heads
    Sr = comp.tokens_read(S)
    avg_b, q_tok = comp.avg_bytes_per_elem(Sr)
    kv_elems_layer = model.kv_elems_per_token_layer()

    # Attention core FLOPs & bytes per layer
    attn_flops_layer = B * model.attn_flops_per_token_layer() * Sr
    softmax_flops_layer = B * Hq * Sr * comp.softmax_flops
    dequant_flops_layer = B * kv_elems_layer * q_tok * comp.dequant_flops
    kv_read_bytes_layer = B * kv_elems_layer * Sr * avg_b
    kv_write_bytes_layer = B * kv_elems_layer * 2.0

    # MLA vs Standard GQA/MHA projections
    if model.attn == "mla":
        # DeepSeek MLA: down/up projections for KV and Q
        d_c = model.kv_lora_rank
        d_r = model.rope_dim
        # W_DKV: H -> d_c; W_UK: d_c -> Hq*d; W_UV: d_c -> Hq*d; W_KRope: H -> d_r
        # W_DQ: H -> q_lora_rank; W_UQ: q_lora_rank -> Hq*d
        qkv_params = H * (d_c + d_r) + 2 * d_c * Hq * d + H * (d_c + d_r)
        o_params = Hq * d * H
    else:
        # Standard GQA / MHA
        q_params = H * (Hq * d)
        k_params = H * (Hkv * d)
        v_params = H * (Hkv * d)
        qkv_params = q_params + k_params + v_params
        o_params = (Hq * d) * H

    qkv_flops = 2.0 * B * qkv_params
    qkv_weight_bytes = qkv_params * bytes_per_weight
    o_flops = 2.0 * B * o_params
    o_weight_bytes = o_params * bytes_per_weight

    # MLP parameters per layer
    # If intermediate is not explicitly stored, standard LLaMA intermediate is ~ (8/3)*H rounded to 256
    # Let's derive dense FFN params from total params minus attention and embed
    n_moe_layers = model.moe.n_moe_layers if model.moe else 0
    non_routed_total = model._nonrouted_params()
    dense_layers = model.n_layers - n_moe_layers
    attn_params_all = model.n_layers * (qkv_params + o_params)
    dense_mlp_params_per_layer = max(0.0, (non_routed_total - attn_params_all) / max(dense_layers, 1))

    layers = []
    for l_idx in range(model.n_layers):
        is_moe = model.moe is not None and l_idx >= (model.n_layers - model.moe.n_moe_layers)
        if is_moe:
            m = model.moe
            # Routed experts touched
            touched = m.n_routed * (1.0 - (1.0 - m.top_k / m.n_routed) ** max(B, 1))
            mlp_params_read = touched * m.routed_expert_params
            mlp_active_params = m.top_k * m.routed_expert_params
            layer_type = f"moe(top_{m.top_k}/{m.n_routed})"
        else:
            mlp_params_read = dense_mlp_params_per_layer
            mlp_active_params = dense_mlp_params_per_layer
            layer_type = "dense"

        mlp_flops = 2.0 * B * mlp_active_params
        mlp_weight_bytes = mlp_params_read * bytes_per_weight

        tot_flops = qkv_flops + o_flops + attn_flops_layer + softmax_flops_layer + dequant_flops_layer + mlp_flops
        tot_weights = qkv_weight_bytes + o_weight_bytes + mlp_weight_bytes
        tot_kv = kv_read_bytes_layer + kv_write_bytes_layer
        tot_bytes = tot_weights + tot_kv

        ai_layer = tot_flops / tot_bytes if tot_bytes > 0 else 0.0
        ai_attn_core = (attn_flops_layer + softmax_flops_layer + dequant_flops_layer) / tot_kv if tot_kv > 0 else 0.0
        ai_mlp = mlp_flops / mlp_weight_bytes if mlp_weight_bytes > 0 else 0.0
        kv_frac = tot_kv / tot_bytes if tot_bytes > 0 else 0.0

        layers.append(LayerMetrics(
            layer_idx=l_idx,
            layer_type=layer_type,
            qkv_flops=qkv_flops,
            qkv_weight_bytes=qkv_weight_bytes,
            o_proj_flops=o_flops,
            o_proj_weight_bytes=o_weight_bytes,
            attn_flops=attn_flops_layer,
            softmax_flops=softmax_flops_layer,
            dequant_flops=dequant_flops_layer,
            kv_read_bytes=kv_read_bytes_layer,
            kv_write_bytes=kv_write_bytes_layer,
            mlp_flops=mlp_flops,
            mlp_weight_bytes=mlp_weight_bytes,
            total_flops=tot_flops,
            total_weight_bytes=tot_weights,
            total_kv_bytes=tot_kv,
            total_bytes=tot_bytes,
            ai_layer=ai_layer,
            ai_attn_core=ai_attn_core,
            ai_mlp=ai_mlp,
            kv_fraction=kv_frac,
        ))

    return layers


def layerwise_dataframe(model: ModelConfig, S: int, B: int,
                        comp: Compression = Compression(),
                        bytes_per_weight: float = 2.0) -> pd.DataFrame:
    layers = analyze_layerwise(model, S, B, comp, bytes_per_weight)
    return pd.DataFrame([l.__dict__ for l in layers])
