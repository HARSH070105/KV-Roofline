"""Core roofline engine: one decode step of an LLM on a (possibly multi-chip) accelerator.

    t_group = max( FLOPs_tensor/P_tc , FLOPs_cuda/P_cuda , Bytes_HBM/BW , Bytes_SRAM/BW_sram )
    t_step  = t_linear + t_attention + t_comm + t_fixed          (kernels serialised)
    tokens/s = B / t_step

Two op groups are modelled separately because they have different binding roofs:
  linear    : weight GEMV/GEMM, tensor-core FLOPs vs weight bytes
  attention : QK^T/PV on cached KV (+ softmax and dequantisation on the CUDA-core roof) vs KV bytes

Parallelism: tensor-parallel over hw.n_chips. GQA/MHA KV is sharded by KV head; the MLA latent is
REPLICATED on every chip (heads are split, the single latent is not), which is why MLA is capacity
hungry on multi-GPU. Set model-level notes accordingly when interpreting results.

Correctness-first: every function returns plain floats / dicts, no hidden state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import math

from .compression import Compression
from .cost import CostParams, DEFAULT_COST, area_added, hbm_pj_per_bit, sram_pj_per_bit
from .hardware import Hardware, MB
from .models import ModelConfig
from .sram import kv_hit_rate, weight_hit_rate

CAP_UTIL = 0.90          # usable fraction of HBM capacity
CAP_RESERVE = 1.0e9      # bytes reserved for activations / workspace per chip
ALLREDUCE_LATENCY = 5e-6  # seconds per all-reduce (ASSUMED)


@dataclass(frozen=True)
class SramPolicy:
    kv_frac: float = 1.0   # share of pinned SRAM given to KV (rest pins weights)
    theta: float = 1.0     # access-skew exponent (1 = uniform)
    rho: float = 0.0       # fraction of KV shared across the batch (prefix reuse)


@dataclass(frozen=True)
class Calibration:
    """Scalars fitted against measurements in week 6 (all 1/0 = ideal roofline)."""
    bw_eff: float = 1.0      # achievable fraction of peak HBM bandwidth
    tc_eff: float = 1.0
    cuda_eff: float = 1.0
    t_fixed_us: float = 0.0  # per-step fixed overhead (launch, sampling, ...)


# --------------------------------------------------------------------------------------
# capacity
# --------------------------------------------------------------------------------------
def _kv_div(model: ModelConfig, n: int) -> int:
    return 1 if model.attn == "mla" else n


def kv_stored_bytes_per_seq_chip(model: ModelConfig, hw: Hardware, S: int, comp: Compression) -> float:
    toks = comp.tokens_stored(S)
    avg, _ = comp.avg_bytes_per_elem(toks)
    return model.n_layers * model.kv_elems_per_token_layer() * toks * avg / _kv_div(model, hw.n_chips)


def max_batch(model: ModelConfig, hw: Hardware, S: int, comp: Compression) -> int:
    """Largest batch whose weights + KV fit in HBM (per chip, TP-sharded). 0 = does not fit."""
    w_chip = model.params_total * comp.w_bits / 8.0 / hw.n_chips
    room = hw.hbm_cap * CAP_UTIL - CAP_RESERVE - w_chip
    kv = kv_stored_bytes_per_seq_chip(model, hw, S, comp)
    if room <= 0 or kv <= 0:
        return 0
    return int(math.floor(room / kv))


def crossover_S(model: ModelConfig, B: int, comp: Compression = Compression()) -> float:
    """S* where KV traffic equals weight traffic (deck slide 4). Ignores eviction and the residual window.
       S* = weight_bytes_read(B) / (B * KV bytes per token)."""
    w_bytes = model.weight_params_read(B) * comp.w_bits / 8.0
    kv_tok = model.n_layers * model.kv_elems_per_token_layer() * comp.bytes_per_elem_quant()
    return w_bytes / (B * kv_tok)


# --------------------------------------------------------------------------------------
# main evaluation
# --------------------------------------------------------------------------------------
def evaluate(model: ModelConfig, hw: Hardware, S: int, B: int,
             comp: Compression = Compression(),
             sram: SramPolicy = SramPolicy(),
             calib: Calibration = Calibration(),
             cost: CostParams = DEFAULT_COST,
             base_hw: Optional[Hardware] = None,
             attn_unit: str = "tensor") -> dict:
    """Evaluate one decode step. attn_unit: 'tensor' (QK/PV on tensor cores) or 'cuda' (on CUDA cores)."""
    if attn_unit not in ("tensor", "cuda"):
        raise ValueError("attn_unit must be 'tensor' or 'cuda'")
    n = hw.n_chips
    L = model.n_layers
    e = model.kv_elems_per_token_layer()
    wb = comp.w_bits / 8.0
    kvd = _kv_div(model, n)
    fd = n                                 # attention FLOPs always split by heads
    deqd = _kv_div(model, n)

    # ---- per-sequence attention quantities (whole model, all layers) -------------------
    Sr = comp.tokens_read(S)
    avg_b, q_tok = comp.avg_bytes_per_elem(Sr)
    kv_read_seq = L * e * Sr * avg_b
    idx_seq = L * comp.index_bytes_per_seq_layer(S, model.n_kv_heads, model.head_dim) \
        if model.attn != "mla" else 0.0
    attn_flops_seq = L * model.attn_flops_per_token_layer() * Sr
    sm_flops_seq = L * model.n_q_heads * Sr * comp.softmax_flops
    deq_flops_seq = L * e * q_tok * comp.dequant_flops
    kv_write_seq = L * e * 2.0             # new token's KV, written at BF16 (into the residual window)

    # ---- per-chip totals for the batch ---------------------------------------------------
    kv_read = B * kv_read_seq / kvd
    idx = B * idx_seq / kvd
    kv_write = B * kv_write_seq / kvd
    attn_flops = B * attn_flops_seq / fd
    sm_flops = B * sm_flops_seq / fd
    deq_flops = B * deq_flops_seq / deqd

    w_read = model.weight_params_read(B) * wb / n
    lin_flops = 2.0 * B * model.active_params_per_token() / n

    # ---- SRAM ---------------------------------------------------------------------------
    C_total = hw.sram_pin_mb * MB
    C_kv = C_total * sram.kv_frac
    C_w = C_total - C_kv
    foot_scale = 1.0
    if comp.evict_policy == "quest" and Sr < S:       # selection is dynamic: pin against the stored cache
        foot_scale = comp.tokens_stored(S) / Sr
    h_kv = kv_hit_rate(C_kv, kv_read, B, sram.theta, sram.rho, foot_scale)
    h_w = weight_hit_rate(C_w, w_read)

    w_hbm, w_sram = w_read * (1 - h_w), w_read * h_w
    kv_hbm, kv_sram = kv_read * (1 - h_kv), kv_read * h_kv
    attn_hbm = kv_hbm + idx + kv_write

    # ---- time ---------------------------------------------------------------------------
    BW = hw.hbm_bw * calib.bw_eff
    Ptc = hw.p_tensor * calib.tc_eff
    Pc = hw.p_cuda * calib.cuda_eff

    lin_terms = {"tensor": lin_flops / Ptc, "hbm": w_hbm / BW, "sram": w_sram / hw.sram_bw}
    if attn_unit == "tensor":
        attn_terms = {"tensor": attn_flops / Ptc, "cuda": (sm_flops + deq_flops) / Pc}
    else:
        attn_terms = {"tensor": 0.0, "cuda": (attn_flops + sm_flops + deq_flops) / Pc}
    attn_terms["hbm"] = attn_hbm / BW
    attn_terms["sram"] = kv_sram / hw.sram_bw

    t_lin = max(lin_terms.values())
    t_attn = max(attn_terms.values())
    bind_lin = max(lin_terms, key=lin_terms.get)
    bind_attn = max(attn_terms, key=attn_terms.get)

    t_comm = 0.0
    if n > 1:
        per_ar = 2.0 * (n - 1) / n * B * model.hidden * 2.0 / hw.link_bw + ALLREDUCE_LATENCY
        t_comm = 2.0 * L * per_ar
    t_fixed = calib.t_fixed_us * 1e-6
    t_step = t_lin + t_attn + t_comm + t_fixed

    # ---- roofline coordinates --------------------------------------------------------------
    flops_chip = lin_flops + attn_flops + sm_flops + deq_flops
    hbm_chip = w_hbm + attn_hbm
    ai_step = flops_chip / hbm_chip if hbm_chip > 0 else float("inf")
    ai_attn = (attn_flops + sm_flops + deq_flops) / attn_hbm if attn_hbm > 0 else float("inf")
    ai_w = lin_flops / w_hbm if w_hbm > 0 else float("inf")
    attainable = min(Ptc, BW * ai_step)
    achieved = flops_chip / t_step

    if t_lin >= t_attn:
        dominant = f"linear:{bind_lin}"
    else:
        dominant = f"attn:{bind_attn}"

    # ---- energy (whole system) -------------------------------------------------------------
    e_hbm = n * hbm_chip * 8 * hbm_pj_per_bit(hw, cost) * 1e-12
    e_sram = n * (w_sram + kv_sram) * 8 * sram_pj_per_bit(hw.sram_pin_mb, cost) * 1e-12
    e_comp = n * ((lin_flops + attn_flops) * hw.tensor_pj_per_flop
                  + (sm_flops + deq_flops) * hw.cuda_pj_per_flop) * 1e-12
    e_static = n * (hw.static_w + hw.sram_pin_mb * cost.sram_leak_mw_per_mb * 1e-3) * t_step
    e_total = e_hbm + e_sram + e_comp + e_static

    # ---- capacity -------------------------------------------------------------------------
    mb = max_batch(model, hw, S, comp)
    kv_gb_total = B * kv_stored_bytes_per_seq_chip(model, hw, S, comp) * kvd / 1e9

    out = {
        "model": model.name, "hw": hw.name, "n_chips": n, "S": S, "B": B, "comp": comp.label(),
        "feasible": (B <= mb) and B >= 1, "max_batch": mb, "kv_stored_gb_total": kv_gb_total,
        # time / throughput
        "t_step_ms": t_step * 1e3, "t_lin_ms": t_lin * 1e3, "t_attn_ms": t_attn * 1e3, "t_comm_ms": t_comm * 1e3,
        "tpot_ms": t_step * 1e3, "tokens_per_s": B / t_step, "tokens_per_s_per_chip": B / t_step / n,
        "dominant": dominant, "bind_lin": bind_lin, "bind_attn": bind_attn,
        # roofline coordinates
        "ai_step": ai_step, "ai_attn": ai_attn, "ai_weights": ai_w,
        "ridge_tensor": hw.ridge_tensor, "ridge_cuda": hw.ridge_cuda,
        "ai_over_ridge": ai_step / hw.ridge_tensor,
        "attainable_tflops": attainable / 1e12, "achieved_tflops": achieved / 1e12,
        "pct_of_roof": achieved / attainable if attainable > 0 else float("nan"),
        # traffic
        "hbm_gb_per_step_chip": hbm_chip / 1e9, "w_hbm_gb": w_hbm / 1e9, "kv_hbm_gb": kv_hbm / 1e9,
        "hbm_bytes_per_token": n * hbm_chip / B,
        "kv_share_of_hbm": (kv_hbm + idx) / hbm_chip if hbm_chip > 0 else 0.0,
        "h_kv": h_kv, "h_w": h_w, "s_star": crossover_S(model, max(B, 1), comp),
        "metadata_overhead": comp.metadata_overhead(),
        # energy
        "energy_mj_per_token": e_total / B * 1e3, "tokens_per_joule": B / e_total,
        "e_hbm_frac": e_hbm / e_total, "e_sram_frac": e_sram / e_total,
        "e_comp_frac": e_comp / e_total, "e_static_frac": e_static / e_total,
        "avg_power_w": e_total / t_step,
        # cuda-core roof diagnostics
        "cuda_time_over_hbm_time": attn_terms["cuda"] / attn_terms["hbm"] if attn_terms["hbm"] > 0 else float("inf"),
    }
    if base_hw is not None:
        out.update(area_added(hw, base_hw, cost))
    return out


def resolve_batch(model: ModelConfig, hw: Hardware, S: int, comp: Compression,
                  mode: str, B_target: int) -> int:
    """Batch to evaluate. mode='fixed': B_target. mode='max': largest feasible batch up to B_target."""
    if mode == "fixed":
        return B_target
    if mode == "max":
        return max(0, min(B_target, max_batch(model, hw, S, comp)))
    raise ValueError("mode must be 'fixed' or 'max'")
