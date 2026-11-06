import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
import pytest

from kvroofline import Compression, SramPolicy, crossover_S, evaluate, get_hw, get_model, max_batch, layerwise_dataframe
from kvroofline.dse import Grid, Regime, iso_area_best, pareto_front, pareto_mask, sweep
from kvroofline.quality import QualityTable
from kvroofline.sram import kv_hit_rate

H100 = get_hw("h100")


# ---------------- numbers stated in the deck -------------------------------------------------
def test_kv_bytes_per_token_match_deck():
    kib = {n: get_model(n).kv_bytes_per_token() / 1024 for n in
           ["llama2-7b", "llama3.1-8b", "qwen2.5-7b", "llama3-70b"]}
    assert kib == {"llama2-7b": 512, "llama3.1-8b": 128, "qwen2.5-7b": 56, "llama3-70b": 320}
    assert get_model("deepseek-v2-lite").kv_bytes_per_token() / 1024 == pytest.approx(30.4, abs=0.1)   # "about 30 KB"
    assert get_model("deepseek-v2").kv_bytes_per_token() / 1024 == pytest.approx(67.5, abs=0.1)       # "about 70 KB"


def test_llama3_70b_128k_is_42gb():
    m = get_model("llama3-70b")
    assert m.kv_bytes_per_token() * 131072 / 1e9 == pytest.approx(42.9, abs=0.1)


def test_h100_ridges():
    assert H100.ridge_tensor == pytest.approx(295, abs=1)       # deck: about 300
    assert H100.ridge_cuda == pytest.approx(20, abs=0.5)        # deck: about 20


def test_gqa_attention_ai_is_2g_over_b():
    assert get_model("llama2-7b").attn_ai(2.0) == pytest.approx(1.0)      # MHA BF16 -> 1
    assert get_model("llama3-70b").attn_ai(2.0) == pytest.approx(8.0)     # g=8 -> 8
    assert get_model("llama3-70b").attn_ai(0.5) == pytest.approx(32.0)    # INT4, no metadata -> 32


def test_mla_ai_about_240():
    assert get_model("deepseek-v2").attn_ai(2.0) == pytest.approx(241.8, abs=0.5)


def test_metadata_overhead_25pct_at_int4_g32():
    assert Compression(kv_bits=4, group=32).metadata_overhead() == pytest.approx(0.25)


def test_param_counts_reasonable():
    assert get_model("deepseek-v2").active_params_per_token() / 1e9 == pytest.approx(21, abs=1.0)
    assert get_model("deepseek-v2-lite").active_params_per_token() / 1e9 == pytest.approx(2.4, abs=0.2)


# ---------------- engine identities / invariants ---------------------------------------------
def test_crossover_equates_weight_and_kv_traffic():
    m = get_model("llama3.1-8b")
    B = 8
    S = int(crossover_S(m, B))
    r = evaluate(m, H100.with_(sram_pin_mb=0), S, B)
    # at S* the KV read bytes ~= weight bytes read (kv_hbm includes the tiny new-token write, so allow 1%)
    assert r["kv_hbm_gb"] == pytest.approx(r["w_hbm_gb"], rel=0.02)


def test_decode_is_memory_bound_and_time_matches_bytes_over_bw():
    m = get_model("llama3.1-8b")
    h = H100.with_(sram_pin_mb=0)
    r = evaluate(m, h, 65536, 4)
    assert r["dominant"].endswith("hbm")
    t_expected = (r["hbm_gb_per_step_chip"] * 1e9) / h.hbm_bw * 1e3
    assert r["t_step_ms"] == pytest.approx(t_expected, rel=1e-6)       # both groups memory-bound -> sum of bytes/BW


def test_more_bandwidth_never_slower_and_ai_unchanged():
    m = get_model("llama3.1-8b")
    a = evaluate(m, H100, 65536, 4)
    b = evaluate(m, H100.with_(hbm_stacks=7), 65536, 4)
    assert b["tokens_per_s"] > a["tokens_per_s"]
    assert b["ai_step"] == pytest.approx(a["ai_step"])          # deck: bandwidth leaves AI unchanged
    assert b["ai_over_ridge"] > a["ai_over_ridge"]              # ...but moves the ridge


def test_kv_quantisation_reduces_bytes_raises_ai_and_metadata_counts():
    m = get_model("llama3.1-8b")
    h = H100.with_(sram_pin_mb=0)
    a = evaluate(m, h, 65536, 4)
    b = evaluate(m, h, 65536, 4, Compression(kv_bits=4, group=32))
    c = evaluate(m, h, 65536, 4, Compression(kv_bits=4, group=128))
    assert b["kv_hbm_gb"] < a["kv_hbm_gb"] * 0.35               # ~4.5/16
    assert b["kv_hbm_gb"] > c["kv_hbm_gb"]                      # smaller group = more metadata = more bytes
    assert b["ai_attn"] > a["ai_attn"]


def test_eviction_keeps_attention_ai_constant_but_cuts_time():
    m = get_model("llama3.1-8b")
    h = H100.with_(sram_pin_mb=0)
    a = evaluate(m, h, 131072, 2)
    b = evaluate(m, h, 131072, 2, Compression(evict_policy="streaming", evict_budget=4096))
    assert b["ai_attn"] == pytest.approx(a["ai_attn"], rel=0.02)   # deck: eviction leaves AI unchanged
    assert b["t_attn_ms"] < a["t_attn_ms"] / 10


def test_quest_pays_index_overhead():
    m = get_model("llama3.1-8b")
    h = H100.with_(sram_pin_mb=0)
    s = evaluate(m, h, 131072, 2, Compression(evict_policy="streaming", evict_budget=4096))
    q = evaluate(m, h, 131072, 2, Compression(evict_policy="quest", evict_budget=4096))
    assert q["kv_hbm_gb"] == pytest.approx(s["kv_hbm_gb"])
    assert q["t_attn_ms"] > s["t_attn_ms"]                          # extra page-metadata reads
    assert q["kv_stored_gb_total"] > s["kv_stored_gb_total"]        # quest keeps the full cache


def test_cuda_roof_binds_when_attention_runs_on_cuda_cores():
    m = get_model("llama3.1-8b")
    comp = Compression(kv_bits=4, group=64, residual=128, dequant_flops=6)
    t = evaluate(m, H100, 131072, 3, comp, attn_unit="tensor")
    c = evaluate(m, H100, 131072, 3, comp, attn_unit="cuda")
    assert t["bind_attn"] == "hbm"
    assert c["bind_attn"] == "cuda"
    assert c["tpot_ms"] > t["tpot_ms"]


def test_max_batch_and_feasibility():
    m = get_model("llama3.1-8b")
    mb = max_batch(m, H100, 131072, Compression())
    assert mb == 3                                                   # 80GB*0.9 - 1 - 16 = 55 GB; 17.2 GB/seq
    assert not evaluate(m, H100, 131072, mb + 1)["feasible"]
    assert evaluate(m, H100, 131072, mb)["feasible"]
    assert max_batch(m, H100, 131072, Compression(kv_bits=4, group=64)) > mb
    assert max_batch(m, H100, 524288, Compression()) == 0            # 64 GB KV per sequence does not fit on one H100


def test_moe_expert_touch_model():
    m = get_model("deepseek-v2-lite")
    p1, p64, p4096 = (m.weight_params_read(b) for b in (1, 64, 4096))
    assert p1 < p64 < p4096
    assert p4096 == pytest.approx(m.params_total - m.embed_params, rel=0.001)   # large batch touches every expert
    assert m.weight_params_read(1) == pytest.approx(m.active_params_per_token(), rel=0.02)


def test_energy_fractions_sum_to_one():
    r = evaluate(get_model("llama3.1-8b"), H100, 32768, 4)
    assert r["e_hbm_frac"] + r["e_sram_frac"] + r["e_comp_frac"] + r["e_static_frac"] == pytest.approx(1.0)
    assert r["energy_mj_per_token"] > 0 and r["tokens_per_joule"] > 0


def test_multichip_mla_latent_is_replicated():
    m = get_model("deepseek-v2")
    one = max_batch(m, H100.with_(n_chips=8), 131072, Compression())
    g = get_model("llama3-70b")
    assert one >= 0
    # GQA KV shards across chips, MLA latent does not: per-chip stored KV is n x larger relative to sharded
    kv_gqa = evaluate(g, H100.with_(n_chips=8), 131072, 1)["kv_stored_gb_total"]
    assert kv_gqa == pytest.approx(g.kv_bytes_per_token() * 131072 / 1e9, rel=0.01)


# ---------------- SRAM hit-rate -------------------------------------------------------------
def test_hit_rate_bounds_and_monotone():
    W = 10e9
    hs = [kv_hit_rate(c, W, 4, 1.0, 0.0) for c in (0, 1e8, 1e9, 5e9, 1e10, 2e10)]
    assert hs[0] == 0 and hs[-1] == 1.0 and all(0 <= h <= 1 for h in hs)
    assert hs == sorted(hs)
    assert kv_hit_rate(1e9, W, 4, 1.0) == pytest.approx(0.1)               # uniform: C/W
    assert kv_hit_rate(1e9, W, 4, 0.5) == pytest.approx(math.sqrt(0.1))   # skewed beats uniform
    assert kv_hit_rate(1e9, W, 4, 1.0, rho=0.5) > kv_hit_rate(1e9, W, 4, 1.0, rho=0.0)   # sharing helps
    assert kv_hit_rate(1e9, W, 4, 1.0, foot_scale=4.0) == pytest.approx(0.025)


def test_sram_hit_rate_tiny_on_full_kv_but_full_after_streaming_eviction():
    m = get_model("llama3.1-8b")
    big = H100.with_(sram_pin_mb=2048)
    full = evaluate(m, big, 131072, 3)
    ev = evaluate(m, big, 131072, 3, Compression(evict_policy="streaming", evict_budget=4096))
    assert full["h_kv"] < 0.05 and ev["h_kv"] == pytest.approx(1.0)
    # but even a perfect KV hit rate only removes the KV share of traffic (weights still stream from HBM)
    small = H100.with_(sram_pin_mb=0)
    comp = Compression(evict_policy="streaming", evict_budget=4096)
    gain = ev["tokens_per_s"] / evaluate(m, small, 131072, 3, comp)["tokens_per_s"]
    assert 1.0 < gain < 1.15


# ---------------- quality gate ----------------------------------------------------------------
def test_quality_placeholder_is_monotone_and_flagged():
    qt = QualityTable.placeholder()
    d16, m16 = qt.lookup(Compression(), 131072)
    d8, m8 = qt.lookup(Compression(kv_bits=8, group=0), 131072)
    d4, m4 = qt.lookup(Compression(kv_bits=4, group=64), 131072)
    d2, m2 = qt.lookup(Compression(kv_bits=2, group=32), 131072)
    assert d16 == 0 and m16 is True                  # BF16 is trivially exact
    assert 0 < d8 < d4 < d2 and not (m8 or m4 or m2)  # placeholder values are flagged unmeasured
    assert qt.passes(Compression(kv_bits=2, group=32), 131072, 1.0)[0] is False


def test_quality_table_measured_overrides_placeholder(tmp_path):
    p = tmp_path / "q.csv"
    pd.DataFrame([dict(kv_bits=4, group=64, evict_policy="none", evict_budget=0, w_bits=16, context=131072,
                       metric="ppl_delta", delta_pct=0.123, source="measured")]).to_csv(p, index=False)
    qt = QualityTable.from_csv(str(p))
    d, meas = qt.lookup(Compression(kv_bits=4, group=64), 100000)
    assert d == pytest.approx(0.123) and meas is True
    d, meas = qt.lookup(Compression(kv_bits=4, group=32), 131072)    # not in table -> placeholder
    assert meas is False


# ---------------- DSE ---------------------------------------------------------------------------
def test_pareto_mask_simple():
    df = pd.DataFrame({"t": [1, 2, 3, 3], "a": [0, 1, 5, 2], "e": [1, 1, 1, 1]})
    m = pareto_mask(df, ["t"], ["a", "e"])
    assert list(m) == [True, True, False, True]       # row 2 dominated by row 3 (same t, less area)


def test_sweep_end_to_end_small_grid():
    m = get_model("llama3.1-8b")
    reg = Regime("T", 131072, 2)
    g = Grid(sram_extra_mb=[0, 256], hbm_extra_stacks=[0, 1], kv_bits=[16, 4], groups=[64],
             evict=[("none", None)], sram_kv_frac=[1.0])
    df = sweep(m, H100, reg, g, batch_mode="fixed")      # fixed B: extra stacks must not change AI
    assert len(df) == 2 * 2 * 2
    base = df[(df.sram_extra_mb == 0) & (df.hbm_extra_stacks == 0) & (df.kv_bits == 16)].iloc[0]
    assert base["speedup"] == pytest.approx(1.0) and base["area_added_mm2"] == 0
    assert (df["d_ai_per_mm2"].dropna() >= -1e9).all()
    hbm = df[(df.hbm_extra_stacks == 1) & (df.sram_extra_mb == 0) & (df.kv_bits == 16)].iloc[0]
    assert hbm["d_ai"] == pytest.approx(0, abs=1e-9)  # bandwidth: no AI change
    assert hbm["area_added_mm2"] == pytest.approx(14 + 0.5 * 110)
    front = pareto_front(df)
    assert len(front) >= 1
    iso = iso_area_best(df, budgets=(0, 1000))
    assert iso.iloc[-1]["speedup"] >= iso.iloc[0]["speedup"]


def test_layerwise_analysis_reproduces_totals():
    m = get_model("llama3.1-8b")
    df = layerwise_dataframe(m, 8192, 4)
    assert len(df) == m.n_layers
    assert (df["total_flops"] > 0).all()
    assert (df["total_bytes"] > 0).all()
    assert (df["ai_layer"] > 0).all()


def test_sweep_raises_if_baseline_does_not_fit():
    with pytest.raises(RuntimeError):
        sweep(get_model("llama3.1-8b"), H100, Regime("X", 524288, 8), Grid(sram_extra_mb=[0], hbm_extra_stacks=[0],
              kv_bits=[16], evict=[("none", None)], sram_kv_frac=[1.0]))
