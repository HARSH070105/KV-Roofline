#!/usr/bin/env python3
"""Analytical verification of crossover context S* on DeepSeek-V2."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kvroofline import Compression, evaluate, get_hw, get_model

m = get_model("deepseek-v2")
comp = Compression(kv_bits=16, group=0, residual=0, dequant_flops=0)
for chips in (1, 8):
    hw = get_hw("h100").with_(n_chips=chips)
    for S in (76895, 610824):
        r = evaluate(m, hw, S=S, B=1, comp=comp)
        print(chips, S, r["dominant"], "kv_hbm_gb=", round(r["kv_hbm_gb"], 2),
              "w_hbm_gb=", round(r["w_hbm_gb"], 2),
              "t_attn_ms=", round(r["t_attn_ms"], 3))
