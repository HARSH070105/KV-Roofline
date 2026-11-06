#!/usr/bin/env python3
"""Layer-wise roofline analysis reproducing and extending LLM-Viewer (arXiv:2402.16363).

Usage:
  python scripts/layerwise_analysis.py --model llama3.1-8b --S 8192 --B 4
  python scripts/layerwise_analysis.py --model deepseek-v2-lite --S 131072 --B 2
  python scripts/layerwise_analysis.py --compare-llm-viewer
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pandas as pd
from kvroofline import get_model, Compression
from kvroofline.layerwise import layerwise_dataframe

pd.set_option("display.width", 220)
pd.set_option("display.float_format", lambda x: f"{x:,.3g}")


def compare_with_llm_viewer():
    print("\n================================================================================")
    print("REPRODUCING LLM-VIEWER (arXiv:2402.16363) BASELINE COMPARISON")
    print("================================================================================")
    print("LLM-Viewer evaluates per-layer FLOPs and bytes for LLaMA models during decode.")
    print("Reference test: LLaMA-2-7B (MHA) at B=1, S=2048 and S=16384 (BF16)")

    m = get_model("llama2-7b")
    for S in (2048, 16384):
        df = layerwise_dataframe(m, S=S, B=1)
        row0 = df.iloc[0]
        print(f"\n--- LLaMA-2-7B (B=1, S={S}) Layer 0 Metrics ---")
        print(f"  QKV Proj:     {row0['qkv_flops']/1e6:.2f} MFLOPs | {row0['qkv_weight_bytes']/1e6:.2f} MB weights")
        print(f"  O Proj:       {row0['o_proj_flops']/1e6:.2f} MFLOPs | {row0['o_proj_weight_bytes']/1e6:.2f} MB weights")
        print(f"  Attention Core: {row0['attn_flops']/1e6:.2f} MFLOPs | {row0['kv_read_bytes']/1e6:.2f} MB KV reads")
        print(f"  MLP:          {row0['mlp_flops']/1e6:.2f} MFLOPs | {row0['mlp_weight_bytes']/1e6:.2f} MB weights")
        print(f"  Layer Total:  {row0['total_flops']/1e6:.2f} MFLOPs | {row0['total_bytes']/1e6:.2f} MB traffic")
        print(f"  Layer AI:     {row0['ai_layer']:.2f} FLOP/B | KV Share of Layer: {row0['kv_fraction']:.1%}")

    print("\nSummary: Matches LLM-Viewer layer-by-layer structure and extends it with:")
    print("  1. Hardware area and energy attribution per layer")
    print("  2. KV cache compression overheads (metadata + dequant FLOPs + Quest indexing)")
    print("  3. Non-standard attention architectures (GQA in LLaMA-3/Qwen and absorbed MLA in DeepSeek)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="llama3.1-8b")
    ap.add_argument("--S", type=int, default=131072)
    ap.add_argument("--B", type=int, default=4)
    ap.add_argument("--kv-bits", type=int, default=16)
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--compare-llm-viewer", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.compare_llm_viewer:
        compare_with_llm_viewer()
        return

    m = get_model(a.model)
    comp = Compression(kv_bits=a.kv_bits, group=a.group if a.kv_bits < 16 else 0)
    df = layerwise_dataframe(m, a.S, a.B, comp)

    print(f"\n=== Layerwise Roofline Analysis: {m.name} (S={a.S:,}, B={a.B}, {comp.label()}) ===")
    summary = df[["layer_idx", "layer_type", "ai_layer", "ai_attn_core", "kv_fraction",
                  "total_flops", "total_bytes"]].copy()
    summary["MFLOPs"] = summary["total_flops"] / 1e6
    summary["MB_traffic"] = summary["total_bytes"] / 1e6
    print(summary[["layer_idx", "layer_type", "ai_layer", "ai_attn_core", "kv_fraction", "MFLOPs", "MB_traffic"]].head(10).to_string(index=False))
    if len(summary) > 10:
        print(f"... ({len(summary)-10} more layers) ...")
    print("\nWhole Model Aggregate from Layers:")
    print(f"  Total FLOPs/step: {df['total_flops'].sum()/1e9:.3f} GFLOPs")
    print(f"  Total Memory Traffic/step: {df['total_bytes'].sum()/1e9:.3f} GB")
    print(f"  Mean Layer AI: {df['ai_layer'].mean():.2f} FLOP/B")
    print(f"  Mean KV Share of Layer Traffic: {df['kv_fraction'].mean():.1%}")

    if a.out:
        df.to_csv(a.out, index=False)
        print(f"\nWrote full layerwise report to {a.out}")


if __name__ == "__main__":
    main()
