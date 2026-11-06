#!/usr/bin/env python3
"""Build the quality table Q (week 5): perplexity degradation of simulated KV quantisation.

Simulation ("fake quant"): K/V projections are quantised and immediately dequantised inside the model,
so the attention sees exactly the rounding error a real INT-b KV cache would introduce. No speed claim.
  * Keys  : per-channel, groups of G *tokens* (KIVI style), applied to k_proj output = PRE-RoPE (KVQuant style)
  * Values: per-token,   groups of G *channels*
  * asymmetric uniform (min/max) quantisation with FP16 scale/zero (their bytes are in compression.py)
  * the incomplete tail group (< G tokens) of K stays full precision (a crude residual window)
Limitations: llama/qwen-style models only (needs q/k/v_proj; not MLA); eviction rows should be produced with
`kvpress` (see README) and appended to the same CSV with source=measured.

  python scripts/measure_quality.py --selftest
  python scripts/measure_quality.py --model-id meta-llama/Llama-3.1-8B-Instruct \
        --contexts 2048 8192 32768 --bits 8 4 2 --groups 32 64 128 --max-chunks 8 --out data/quality_table.csv

Output rows: kv_bits,group,evict_policy,evict_budget,w_bits,context,metric,delta_pct,source
"""
import argparse
import csv
import math
import os


def fake_quant(x, bits, group, axis):
    """Asymmetric uniform fake-quant of x along `axis` in groups of `group` elements.
    axis=-1: groups of channels (per-token, for V).  axis=1: groups of tokens (per-channel, for K) on [B,S,C]."""
    import torch
    levels = 2 ** bits - 1
    if axis == -1:
        B, S, C = x.shape
        g = min(group, C)
        assert C % g == 0, f"channels {C} not divisible by group {g}"
        y = x.reshape(B, S, C // g, g).float()
        mn, mx = y.amin(-1, keepdim=True), y.amax(-1, keepdim=True)
        sc = ((mx - mn) / levels).clamp_min(1e-8)
        q = torch.round((y - mn) / sc).clamp(0, levels)
        return (q * sc + mn).reshape(B, S, C).to(x.dtype)
    B, S, C = x.shape
    full = (S // group) * group
    if full == 0:
        return x
    head = x[:, :full].reshape(B, full // group, group, C).float()
    mn, mx = head.amin(2, keepdim=True), head.amax(2, keepdim=True)
    sc = ((mx - mn) / levels).clamp_min(1e-8)
    q = torch.round((head - mn) / sc).clamp(0, levels)
    deq = (q * sc + mn).reshape(B, full, C).to(x.dtype)
    return torch.cat([deq, x[:, full:]], dim=1)


def install_hooks(model, bits, group):
    handles = []
    for layer in model.model.layers:
        at = layer.self_attn
        handles.append(at.k_proj.register_forward_hook(lambda m, i, o: fake_quant(o, bits, group, axis=1)))
        handles.append(at.v_proj.register_forward_hook(lambda m, i, o: fake_quant(o, bits, group, axis=-1)))
    return handles


def perplexity(model, chunks, device):
    import torch
    nll, n = 0.0, 0
    with torch.inference_mode():
        for ids in chunks:
            ids = ids.to(device)
            logits = model(input_ids=ids).logits[:, :-1].float()
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1),
                                                     reduction="sum")
            nll += loss.item()
            n += ids.size(1) - 1
    return math.exp(nll / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id")
    ap.add_argument("--contexts", type=int, nargs="+", default=[2048, 8192])
    ap.add_argument("--bits", type=int, nargs="+", default=[8, 4, 2])
    ap.add_argument("--groups", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--max-chunks", type=int, default=8, help="chunks per context (eval cost control)")
    ap.add_argument("--dataset", default="wikitext", help="wikitext (WikiText-2 test) | random (selftest)")
    ap.add_argument("--out", default="data/quality_table.csv")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if a.selftest:
        import tempfile
        from transformers import LlamaConfig, LlamaForCausalLM
        cfg = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                          num_key_value_heads=2, vocab_size=512, max_position_embeddings=1024)
        model, dev = LlamaForCausalLM(cfg).eval(), "cpu"
        a.contexts, a.bits, a.groups, a.max_chunks, a.dataset, a.out = [64, 128], [8, 4], [16], 2, "random", os.path.join(tempfile.gettempdir(), "selftest_quality.csv")
        tokens = torch.randint(0, 512, (1, 4096))
    else:
        dev = "cuda"
        tok = AutoTokenizer.from_pretrained(a.model_id)
        model = AutoModelForCausalLM.from_pretrained(a.model_id, torch_dtype=torch.bfloat16, device_map="cuda").eval()
        from datasets import load_dataset
        text = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
        tokens = tok(text, return_tensors="pt").input_ids

    rows = []
    for ctx in a.contexts:
        n = min(a.max_chunks, tokens.size(1) // ctx)
        if n == 0:
            print(f"context {ctx}: not enough tokens ({tokens.size(1)}), skipping")
            continue
        chunks = [tokens[:, i * ctx:(i + 1) * ctx] for i in range(n)]
        base = perplexity(model, chunks, dev)
        print(f"ctx {ctx}: BF16 ppl {base:.3f} ({n} chunks)")
        for bits in a.bits:
            for g in a.groups:
                hs = install_hooks(model, bits, g)
                ppl = perplexity(model, chunks, dev)
                for h in hs:
                    h.remove()
                d = 100.0 * (ppl / base - 1.0)
                print(f"  INT{bits} g{g}: ppl {ppl:.3f}  delta {d:+.3f}%")
                rows.append([bits, g, "none", 0, 16, ctx, "ppl_delta", f"{d:.4f}",
                             "selftest" if a.selftest else "measured"])
    new = not os.path.exists(a.out) or os.path.getsize(a.out) == 0
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["kv_bits", "group", "evict_policy", "evict_budget", "w_bits", "context", "metric",
                        "delta_pct", "source"])
        w.writerows(rows)
    print("appended", len(rows), "rows to", a.out)


if __name__ == "__main__":
    main()
