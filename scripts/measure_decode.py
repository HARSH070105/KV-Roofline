#!/usr/bin/env python3
"""Measure real decode latency (TPOT), peak memory and energy for one (model, S, B) point (weeks 5-6).

Uses plain HuggingFace Transformers (BF16 KV, DynamicCache) so it runs on any 24-80 GB GPU.
For FP8 / quantised KV use scripts/measure_vllm.py.

  # 0) smoke test on CPU with a tiny random Llama (no downloads, no GPU):
  python scripts/measure_decode.py --selftest

  # 1) real run (needs HF access to the model):
  python scripts/measure_decode.py --model-id meta-llama/Llama-3.1-8B-Instruct --key llama3.1-8b \
        --hw h100 --S 32768 --B 4 --out data/measurements.csv

  # 2) HBM bytes per step with Nsight Compute (slow, one step only) - see scripts/ncu_hbm_bytes.sh

Output CSV columns (consumed by scripts/validate.py):
  model,hw,n_chips,S,B,kv_bits,group,backend,tpot_ms,tpot_std_ms,peak_mem_gb,energy_mj_per_token,hbm_gb_per_step
Prefill is chunked so long prompts do not OOM. Prompts are random token ids (decode cost does not
depend on content; only on S, B and the model).
"""
import argparse
import csv
import os
import statistics
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id")
    ap.add_argument("--key", help="kvroofline model key, e.g. llama3.1-8b")
    ap.add_argument("--hw", default="h100", help="kvroofline hardware key (label only)")
    ap.add_argument("--S", type=int, default=4096)
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--prefill-chunk", type=int, default=4096)
    ap.add_argument("--out", default="data/measurements.csv")
    ap.add_argument("--nvtx", action="store_true", help="wrap each decode step in an NVTX range (for ncu)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, DynamicCache

    if a.selftest:
        from transformers import LlamaConfig, LlamaForCausalLM
        cfg = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                          num_key_value_heads=2, vocab_size=512, max_position_embeddings=1024)
        model = LlamaForCausalLM(cfg).eval()
        a.S, a.B, a.steps, a.warmup, a.key, a.out = 128, 2, 8, 2, "selftest", "/tmp/selftest_measure.csv"
        dev = "cpu"
    else:
        if not a.model_id or not a.key:
            ap.error("--model-id and --key are required (or use --selftest)")
        dev = "cuda"
        model = AutoModelForCausalLM.from_pretrained(a.model_id, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    vocab = model.config.vocab_size

    def sync():
        if dev == "cuda":
            torch.cuda.synchronize()

    # ---- optional energy counter (NVML total energy, mJ) -----------------------------------
    energy = None
    if dev == "cuda":
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
            energy = lambda: pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)  # noqa: E731
        except Exception as e:  # noqa: BLE001
            print("energy counter unavailable:", e)

    g = torch.Generator(device="cpu").manual_seed(0)
    ids = torch.randint(100, vocab - 100, (a.B, a.S), generator=g).to(dev)

    with torch.inference_mode():
        cache = DynamicCache()
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        out = None
        for i in range(0, a.S, a.prefill_chunk):
            out = model(input_ids=ids[:, i:i + a.prefill_chunk], past_key_values=cache, use_cache=True)
            cache = out.past_key_values
        sync()
        print(f"prefill {a.B}x{a.S}: {time.time() - t0:.1f}s")
        nxt = out.logits[:, -1:].argmax(-1)

        times = []
        e0 = energy() if energy else None
        for step in range(a.warmup + a.steps):
            if step == a.warmup:
                e0 = energy() if energy else None
                times = []
            sync()
            t = time.perf_counter()
            if a.nvtx and dev == "cuda":
                torch.cuda.nvtx.range_push("decode_step")
            out = model(input_ids=nxt, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            nxt = out.logits[:, -1:].argmax(-1)
            if a.nvtx and dev == "cuda":
                torch.cuda.nvtx.range_pop()
            sync()
            times.append(time.perf_counter() - t)
        e1 = energy() if energy else None

    tpot = statistics.mean(times) * 1e3
    std = statistics.pstdev(times) * 1e3
    peak = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else float("nan")
    e_tok = (e1 - e0) / (a.steps * a.B) if (energy and e0 is not None) else float("nan")  # mJ per token
    print(f"TPOT {tpot:.2f} +- {std:.2f} ms | {a.B / (tpot / 1e3):.1f} tok/s | peak {peak:.1f} GB | {e_tok:.1f} mJ/token")

    new = not os.path.exists(a.out)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["model", "hw", "n_chips", "S", "B", "kv_bits", "group", "backend", "tpot_ms", "tpot_std_ms",
                        "peak_mem_gb", "energy_mj_per_token", "hbm_gb_per_step"])
        w.writerow([a.key, a.hw, 1, a.S, a.B, 16, 0, "hf", f"{tpot:.4f}", f"{std:.4f}", f"{peak:.3f}",
                    f"{e_tok:.3f}", ""])


if __name__ == "__main__":
    main()
