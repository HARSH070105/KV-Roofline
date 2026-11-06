#!/usr/bin/env python3
"""Measure decode TPOT with vLLM (supports FP8 KV cache). TPOT = (t(N tokens) - t(1 token)) / (N - 1).

  python scripts/measure_vllm.py --model-id meta-llama/Llama-3.1-8B-Instruct --key llama3.1-8b \
         --hw h100 --S 32768 --B 4 --kv-dtype fp8 --out data/measurements.csv

vLLM's API changes between releases; if `LLM.generate` rejects the prompt format, adapt the two lines
marked ADAPT. Prefix caching is disabled so every sequence really reads its own KV.
Note: this measures end-to-end serving latency (scheduler + sampling included), so expect a larger
fixed overhead than the roofline predicts; scripts/validate.py --fit absorbs it in t_fixed.
"""
import argparse
import csv
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--hw", default="h100")
    ap.add_argument("--S", type=int, default=4096)
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--N", type=int, default=64, help="decode tokens")
    ap.add_argument("--kv-dtype", default="auto", choices=["auto", "fp8"])
    ap.add_argument("--out", default="data/measurements.csv")
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()

    import random
    from vllm import LLM, SamplingParams

    llm = LLM(model=a.model_id, max_model_len=a.S + a.N + 16, kv_cache_dtype=a.kv_dtype,
              enable_prefix_caching=False, gpu_memory_utilization=0.92, dtype="bfloat16")
    rng = random.Random(0)
    vocab = llm.get_tokenizer().vocab_size
    prompts = [{"prompt_token_ids": [rng.randrange(100, vocab - 100) for _ in range(a.S)]} for _ in range(a.B)]  # ADAPT

    def run(n):
        sp = SamplingParams(max_tokens=n, ignore_eos=True, temperature=0.0)
        t = time.perf_counter()
        llm.generate(prompts, sp, use_tqdm=False)  # ADAPT
        return time.perf_counter() - t

    run(2)  # warmup / graph capture
    tpots = []
    for _ in range(a.repeats):
        t1, tn = run(1), run(a.N)
        tpots.append((tn - t1) / (a.N - 1) * 1e3)
    tpot = sum(tpots) / len(tpots)
    print(f"TPOT {tpot:.2f} ms  ({a.B / (tpot / 1e3):.1f} tok/s)  kv={a.kv_dtype}")

    new = not os.path.exists(a.out)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["model", "hw", "n_chips", "S", "B", "kv_bits", "group", "backend", "tpot_ms", "tpot_std_ms",
                        "peak_mem_gb", "energy_mj_per_token", "hbm_gb_per_step"])
        w.writerow([a.key, a.hw, 1, a.S, a.B, 8 if a.kv_dtype == "fp8" else 16, 0, "vllm", f"{tpot:.4f}", "", "", "", ""])


if __name__ == "__main__":
    main()
