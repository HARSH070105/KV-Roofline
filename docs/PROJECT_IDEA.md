# What you are actually doing (Project 13, in plain language)

## The one-paragraph version

When an LLM generates text, each new token must read **all the model weights** and **the whole KV cache** (the saved keys/values of every earlier token) from GPU memory (HBM), and does very little arithmetic with them. So decode speed is set by **memory bandwidth, not compute**. As context grows (8k → 128k → 512k tokens) the KV cache becomes the dominant thing being read. Many fixes exist: compress the KV cache (quantisation), throw tokens away (eviction), add a bigger on-chip cache (SRAM), or buy more HBM bandwidth. **Nobody has ranked these fixes against each other on one hardware-aware scale (area and energy), per context regime.** Your project builds the calculator that does that and uses it to recommend a memory-hierarchy configuration for 2–3 context-length regimes.

## The model in five equations (all in the deck)

| Idea | Formula | Meaning |
|---|---|---|
| Roofline | `Perf = min(P_peak, BW × AI)` | Speed is capped by compute *or* by bandwidth × arithmetic intensity (FLOP per byte) |
| Ridge | `ridge = P_peak / BW` (H100: ~295 FLOP/B) | Workloads with AI below this are memory-bound. Decode AI is single digits |
| KV size | `2·L·H_kv·d·b` bytes per token | Llama-3.1-8B: 128 KiB/token → 17 GB for one 128k sequence |
| Attention AI | `2g/b` (g = query heads per KV head, b = bytes/element) | Independent of context length *and* batch: only architecture (GQA/MLA) and precision change it |
| Crossover | `S* = weight_bytes / (B · KV_bytes_per_token)` | Context above which KV traffic exceeds weight traffic. Below S*, compressing KV barely helps |

## The three kinds of knob (what the tool sweeps)

| Knob | What it changes | Cost | In the tool |
|---|---|---|---|
| **KV compression** (bits, group size, eviction budget) | Fewer bytes per token → higher AI, smaller KV, **bigger batch fits** | ~0 area; **quality loss**; metadata (+12–25 %), dequant FLOPs, indexing | `compression.py` |
| **Bigger SRAM** | Serves a fraction `h` of KV reads on-chip: `HBM bytes = KV·(1−h)` | mm² of die; SRAM scaling has stalled | `sram.py`, `cost.py` |
| **More HBM bandwidth** | Steeper memory roof (AI unchanged, ridge moves) | package area, pJ/bit, **adds capacity too** | `hardware.py` |

Quality-affecting knobs (compression) are treated as a **constraint**: configs worse than ~1 % of BF16 are dropped *before* ranking. The ranking itself is on throughput, area, energy (a 3-way Pareto front).

## Key Empirical Findings (H100 SXM, LLaMA-3.1-8B, Calibrated Model)

| Regime | Baseline (BF16 KV) | Best Single Knob | Key Architecture Finding |
|---|---|---|---|
| A: 8k, B=8 | weight-bound (`linear:hbm`), KV is 36% of bytes | HBM stacks (×1.40 for +2 stacks) ≈ KV8 (×1.37) | Below S*, compressing KV has limited impact. SRAM gives only ×1.05 for 263 mm². |
| B: 128k | KV-bound (77% of bytes), max batch 3 | **KV4 g64 + Quest: ×6.23** at ~0 mm² (B: 3→8) | Working set ≈ 52 GB vs ≤ 1 GB SRAM. Compression unlocks batch capacity. Combined with +2 HBM stacks: ×8.72. |
| C: 512k (2 GPUs) | KV-bound, capacity-limited, B=1 | **KV4 g64 + Quest: ×11.15** (B: 1→6) | Compression is strictly required for physical feasibility. Combined with +2 HBM stacks: ×16.96. |

Three core architectural takeaways:
1. **SRAM scaling alone is ineffective against the full KV cache**: Each KV byte is read once per step, so a cache of size C covers only C/W of reads (e.g. 1 GB covers <2% of a 52 GB working set). SRAM only pays off after token eviction shrinks the working set, or with extreme access skew ($\theta \le 0.5$).
2. **INT4 attention conditionally binds on CUDA cores**: When QK/PV run on tensor cores, decode remains memory-bound until dequantization exceeds $\sim 10\text{ FLOP/elem}$. If matmuls run on CUDA cores, it flips to CUDA-bound immediately.
3. **Compression throughput gains are largely capacity gains**: Smaller KV allows batch size to expand ($B=3 \to 8$), multiplying throughput while keeping per-token latency bounded. Report `tpot_ms` (latency) and `tokens_per_s` (throughput) side-by-side.

## How the 8-week roadmap maps onto the code

| Week | Deck milestone | Code / command |
|---|---|---|
| 1 | Scope, verify models, freeze metrics + quality gate | `python -m pytest tests` (deck numbers are unit tests); edit `models.py` against each `config.json`; freeze `--gate` |
| 2 | Core engine, reproduce LLM-Viewer | `kvroofline/engine.py`; compare per-layer FLOPs/bytes with LLM-Viewer for the same config |
| 3 | Compression modules, overheads, CUDA roof, SRAM hit rate | `compression.py`, `sram.py`; `scripts/explore.py` |
| 4 | Area + energy, **M1: tool v0** | `cost.py`; replace placeholders with CACTI + HBM pJ/bit sources; `scripts/run_dse.py` |
| 5 | Measure with vLLM + Nsight, build Q | `measure_decode.py`, `measure_vllm.py`, `ncu_hbm_bytes.sh`, `measure_quality.py` |
| 6 | Validate (MAPE, roof attainment), **M2** | `validate.py --meas data/measurements.csv --hbm data/hbm_bytes.csv --fit --holdout 4`; cross-check LLM-Viewer / LLMCompass |
| 7 | Quality-gated sweeps, Pareto, recommendations | `run_dse.py --quality-csv data/quality_table.csv --calib data/calib.json` |
| 8 | Report, docs, **M3** | `results/*/RECOMMENDATIONS.md`, figures |
