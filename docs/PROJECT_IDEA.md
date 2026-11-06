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

## What the tool already tells you (first results, H100, Llama-3.1-8B, *placeholder quality numbers*)

| Regime | Baseline (BF16 KV) | Best single knob | Observation |
|---|---|---|---|
| A: 8k, B=8 | weight-bound (`linear:hbm`), KV is 36 % of bytes | HBM stacks (×1.2 for +1 stack) ≈ KV8 (×1.2) | KV compression barely helps below S*; SRAM ×1.05 for 263 mm² |
| B: 128k | KV-bound (77 % of bytes), max batch only 3 | **KV4 g64: ×3.3** at ~0 mm², then HBM stacks | SRAM ×1.02 for 263 mm²: working set ≈ 50 GB vs ≤ 1 GB of SRAM |
| C: 512k (2 GPUs) | KV-bound, capacity-limited, B=1 | **KV4: ×3.9** (capacity lets B grow 1→6) | Compression is needed for *feasibility*, not just speed |

Three non-obvious findings you should verify and turn into report claims:
1. **SRAM alone is nearly useless against the full KV cache.** Each KV byte is read once per step, so a cache of size C covers only C/W of reads. It only pays off after eviction shrinks the working set, or if access is skewed or shared (`theta`, `rho` in `sram.py`). The hit-rate curve `h(C)` is *the* empirical unknown of the project.
2. **INT4 attention only hits the CUDA-core roof under conditions.** With QK/PV on tensor cores it stays memory-bound until dequant costs ≳ 10 FLOP/element; if the matmuls run on CUDA cores it flips to CUDA-bound at once (run `scripts/explore.py`, experiment E2). So your deck's hypothesis is conditional: it becomes a *measurable* claim about the kernel.
3. **Part of the "compression" benefit is really a capacity benefit.** Smaller KV → larger batch → higher throughput. Report `tpot_ms` (latency) and `tokens_per_s` (throughput) separately.

## How the 8-week roadmap maps onto the code

| Week | Deck milestone | Code / command |
|---|---|---|
| 1 | Scope, verify models, freeze metrics + quality gate | `python -m pytest tests` (deck numbers are unit tests); edit `models.py` against each `config.json`; freeze `--gate` |
| 2 | Core engine, reproduce LLM-Viewer | `kvroofline/engine.py`; compare per-layer FLOPs/bytes with LLM-Viewer for the same config |
| 3 | Compression modules, overheads, CUDA roof, SRAM hit rate | `compression.py`, `sram.py`; `scripts/explore.py` |
| 4 | Area + energy, **M1: tool v0** | `cost.py`; replace placeholders with CACTI + HBM pJ/bit sources; `scripts/run_dse.py` |
| 5 | Measure with vLLM + Nsight, build Q | `measure_decode.py`, `measure_vllm.py`, `ncu_hbm_bytes.sh`, `measure_quality.py` |
| 6 | Validate (MAPE, roof attainment), **M2** | `validate.py --fit --holdout 4`; cross-check LLM-Viewer / LLMCompass |
| 7 | Quality-gated sweeps, Pareto, recommendations | `run_dse.py --quality-csv data/quality_table.csv --calib data/calib.json` |
| 8 | Report, docs, **M3** | `results/*/RECOMMENDATIONS.md`, figures |
