# kvroofline

Parameterised roofline model + design-space exploration (DSE) for **KV-cache-bound long-context LLM decode**.
Project 13: *which memory-hierarchy change buys the most arithmetic intensity per unit area/energy: more SRAM, more HBM bandwidth, or a smaller KV cache?*

Read `HANDOFF.md` (state, merits, limits, TODO) and `docs/PROJECT_IDEA.md` (idea) first. This file is the how-to.

## 1. Setup (modelling needs no GPU)

```bash
cd kvroofline
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # numpy pandas matplotlib scipy pytest
python -m pytest -q tests                  # 25 tests; they pin every number in your deck
```
GPU extras (weeks 5-6 only): `pip install torch transformers datasets pynvml` (+ `vllm`, `kvpress`, `lm-eval` as needed).

## 2. Quick start: three commands

```bash
python scripts/explore.py                  # 4 experiments: KV share/S*, CUDA-roof hypothesis, SRAM hit rate, roofline figure
python scripts/run_dse.py --model llama3.1-8b --hw h100      # full DSE over regimes A/B/C -> results/llama3.1-8b_h100/
python scripts/run_dse.py --model llama3-70b --hw h100 --chips 8 --theta 0.5   # analytical multi-GPU case
```
Outputs per regime: `*_sweep.csv` (every config, ~50 metrics), `*_pareto.csv`, `*_pareto.png`, `*_knobs.png`, `*_roofline.png`,
`*_sram_hbm_heatmap.png`, and `RECOMMENDATIONS.md` (baseline, best config per knob family, iso-area winners).

Useful flags of `run_dse.py`:

| Flag | Meaning |
|---|---|
| `--regime NAME:S:B[:chips]` (repeatable) | custom regime, e.g. `--regime long:131072:8:1` |
| `--batch-mode max\|fixed` | `max` (default): largest feasible batch up to B, so compression's capacity benefit counts. `fixed`: exactly B (isolates pure bandwidth/AI effects) |
| `--gate 1.0` | quality gate in % vs BF16 |
| `--quality-csv data/quality_table.csv` | measured quality table (otherwise **placeholder heuristic**, flagged in the output) |
| `--theta 0.5 --rho 0.2` | KV access skew / cross-sequence sharing for the SRAM hit-rate model |
| `--dequant-flops 6 --attn-unit tensor\|cuda` | dequantisation cost per element; which units run QK/PV |
| `--hbm-speeds 1.0,1.25` | add pin-speed scaling (zero area in the model, see limitations) |
| `--calib data/calib.json` | calibration fitted in week 6 |

Use it as a library:
```python
from kvroofline import evaluate, get_hw, get_model, Compression
r = evaluate(get_model("llama3.1-8b"), get_hw("h100"), S=131072, B=3,
             comp=Compression(kv_bits=4, group=64, residual=128))
r["tpot_ms"], r["ai_attn"], r["dominant"], r["energy_mj_per_token"]
```

## 3. Layout

```
kvroofline/
  models.py       6 model configs; KV layout, attention FLOPs (GQA and absorbed MLA), MoE expected-expert-touch weights
  hardware.py     presets (h100, h200, a100-80/40, b200~, l4, rtx4090) + the knobs (sram_pin_mb, hbm_stacks, hbm_speed_scale)
  compression.py  KV bits, group metadata (4B scale+zero), residual window, streaming/Quest eviction, dequant + index overheads
  sram.py         on-chip hit rate h(C): pinning, skew (theta), prefix sharing (rho), Quest footprint
  engine.py       per-op roofline with tensor / CUDA / HBM / SRAM roofs, capacity, multi-chip TP, energy; crossover S*
  cost.py         area (SRAM mm2, HBM PHY + package) and energy (pJ/bit) models: ALL PLACEHOLDERS
  quality.py      quality table Q + gate; placeholder heuristic until measured
  dse.py          grid sweep, Pareto, iso-area winners, per-family best, ratios (dAI/mm2, speedup/mm2, ...)
  plots.py        figures
scripts/
  explore.py run_dse.py                           modelling (no GPU)
  measure_decode.py measure_vllm.py ncu_hbm_bytes.sh parse_ncu.py   GPU measurements
  measure_quality.py                              simulated KV-quant perplexity -> data/quality_table.csv
  validate.py                                     MAPE + calibration fit
tests/test_engine.py
```

## 4. What to replace before quoting any number (placeholders)

The engine is correct-by-construction against the deck; the *constants* are not yet sourced. Everything below must be cited or measured before it appears in the report.

| Placeholder | Where | Replace with |
|---|---|---|
| SRAM density 3.9 MB/mm², pJ/bit `0.10+0.02·sqrt(MB)`, leakage | `cost.py` | CACTI 7 output for your node (week 4) |
| HBM pJ/bit (3.9), PHY 14 mm² / package 110 mm² per stack, pkg weight 0.5, speed-energy exponent | `hardware.py`, `cost.py` | vendor / paper figures |
| Hardware datasheet values (esp. `b200`, `l4`, `rtx4090`, L2 pin fraction, L2 bandwidth) | `hardware.py` | datasheets (each is labelled VERIFY) |
| `theta` (access skew), `rho` (sharing) | `sram.py` | measure attention-mass concentration / reuse on a real model (this *is* h(C), the key empirical input) |
| `dequant_flops` = 6, `softmax_flops` = 5 | `compression.py` | count instructions in the real dequant kernel (KIVI / KVQuant) |
| Quality heuristic | `quality.py` | `measure_quality.py` + `kvpress` rows in `data/quality_table.csv` (week 5) |
| Model configs | `models.py` | re-confirm each from the HF `config.json` (week 1) |

## 5. Measurement workflow (weeks 5-6, needs a 24-80 GB GPU)

```bash
# smoke tests first (CPU, tiny random model, no downloads):
python scripts/measure_decode.py --selftest
python scripts/measure_quality.py --selftest

# latency + energy at 2-3 contexts (BF16 KV, HuggingFace)
for S in 4096 16384 32768; do for B in 1 4; do
  python scripts/measure_decode.py --model-id meta-llama/Llama-3.1-8B-Instruct --key llama3.1-8b --hw h100 --S $S --B $B
done; done
# FP8 KV through vLLM
python scripts/measure_vllm.py --model-id meta-llama/Llama-3.1-8B-Instruct --key llama3.1-8b --S 32768 --B 4 --kv-dtype fp8
# HBM bytes per decode step (Nsight Compute, slow, one step)
bash scripts/ncu_hbm_bytes.sh meta-llama/Llama-3.1-8B-Instruct llama3.1-8b h100 8192 2
# quality table (simulated KV quantisation perplexity)
python scripts/measure_quality.py --model-id meta-llama/Llama-3.1-8B-Instruct --contexts 2048 8192 --bits 8 4 2 --groups 32 64 128
# validate and calibrate
python scripts/validate.py --meas data/measurements.csv --fit --holdout 4 --calib-out data/calib.json
python scripts/validate.py --hbm data/hbm_bytes.csv
```
Set the hardware label (`--hw`) to match the GPU you really have (`l4`, `rtx4090`, `a100-40`, ...) so predictions use that chip's roofs.
Eviction quality rows: run `kvpress` (StreamingLLM / Expected Attention presses) and append rows with
`evict_policy,evict_budget,...,source=measured` using the CSV schema documented in `kvroofline/quality.py`.
Only `source=measured` rows are used; anything else falls back to the placeholder and is flagged `quality_measured=False`.

**Honesty note:** the measurement scripts were written to run on a GPU box and could not be executed against a real model in the environment where this repo was generated. Their numerical core (calibration fit) is tested on synthetic data; run the two `--selftest` commands first and expect to adapt small API details (vLLM and transformers change often).

## 6. Modelling choices and known limitations (discuss with your teammates)

* **ΔAI is zero for bandwidth** by construction (deck: "leaves AI unchanged"). The tool therefore also reports `d_ai_over_ridge` and tokens/s per mm²; decide which is the headline currency.
* **"HBM stack" bundles bandwidth *and* capacity** (more stacks ⇒ larger max batch ⇒ higher AI in `max` batch mode). Use `--batch-mode fixed` or `--hbm-speeds` to isolate pure bandwidth. Pin-speed scaling has no area cost in the model, so it is opt-in.
* **Compression has ~0 area**, so area-normalised ratios are undefined for it; compare knob families with the **iso-area table** (best config under an area budget, quality-gated).
* Throughput (`tokens_per_s`) at max batch conflates latency and capacity effects: always read it with `tpot_ms` and `B`.
* Regimes are defined by (S, B cap); S* depends on B, so "regime A is weight-dominated" holds only for small B (the default A uses B=8).
* Tensor-parallel only; MLA latent is **replicated** on each chip (not sharded). Decode only (no prefill, no speculative decoding, no PIM/CXL, as scoped in the deck).
* Ideal roofline: `Calibration` (bw_eff, tc_eff, t_fixed) absorbs real-kernel inefficiency after week 6.
* Quest-style selection keeps the full cache stored and pays page-metadata reads; SRAM coverage for Quest is computed against the stored footprint.
* Simulated KV quantisation applies K quantisation pre-RoPE and excludes the full-precision residual window except for the incomplete tail group.
