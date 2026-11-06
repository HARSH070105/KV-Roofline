# kvroofline: Roofline-Guided Design-Space Exploration for KV-Cache-Bound Long-Context LLM Inference

Parameterised roofline model and design-space exploration (DSE) framework for **KV-cache-bound long-context LLM decode**.

> **Project Core Question**: *Which memory-hierarchy intervention buys the most arithmetic intensity per unit area and energy: more on-chip SRAM, more HBM bandwidth, or KV-cache compression?*

* **Primary Technical Report**: [`docs/FINAL_REPORT.md`](docs/FINAL_REPORT.md)
* **Conceptual Guide**: [`docs/PROJECT_IDEA.md`](docs/PROJECT_IDEA.md)

---

## 1. Quick Start

### Local Setup (Modelling & Analysis)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q tests                  # 26 unit tests verifying all physical invariants
```

### Run Experiments
```bash
python scripts/explore.py                  # 4 core experiments: KV share, CUDA roof, SRAM hit rate, roofline
python scripts/layerwise_analysis.py --compare-llm-viewer  # Reproduce LLM-Viewer (arXiv:2402.16363)
python scripts/sensitivity.py              # 11 parametric cost and technology perturbations
python scripts/run_dse.py --model llama3.1-8b --hw h100 --quality-csv data/quality_table.csv --calib data/calib.json
```

---

## 2. HPC & Multi-GPU Execution

For execution on HPC clusters (e.g. Slurm, multi-GPU nodes with NVIDIA H100, A100, RTX 4090, or L4 GPUs):

### Turn-Key Shell Runner
```bash
# Automatic hardware detection, live benchmarks, calibration, and full DSE sweeps:
bash run_hpc.sh

# Or run live GPU latency profiling and calibration only:
bash run_hpc.sh --benchmark-gpu
```

### Slurm Batch Submission
```bash
sbatch run_hpc.slurm
```

---

## 3. Repository Architecture

```
kvroofline/
  models.py       6 model configs (LLaMA-3.1-8B, LLaMA-3-70B, DeepSeek-V2-Lite, DeepSeek-V2, Qwen-2.5-7B, LLaMA-2-7B)
  hardware.py     Hardware presets (H100, H200, B200, A100-80, A100-40, L4, RTX 4090) and hardware knobs
  compression.py  KV bits, scale/zero metadata overhead, residual window, eviction (StreamingLLM, Quest), dequant FLOPs
  sram.py         On-chip hit rate h(C): capacity pinning, access skew (theta), prefix sharing (rho), Quest footprint
  engine.py       Two-group hierarchical roofline (tensor, CUDA, HBM, SRAM roofs), capacity limits, multi-chip TP
  layerwise.py    Layer-by-layer FLOP and memory breakdown matching and extending LLM-Viewer
  cost.py         Sourced area (SRAM mm2, HBM PHY + package) and energy (pJ/bit) models
  quality.py      Quality table Q loader and accuracy constraint gating
  dse.py          Multi-objective grid sweeps, Pareto front extraction, iso-area evaluation
  plots.py        Publication-grade figures (roofline, crossover, Pareto, heatmaps)

scripts/
  run_hpc.sh              Turn-key HPC execution runner with automatic GPU detection
  run_hpc.slurm           Slurm batch job submission script
  explore.py              Core thesis experiments (KV share, S*, CUDA roof, SRAM sensitivity)
  layerwise_analysis.py   Per-layer breakdown and comparison with LLM-Viewer baseline
  sensitivity.py          Parametric robustness analysis under +-30% to +-50% cost perturbations
  validate.py             Empirical calibration and MAPE evaluation (held-out TPOT, HBM bytes)
  run_dse.py              Full design-space exploration over context regimes
  measure_decode.py       HuggingFace decode latency (TPOT), peak memory, and NVML energy profiling
  measure_vllm.py         vLLM decode throughput benchmark (FP8/BF16 KV)
  measure_quality.py      Simulated KV-quantization perplexity evaluation
  ncu_hbm_bytes.sh        Nsight Compute kernel profiling for HBM traffic validation
  parse_ncu.py            Parser extracting total dram bytes from ncu logs

data/
  quality_table.csv       Verified empirical quality degradation table across contexts
  measurements.csv        Empirical decode latency (TPOT) benchmarking dataset
  hbm_bytes.csv           Nsight Compute profiled HBM memory traffic
  calib.json              Fitted hardware calibration parameters

docs/
  FINAL_REPORT.md         Complete final deliverables report & memory-hierarchy recommendations
  PROJECT_IDEA.md         Conceptual summary and physical derivations
```

---

## 4. Physical Sourcing and Constants

All microarchitectural parameters are sourced from technology node publications, JEDEC standards, and CACTI 7.0 models:

| Parameter | Value | Source / Methodology |
|---|---|---|
| **SRAM Macro Density** | $3.90\text{ MB/mm}^2$ | TSMC N5/N4 high-density 6T bitcell ($0.021\ \mu\text{m}^2$) + periphery (ISSCC 2020 / CACTI 7.0) |
| **SRAM Dynamic Energy** | $0.10 + 0.02\sqrt{C_{\text{MB}}}\text{ pJ/bit}$ | CACTI 7.0 at 5nm with H-tree wire scaling |
| **SRAM Static Leakage** | $25.0\text{ mW/MB}$ | CACTI 7.0 at $85^\circ\text{C}$ operating temperature |
| **HBM3 PHY Area** | $14.0\text{ mm}^2\text{ / stack}$ | Synopsys HBM3 PHY IP in TSMC N5 |
| **HBM Package Overhead**| $110.0\text{ mm}^2\text{ / stack}$ | TSMC CoWoS / interposer substrate footprint per stack |
| **HBM3 / HBM3e Energy** | $3.9\text{ pJ/bit} / 3.5\text{ pJ/bit}$ | JEDEC JESD238 standard and SK Hynix Hot Chips 35 |
| **Hardware Datasheets** | H100, H200, B200, A100, L4, RTX 4090 | NVIDIA Architecture Whitepapers and official datasheets |
| **Quality Degradation**| `data/quality_table.csv` | Empirical literature data (KIVI, KVQuant, PM-KVQ, StreamingLLM, Quest) |

---

## 5. Empirical Validation Highlights

Validation against real profiled runs across 29 hardware configurations yields:
* **HBM Traffic (Nsight Compute)**: **$\text{MAPE} = 3.3\%$**
* **Held-Out Decode Latency (TPOT)**: **$\text{MAPE} = 12.9\%$** (surpassing the $\le 15\%$ accuracy target)
* **Layer-wise Consistency**: Exact agreement with LLM-Viewer layerwise totals for LLaMA-2-7B.

---

## 6. Key Findings & Recommendations

1. **Regime A (Short/Medium: $\le 8\text{k}$)**:
   * Workload is weight-dominated ($S < S^*$).
   * **Recommendation**: **Baseline SRAM + 6 HBM Stacks (H200-style) + FP8 KV** ($1.65\times - 1.92\times$ speedup). SRAM scaling provides negligible ROI ($<1.05\times$ for $263\text{ mm}^2$).
2. **Regime B (Long: $128\text{k}$)**:
   * Workload is KV-dominated ($77\%$ of HBM bytes).
   * **Recommendation**: **INT4 g64 KV + Quest 16k Eviction + Baseline HBM** (**$6.23\times$ speedup at $0\text{ mm}^2$**, $+0.66\%$ quality loss). Expands feasible batch concurrency $B=3 \to 8$.
3. **Regime C (Ultra-Long: $512\text{k}+$ רבים)**:
   * Workload is capacity-bound. BF16 KV ($68.7\text{ GB/seq}$) is physically infeasible on 1 GPU.
   * **Recommendation**: **INT4 g64 KV + Quest Eviction + 2-GPU Tensor Parallelism / 6 HBM Stacks** (**$11.1\times - 17.0\times$ speedup**, $+0.89\%$ quality loss).
