# Repository Structure: KV-Roofline

**Project #13**: *Roofline-Guided Design-Space Exploration for KV-Cache-Bound Long-Context LLM Inference*
**Repository Path**: repository root
**Core Framework**: `kvroofline` (Pure Python / NumPy / SciPy / Pandas / Matplotlib)

---

## 1. High-Level Architecture Overview

The `KV-Roofline` repository implements a parameterised, two-group hierarchical roofline evaluation and design-space exploration (DSE) framework. It evaluates the trade-offs between memory-hierarchy interventions (**on-chip SRAM scaling**, **HBM bandwidth/stacking**, and **KV-cache compression/eviction**) for autoregressive decode across varying context lengths ($2\text{k} \to 512\text{k}+$ tokens), batch sizes, and hardware architectures (NVIDIA H100, H200, B200, A100, L4, RTX 4090).

```
                            ┌────────────────────────────────────────┐
                            │              Input Configs             │
                            │  models.py | hardware.py | compression │
                            └───────────────────┬────────────────────┘
                                                │
                                                ▼
                            ┌────────────────────────────────────────┐
                            │           kvroofline Engine            │
                            │   engine.py | sram.py | layerwise.py   │
                            │   cost.py (Area & pJ) | quality.py     │
                            └───────────────────┬────────────────────┘
                                                │
                     ┌──────────────────────────┼──────────────────────────┐
                     ▼                          ▼                          ▼
        ┌─────────────────────────┐┌─────────────────────────┐┌─────────────────────────┐
        │       DSE Sweeps        ││   Empirical Validation  ││   Research Extensions   │
        │ scripts/run_dse.py      ││ scripts/validate.py     ││ extensions/             │
        │ kvroofline/dse.py       ││ data/measurements.csv   ││   cuda_ceiling_study.py │
        │ results/*/              ││ data/hbm_bytes.csv      ││   frontier_schemes.py   │
        │   Pareto fronts & plots ││ MAPE ≤ 15% verification ││ hypervolume_and_validation.py │
        └─────────────────────────┘└─────────────────────────┘└─────────────────────────┘
```

---

## 2. Complete Visual Directory Tree

```
KV-Roofline/
├── .agents/                                # AI agent policies and workspace settings
│   ├── rules/
│   │   └── workflow.md                     # Agent workflow rules, handoff protocol & git guidelines
│   └── settings.json                       # Agent configuration settings
│
├── .gitignore                              # Git exclusion rules (virtual environments, checkpoints)
├── AGENTS.md                               # Agent orientation guidelines (handoff-first, no git push)
├── PRESENTATION.md                         # Master slide-by-slide presentation blueprint & defense Q&A
├── README.md                               # Primary project README, quick start & empirical findings
├── requirements.txt                        # Base Python dependencies (numpy, scipy, pandas, matplotlib)
├── requirements-gpu.txt                    # GPU execution dependencies (torch, vllm, transformers, pynvml)
│
├── data/                                   # Empirical benchmarking datasets and calibration
│   ├── calib.json                          # Fitted hardware calibration parameters (bw_eff, fixed overheads)
│   ├── hbm_bytes.csv                       # Nsight Compute profiled HBM memory traffic
│   ├── measurements.csv                    # Empirical decode latency (TPOT) across hardware setups
│   └── quality_table.csv                   # Verified empirical quality degradation data across contexts
│
├── docs/                                   # Reports, academic literature, paper source and figures
│   ├── amcas_lit_rev.pdf                   # Prior literature review on roofline models, MLA & KV quant
│   ├── FINAL_PAPER.tex                     # Publication-ready IEEEtran conference paper source
│   ├── FINAL_PAPER.pdf                     # Compiled PDF from FINAL_PAPER.tex
│   ├── FINAL_REPORT.md                     # Complete project final report, methodology & findings
│   ├── PROJECT_IDEA.md                     # Conceptual guide, 5 core equations & plain-language summary
│   ├── references.bib                      # Sourced and corrected BibTeX citations
│   ├── roofline-kv-cache-intro copy.pptx   # Team introductory presentation slide deck (13 slides)
│   └── figures/                            # High-resolution figures for paper and presentation
│       ├── fig_decomposition.png           # Fixed-batch vs batch expansion decomposition
│       ├── fig_hypervolume.png             # 3D Pareto hypervolume co-design expansion
│       ├── fig_phase_oscillation.png       # Bottleneck oscillation phase map across (S, B)
│       └── fig_roofline.png                # Dual roofline trajectory (Tensor & CUDA dequant)
│
├── kvroofline/                             # Core Python analytical modelling package
│   ├── __init__.py                         # Package exports (evaluate, get_hw, get_model, Compression, etc.)
│   ├── compression.py                      # KV/weight quantization, metadata overheads, eviction (Quest)
│   ├── cost.py                             # Area (SRAM mm², HBM PHY) and energy (pJ/bit, leakage) models
│   ├── dse.py                              # Multi-objective sweeps, 3D Pareto fronts & iso-area ranking
│   ├── engine.py                           # Two-group hierarchical roofline engine & capacity checker
│   ├── hardware.py                         # Accelerator presets (H100, H200, B200, A100, L4, RTX 4090)
│   ├── layerwise.py                        # Per-layer FLOPs/bytes breakdown matching LLM-Viewer
│   ├── models.py                           # 6 LLM architectures (LLaMA-3.1, LLaMA-3, DeepSeek-V2, Qwen-2.5)
│   ├── plots.py                            # Publication-grade plotting routines (roofline, Pareto, heatmaps)
│   ├── quality.py                          # Quality table Q loader, interpolation & constraint gating
│   └── sram.py                             # On-chip SRAM hit rate h(C) under pinning, skew & sharing
│
├── scripts/                                # CLI runners, empirical benchmarks, and validation tools
│   ├── explore.py                          # Core thesis experiments (KV share, S*, CUDA roof, SRAM hit rate)
│   ├── layerwise_analysis.py               # Layer-by-layer roofline analyzer and LLM-Viewer reproduction
│   ├── measure_decode.py                   # HuggingFace decode latency (TPOT), peak memory & NVML energy
│   ├── measure_quality.py                  # Perplexity evaluation for simulated KV quantization
│   ├── measure_vllm.py                     # vLLM throughput benchmarking (FP8 and BF16 KV cache)
│   ├── ncu_hbm_bytes.sh                    # NVIDIA Nsight Compute shell profiler for DRAM traffic
│   ├── parse_ncu.py                        # Parser converting Nsight Compute logs to dram bytes CSV
│   ├── run_dse.py                          # Full design-space exploration CLI runner over context regimes
│   ├── sensitivity.py                      # Robustness analysis under ±30% to ±50% cost perturbations
│   └── validate.py                         # Empirical validation and MAPE evaluation against held-out data
│
├── extensions/                             # Advanced research studies and extended workstreams
│   ├── check_crossover.py                  # Analytical verification of crossover context S* on DeepSeek-V2
│   ├── cuda_ceiling_study.py               # Workstream 1: Secondary CUDA ceiling & bottleneck oscillation
│   ├── frontier_schemes.py                 # Workstream 2: 2025-2026 frontier compression schemes evaluation
│   ├── hypervolume_and_validation.py       # Workstream 3: LLM-Viewer cross-check, Pareto HV & quality audit
│   ├── inspect_dse.py                      # DSE structure, API signatures, and dataframe inspector
│   ├── sram_budget_sweep.py                # WS2 follow-up: Eviction budget vs pinned SRAM capacity sweep
│   └── outputs/                            # Artifacts, plots, CSVs, and summaries from workstreams
│       ├── cuda_ceiling_summary.md          # Findings on CUDA-core dequantization bottleneck flips
│       ├── exp1_cuda_thresholds.csv        # Critical dequantization FLOP thresholds
│       ├── exp1_dequant_sweep.csv          # Parameter sweep over dequantization overhead
│       ├── exp2_oscillation_map.csv        # Feasible (S, B) bottleneck boundary mappings
│       ├── exp3_dual_roofline.csv          # Dual-roofline arithmetic intensity coordinates
│       ├── exp4_first_compute_bound_B.csv  # Minimum batch size for compute-bound transitions
│       ├── exp4_large_batch.csv            # Large batch concurrency scaling
│       ├── inspect_dse.txt                 # Exported API signatures and dataframe column listings
│       ├── llm_viewer_crosscheck.tex       # LaTeX comparison table with LLM-Viewer baseline
│       ├── ws2_decomposition.csv           # Fixed-batch vs batch-expansion speedup decomposition
│       ├── ws2_full_comparison.csv         # Full evaluation of frontier compression schemes
│       ├── ws2_profiles_check.csv          # Effective compression ratio verification
│       ├── ws2_sram_budget_sweep.csv       # Working-set eviction budget vs SRAM sensitivity data
│       ├── ws2_summary.md                  # Frontier scheme recommendations by regime
│       ├── ws3_hypervolume_quality_matrix.csv # Hypervolume sensitivity across quality gates
│       ├── ws3_layerwise_crosscheck.csv    # Layerwise FLOP/byte crosscheck against independent formulas
│       ├── ws3_pareto_hypervolume.csv      # 3D Pareto hypervolume metrics (tokens/s, area, energy)
│       ├── ws3_quality_gate_audit.csv      # Empirical quality gate passing audits
│       └── *.png / *.pdf                   # Publication-grade figures (phase plots, heatmaps, Pareto)
│
├── results/                                # Output DSE sweeps, Pareto fronts, charts, and recommendations
│   ├── explore/                            # Core thesis exploratory plots (E1 crossover, E4 roofline)
│   ├── deepseek-v2-lite_h100/              # DSE outputs for DeepSeek-V2-Lite (MLA+MoE) on H100
│   ├── llama3-70b_h100/                    # DSE outputs for LLaMA-3-70B (8 chips, TP=8) on H100
│   ├── llama3.1-8b_h100/                   # DSE outputs for LLaMA-3.1-8B on H100 SXM
│   ├── llama3.1-8b_rtx4090/                # DSE outputs for LLaMA-3.1-8B on consumer RTX 4090
│   └── qwen2.5-7b_h100/                    # DSE outputs for Qwen-2.5-7B on H100 SXM
│       ├── A_short_medium_sweep.csv        # Full parameter grid evaluations (Regime A: ≤8k context)
│       ├── A_short_medium_pareto.csv       # Optimal 3D Pareto frontier points (throughput, area, energy)
│       ├── A_short_medium_roofline.png     # Roofline visualization showing operational points
│       ├── A_short_medium_knobs.png        # Marginal gain curves per intervention knob
│       ├── A_short_medium_sram_hbm_heatmap.png # SRAM capacity vs HBM bandwidth trade-off heatmap
│       ├── B_long_* / C_ultra_long_*       # Corresponding sweeps, Pareto CSVs & figures for Regimes B & C
│       └── RECOMMENDATIONS.md              # Sourced memory-hierarchy recommendations per context regime
│
└── tests/                                  # Unit test suite
    └── test_engine.py                      # 26 unit tests verifying physical invariants & deck equations
```

---

## 3. Detailed Component Breakdown

### 3.1 Core Modelling Engine (`kvroofline/`)

The core package implements analytical physics for LLM autoregressive token generation. It models execution time by splitting each decode step into two non-overlapping operational groups:

1. **Linear Group**: Model parameter weights read from memory vs. GEMV matrix-vector multiplications.
2. **Attention Group**: Accumulated KV-cache tensors read from memory vs. query-key and attention-value multiplications, softmax, and dequantization.

| File                                           | Primary Functions / Classes                                        | Responsibilities & Physics                                                                                                                                                                                                                                                                                                                                                        |
| ---------------------------------------------- | ------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`models.py`](kvroofline/models.py)           | `ModelConfig`, `get_model()`, `register_model()`             | Defines 6 models: `llama2-7b` (MHA baseline), `llama3.1-8b` (GQA), `qwen2.5-7b` (GQA), `llama3-70b` (GQA), `deepseek-v2-lite` (MLA+MoE), and `deepseek-v2` (MLA+MoE). Calculates active parameters, KV scalar counts per layer, and expected active MoE experts touched under batch size $B$: $E_{\text{active}} = E \cdot (1 - (1 - k/E)^B)$.                    |
| [`hardware.py`](kvroofline/hardware.py)       | `Hardware`, `get_hw()`, `register_hw()`                      | Hardware specifications for NVIDIA H100 SXM, H200, B200, A100-80, A100-40, L4, and RTX 4090. Derived properties include aggregate memory bandwidth $BW_{\text{HBM}}$, capacity, tensor-core peak $P_{\text{tc}}$, CUDA-core peak $P_{\text{cuda}}$, ridge points, and scaling knobs (`sram_pin_mb`, `hbm_stacks`, `hbm_speed_scale`, `n_chips`).                    |
| [`compression.py`](kvroofline/compression.py) | `Compression`, `metadata_overhead()`, `effective_kv_bytes()` | Models KV and weight compression. Tracks bit precision (16, 8, 4, 2), group quantization ($G$), per-group metadata ($4\text{ Bytes}$ per group for FP16 scale/zero), residual FP16 token windows, and eviction policies (`none`, `streaming`, `quest`). Implements dequantization compute requirements and Quest page-index reading overheads.                          |
| [`sram.py`](kvroofline/sram.py)               | `hit_rate()`, `weight_pinned_fraction()`                       | Models on-chip SRAM hit rate $h(C)$ for KV cache reads. Accounts for capacity pinning fraction $C/W$, access skew exponent $\theta$ (Zipfian/power-law attention locality), cross-sequence prefix sharing $\rho$, and Quest sparse footprint scaling.                                                                                                                     |
| [`engine.py`](kvroofline/engine.py)           | `evaluate()`, `max_batch()`, `crossover_S()`                 | Evaluates a single decode step and returns ~50 physical metrics. Computes group latencies $t = \max(\text{FLOPs}/P, \text{Bytes}/BW)$, communication latency for multi-chip tensor parallelism (AllReduce), capacity feasibility constraints, arithmetic intensity ($AI$), time per output token (TPOT), throughput (tokens/s), and bottleneck identification (`dominant`). |
| [`layerwise.py`](kvroofline/layerwise.py)     | `layerwise_breakdown()`, `layerwise_dataframe()`               | Calculates layer-by-layer FLOPs, memory transactions, and arithmetic intensities. Verified for exact parity against the reference LLM-Viewer methodology (arXiv:2402.16363).                                                                                                                                                                                                      |
| [`cost.py`](kvroofline/cost.py)               | `silicon_area_mm2()`, `energy_breakdown_pj()`                  | Sourced silicon area models (TSMC N5/N4 SRAM macro density of $3.90\text{ MB/mm}^2$, HBM3 PHY area of $14.0\text{ mm}^2/\text{stack}$, package substrate interposer overhead of $110\text{ mm}^2/\text{stack}$) and dynamic/leakage energy models ($0.10 + 0.02\sqrt{C_{\text{MB}}}\text{ pJ/bit}$ SRAM, $3.9\text{ pJ/bit}$ HBM3).                                     |
| [`quality.py`](kvroofline/quality.py)         | `QualityTable`, `quality_gate_passed()`                        | Loads empirical quality degradation data (`data/quality_table.csv`). Applies quality filtering constraints (e.g. dropping any intervention causing $>1.0\%$ relative perplexity degradation) before Pareto ranking.                                                                                                                                                           |
| [`dse.py`](kvroofline/dse.py)                 | `sweep()`, `pareto_front()`, `iso_area_best()`, `Regime`   | Multi-objective design-space exploration over parameter grids. Identifies non-dominated 3D Pareto frontiers across throughput, silicon area, and token energy. Evaluates iso-area trade-offs.                                                                                                                                                                                     |
| [`plots.py`](kvroofline/plots.py)             | `plot_roofline()`, `plot_pareto()`, `plot_heatmap()`         | Generates publication-ready Matplotlib figures, including dual-rooflines, crossover context curves $S^*$, Pareto frontiers, and SRAM $\times$ HBM heatmaps.                                                                                                                                                                                                                   |

---

### 3.2 Scripts & Toolchain (`scripts/`)

| Script                                                    | Purpose & Execution                                                                                                                                                                                                                                                                                                        |
| --------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`explore.py`](scripts/explore.py)                       | Executes the 4 foundational thesis experiments: (E1) KV-cache traffic share vs context length and analytical crossover $S^*$, (E2) CUDA-core roof binding hypothesis under INT4/INT2 dequantization, (E3) SRAM hit-rate sensitivity under access skew $\theta$, and (E4) Hierarchical two-group roofline plots.        |
| [`run_dse.py`](scripts/run_dse.py)                       | Main CLI entry point for full design-space exploration. Performs exhaustive grid sweeps over context regimes A ($8\text{k}$), B ($128\text{k}$), and C ($512\text{k}$), filters candidates against the empirical quality gate, calculates Pareto frontiers, and generates `RECOMMENDATIONS.md` and visualizations. |
| [`validate.py`](scripts/validate.py)                     | Fits hardware calibration parameters ($\text{bw\_eff}$, $t_{\text{fixed}}$) and computes Mean Absolute Percentage Error (MAPE) against held-out benchmark datasets (`data/measurements.csv` and `data/hbm_bytes.csv`).                                                                                             |
| [`layerwise_analysis.py`](scripts/layerwise_analysis.py) | Generates layer-by-layer breakdown of computational FLOPs and memory bytes. Cross-checks predictions directly against published LLM-Viewer baseline numbers.                                                                                                                                                               |
| [`sensitivity.py`](scripts/sensitivity.py)               | Conducts 11 parametric cost and technology perturbation studies ($\pm 30\%$ to $\pm 50\%$ in SRAM density, leakage, HBM PHY area, and interconnect overhead) to verify ranking robustness.                                                                                                                             |
| [`measure_decode.py`](scripts/measure_decode.py)         | Real GPU benchmarking script utilizing HuggingFace Transformers and PyNVML. Measures empirical decode latency (TPOT), peak VRAM memory, and NVML hardware energy per token.                                                                                                                                                |
| [`measure_vllm.py`](scripts/measure_vllm.py)             | Profiling runner using vLLM to measure empirical decode throughput and concurrency scaling with FP8 and BF16 KV caches.                                                                                                                                                                                                    |
| [`measure_quality.py`](scripts/measure_quality.py)       | Evaluates language model perplexity degradation under simulated KV cache quantization and token eviction algorithms.                                                                                                                                                                                                       |
| [`ncu_hbm_bytes.sh`](scripts/ncu_hbm_bytes.sh)           | Shell wrapper executing NVIDIA Nsight Compute (`ncu`) to capture low-level DRAM read/write throughput and memory transaction counters during decode kernels.                                                                                                                                                             |
| [`parse_ncu.py`](scripts/parse_ncu.py)                   | Extracts and parses `dram__bytes_read.sum` and `dram__bytes_write.sum` metrics from raw Nsight Compute CSV exports.                                                                                                                                                                                                    |

---

### 3.3 Advanced Research Workstreams (`extensions/`)

The `extensions/` directory contains secondary research studies, extended parameter sweeps, and validation cross-checks developed on top of the frozen `kvroofline` engine.

| Extension File                                                               | Research Focus & Deliverables                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`cuda_ceiling_study.py`](extensions/cuda_ceiling_study.py)                 | **Workstream 1: Secondary CUDA Ceiling & Bottleneck Oscillation**<br>Investigates whether INT4/INT2 KV quantization shifts decode execution from being memory-bandwidth bound to being compute-bound on FP32 CUDA cores due to dequantization overhead. Sweeps $c_{\text{deq}} \in [0, 16]\text{ FLOP/elem}$, maps bottleneck shifts across $(S, B)$ grids, and tracks "bottleneck oscillation" where weight reads re-emerge as the dominant bottleneck once KV bytes are suppressed. |
| [`frontier_schemes.py`](extensions/frontier_schemes.py)                     | **Workstream 2: 2025–2026 Frontier KV-Compression Schemes**<br>Evaluates compression techniques: AATC transform coding, PM-KVQ progressive mixed precision, Expected Attention, StreamingLLM, Quest, KIVI INT2/INT4, and KVQuant. Decomposes performance gains into fixed-batch latency reductions versus batch concurrency expansion enabled by KV memory savings. |
| [`hypervolume_and_validation.py`](extensions/hypervolume_and_validation.py) | **Workstream 3: Validation, Hypervolume & Quality-Gate Audit**Performs 3-way Pareto hypervolume integration across throughput, silicon area, and energy. Conducts layerwise cross-checks against independent formula implementations and audits the empirical quality gate across multiple tolerance thresholds ($0.25\% \to 5.0\%$).                                                                                                                                               |
| [`sram_budget_sweep.py`](extensions/sram_budget_sweep.py)                   | **Workstream 2 Follow-up: SRAM vs Eviction Working Set**Analyzes the interaction between token eviction budget and on-chip SRAM capacity, determining the exact eviction threshold below which on-chip SRAM pinning delivers meaningful speedups.                                                                                                                                                                                                                                     |
| [`check_crossover.py`](extensions/check_crossover.py)                       | Analytical verification script confirming the crossover context length $S^*$ on DeepSeek-V2 under single-chip and 8-chip tensor-parallel configurations.                                                                                                                                                                                                                                                                                                                                  |
| [`inspect_dse.py`](extensions/inspect_dse.py)                               | Introspection script printing functional signatures, schema attributes, and sample outputs of `kvroofline.dse`.                                                                                                                                                                                                                                                                                                                                                                           |
| [`outputs/`](extensions/outputs/)                                           | Stores all generated publication figures, summary tables (`cuda_ceiling_summary.md`, `ws2_summary.md`), LaTeX tables (`llm_viewer_crosscheck.tex`), and numerical CSV datasets from Workstreams 1, 2, and 3.                                                                                                                                                                                                                                                                          |

---

### 3.4 Empirical Data & Calibration (`data/`)

| File                                           | Content & Function                                                                                                                                                                                                                                          |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`quality_table.csv`](data/quality_table.csv) | Verified empirical quality degradation data compiled from published literature (KIVI, KVQuant, PM-KVQ, StreamingLLM, Quest). Maps quantization bits, group sizes, and eviction limits to relative perplexity / accuracy degradation across context lengths. |
| [`measurements.csv`](data/measurements.csv)   | Empirical dataset containing real profiled decode latency (TPOT), peak memory usage, and execution times across various GPU models (H100, RTX 4090, L4) and context regimes.                                                                                |
| [`hbm_bytes.csv`](data/hbm_bytes.csv)         | Ground-truth memory traffic captured via NVIDIA Nsight Compute (`ncu`) profiling across multiple batch sizes and context lengths. Used to validate the memory transfer formulas.                                                                          |
| [`calib.json`](data/calib.json)               | Calibrated hardware execution parameters, including empirical memory bandwidth efficiency ($\text{bw\_eff} = 0.6038$), tensor-core efficiency, CUDA-core efficiency, and fixed kernel launch overheads.                                                   |

---

### 3.5 Documentation (`docs/`)

| File                                                                               | Purpose                                                                                                                                                                                                                                             |
| ---------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`FINAL_REPORT.md`](docs/FINAL_REPORT.md)                                         | The definitive technical deliverables report. Documents governing physical equations, overhead terms, empirical validation MAPE, detailed findings across context regimes A, B, and C, and architectural recommendations for accelerator designers. |
| [`PROJECT_IDEA.md`](docs/PROJECT_IDEA.md)                                         | Plain-language conceptual summary explaining why LLM decode is memory-bound, detailing the 5 governing equations, the 3 intervention classes, and roadmap milestones.                                                                               |
| [`amcas_lit_rev.pdf`](docs/amcas_lit_rev.pdf)                                     | Literature review analyzing prior work on standard roofline models, multi-head latent attention (MLA), KV quantization algorithms, PagedAttention, and CXL memory expansion.                                                                        |
| [`roofline-kv-cache-intro copy.pptx`](<docs/roofline-kv-cache-intro%20copy.pptx>) | The team's 13-slide introductory presentation deck defining the original project brief, mathematical definitions, target models, and research gaps.                                                                                                 |

---

GPU-specific measurement scripts are available under `scripts/` and require the
optional dependencies in `requirements-gpu.txt`.

---

## 4. Context Regimes and Architectural Recommendations

The design-space exploration classifies inference into three distinct operational regimes based on the relationship between context length $S$ and the weight-KV crossover point $S^* = \frac{\text{WeightBytes}}{B \cdot \text{KVBytesPerToken}}$:

```
Context Length (S) ────────────────────────────────────────────────────────────────────────►
0               8k                               128k                             512k+
├────────────────┼────────────────────────────────┼─────────────────────────────────┤
│   Regime A     │            Regime B            │            Regime C             │
│ (Short/Medium) │             (Long)             │          (Ultra-Long)           │
│ S < S*         │ S > S*                         │ Capacity-Bound                  │
│ Weight-Bound   │ KV-Bound                       │ Single GPU Infeasible           │
└────────────────┴────────────────────────────────┴─────────────────────────────────┘
```

| Regime                            | Context Range            | Dominant Bottleneck                                                                                                      | Best Intervention                                                | Performance Gain                                                     | Area & Energy Impact                                                                              |
| --------------------------------- | ------------------------ | ------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------- | -------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| **Regime A** (Short/Medium) | $\le 8\text{k}$ tokens | **Weight-Bound** (`linear:hbm`)KV cache accounts for $<36\%$ of transferred bytes.                             | **Baseline SRAM + 6 HBM Stacks (H200-style) + FP8 KV**     | **$1.65\times - 1.92\times$** speedup                        | SRAM scaling yields negligible ROI ($<1.05\times$ for $263\text{ mm}^2$ die area).            |
| **Regime B** (Long)         | $128\text{k}$ tokens   | **KV-Bound** (`attn:hbm`)KV cache accounts for $77\%$ of memory traffic. Working set $\approx 52\text{ GB}$. | **INT4 g64 KV + Quest 16k Eviction + Baseline HBM**        | **$6.23\times$** speedup ($8.72\times$ with +2 HBM stacks) | $+0.66\%$ quality loss; $0\text{ mm}^2$ die area overhead; expands batch size $B: 3 \to 8$. |
| **Regime C** (Ultra-Long)   | $512\text{k}+$ tokens  | **Capacity-Bound**BF16 KV ($68.7\text{ GB/seq}$) exceeds physical memory of a single GPU.                        | **INT4 g64 KV + Quest Eviction + 2-GPU TP / 6 HBM Stacks** | **$11.1\times - 17.0\times$** speedup                        | $+0.89\%$ quality loss; strictly required for physical execution feasibility.                   |

---

## 5. Verification & Validation Metrics

The framework has been calibrated and verified against real hardware measurements:

* **HBM Traffic Accuracy**: **$\text{MAPE} = 3.3\%$** against NVIDIA Nsight Compute hardware DRAM counters.
* **Decode Latency Accuracy (TPOT)**: **$\text{MAPE} = 12.9\%$** against held-out real GPU decode measurements (surpassing the $\le 15\%$ course target).
* **Layerwise Consistency**: Exact agreement with LLM-Viewer (arXiv:2402.16363) per-layer totals for LLaMA-2-7B.
* **Unit Test Suite**: 26 unit tests in [`tests/test_engine.py`](tests/test_engine.py) verifying physical invariants (S* identity, memory bandwidth limits, Quest overheads, MoE expert activation, and multi-chip communication).
