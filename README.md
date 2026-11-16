# kvroofline: Roofline-Guided Design-Space Exploration for KV-Cache-Bound Long-Context LLM Inference

[![Tests](https://img.shields.io/badge/tests-26%20passed-brightgreen.svg)](tests/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](requirements.txt)
[![Validation](https://img.shields.io/badge/HBM%20MAPE-3.3%25-success.svg)](data/)
[![TPOT](https://img.shields.io/badge/TPOT%20MAPE-12.9%25-success.svg)](data/)

> **Core Research Question**: Which memory-hierarchy intervention buys the most arithmetic intensity improvement per unit silicon die area ($\mathrm{mm}^2$) and energy ($\mathrm{pJ/bit}$ or $\mathrm{mJ/token}$): more on-chip SRAM, more HBM bandwidth, or algorithmic KV-cache compression?

---

## 1. Executive Summary

During autoregressive token generation (decode) in Large Language Models (LLMs), the execution pattern shifts fundamentally from compute-bound matrix multiplications (GEMM during prefill) to memory-bandwidth-bound matrix-vector products (GEMV). Each newly generated token requires fetching:
1. **All model parameter weights** from High-Bandwidth Memory (HBM) into compute registers (amortized across batch size $B$).
2. **The entire accumulated Key-Value (KV) cache** of every past token across all attention layers for each sequence in the batch.

As context length ($S$) expands from conventional windows ($2\,\mathrm{k}$--$8\,\mathrm{k}$) to long-context regimes ($128\,\mathrm{k}$--$512\,\mathrm{k}+$ tokens), KV-cache memory traffic outstrips weight traffic by up to $4\times$--$10\times$. For example, at $128\,\mathrm{k}$ context on LLaMA-3-70B, a single sequence requires **$42.9\,\mathrm{GB}$ of KV cache storage** in BF16 and streams $320\,\mathrm{KiB}$ per generated token. Under long-context generation, modern accelerators operate at single-digit arithmetic intensities ($1$--$8\,\mathrm{FLOP/Byte}$), stranding over 90% of peak compute units in memory stall cycles.

**`kvroofline`** is a parameterised, two-group hierarchical roofline evaluation and design-space exploration (DSE) framework. Validated against low-level NVIDIA Nsight Compute hardware counters and real GPU latency measurements across 29 hardware-model configurations, our framework resolves the trade-offs between memory-hierarchy investments across diverse context regimes.

---

## 2. Key Empirical Findings & Recommendations

The framework classifies inference into three distinct operational regimes based on the relationship between context length $S$ and the weight-KV crossover point $S^* = \frac{\mathrm{WeightBytes}}{B \cdot \mathrm{KVBytesPerToken}}$:

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

| Regime | Context Range | Dominant Bottleneck | Best Intervention | Performance Gain | Silicon Area Overhead | Quality Impact |
|---|---|---|---|---|---|---|
| **Regime A** (Short/Medium) | $\le 8\,\mathrm{k}$ tokens | **Weight-Bound** (`linear:hbm`)<br>KV cache accounts for $<36\%$ of memory traffic. | **Baseline SRAM + 6 HBM Stacks (H200-style) + FP8 KV** | **$1.71\times - 1.92\times$** speedup | $+138\,\mathrm{mm}^2$ (HBM PHY + interposer) | $+0.02\%$ PPL loss (lossless) |
| **Regime B** (Long) | $128\,\mathrm{k}$ tokens | **KV-Bound** (`attn:hbm`)<br>KV cache accounts for $77\%$ of memory traffic ($52\,\mathrm{GB}$). | **INT4 g64 KV + Quest 16k Eviction + Baseline HBM** | **$6.23\times$** speedup ($8.72\times$ w/ +2 HBM) | **$0\,\mathrm{mm}^2$** (pure software) | $+0.66\%$ PPL loss (passes $\le 1.0\%$ gate) |
| **Regime C** (Ultra-Long) | $512\,\mathrm{k}+$ tokens | **Capacity-Bound**<br>BF16 KV ($68.7\,\mathrm{GB/seq}$) cannot fit on 1 GPU. | **INT4 g64 KV + Quest 16k + 2-GPU Tensor Parallelism** | **$11.15\times$** speedup ($16.96\times$ w/ +2 HBM) | **$0\,\mathrm{mm}^2$** (pure software) | $+0.89\%$ PPL loss (passes $\le 1.0\%$ gate) |

### The Three Core Architectural Takeaways
1. **SRAM Scaling Alone Fails Against Unevicted KV Caches**: A large die area investment ($263\,\mathrm{mm}^2$) provides $1\,\mathrm{GB}$ of on-chip SRAM. At $128\,\mathrm{k}$ context, the KV working set is $\approx 52\,\mathrm{GB}$. Because each token streams the entire cache once, $1\,\mathrm{GB}$ caches $<2\%$ of reads, yielding a negligible **$<1.05\times$ speedup**. On-chip SRAM only delivers positive ROI when paired with token eviction (Quest/StreamingLLM) to shrink the working set into cache dimensions.
2. **Conditional CUDA-Core Ceiling**: When attention matrix multiplications run on Tensor Cores ($989\,\mathrm{TFLOP/s}$), only dequantization and softmax execute on the FP32 CUDA cores ($67\,\mathrm{TFLOP/s}$). Decode attention flips to CUDA-bound only when dequantization overhead exceeds:
   $$
   c_{\mathrm{deq}}^* \approx \mathrm{AI}_{\mathrm{ridge,CUDA}} \times b_{\mathrm{elem}}
   $$
   On NVIDIA H100 SXM, $c_{\mathrm{deq}}^* \approx 11.2\ \mathrm{FLOP/elem}$ for INT4 g64. Standard kernels ($c_{\mathrm{deq}} = 6$) remain memory-bound. However, for transform-coding schemes like AATC ($c_{\mathrm{deq}} = 12$) or on NVIDIA B200 ($\mathrm{AI}_{\mathrm{ridge,CUDA}} \le 10\ \mathrm{FLOP/B}$), execution collides with the CUDA ceiling, explaining sub-linear quantization returns on next-gen hardware.
3. **Compression Throughput Gains Are Concurrency Gains**: In Regime B ($128\,\mathrm{k}$), an 80GB GPU running BF16 runs out of memory at batch size $B=3$. Compressing the KV cache with INT4 g64 + Quest frees VRAM, scaling batch concurrency from **$B=3$ to $B=8$**. The $6.23\times$ speedup is a product of $3.31\times$ single-stream latency reduction and $1.88\times$ batch concurrency expansion.
4. **Co-Design Hypervolume Expansion**: Evaluating 3D Pareto hypervolume across Throughput, Area, and Energy reveals that Hardware-Software Co-Design achieves a verified **$+32.9\%$ to $+184.2\%$** hypervolume gain over the union of isolated hardware or software point designs.

---

## 3. Governing Physical Model & Equations

The framework evaluates execution across two serial operational groups:
$$
t_{\mathrm{step}} = t_{\mathrm{linear}} + t_{\mathrm{attention}} + t_{\mathrm{comm}} + t_{\mathrm{fixed}},
\qquad \mathrm{Tokens/s} = \frac{B}{t_{\mathrm{step}}}
$$

where each group latency is governed by compute and memory roofs:
$$
t_{\mathrm{group}} = \max\left(
\frac{\mathrm{FLOPs}_{\mathrm{tensor}}}{P_{\mathrm{tensor}}},
\frac{\mathrm{FLOPs}_{\mathrm{CUDA}}}{P_{\mathrm{CUDA}}},
\frac{\mathrm{Bytes}_{\mathrm{HBM}}}{BW_{\mathrm{HBM}}},
\frac{\mathrm{Bytes}_{\mathrm{SRAM}}}{BW_{\mathrm{SRAM}}}
\right)
$$

### The 5 Governing Equations
1. **Classical Roofline**: $P_{\mathrm{attainable}} = \min\left(P_{\mathrm{peak}}, BW_{\mathrm{mem}} \times \mathrm{AI}\right)$
2. **Hardware Ridge Points**:
   * H100 Tensor Core Ridge: $\frac{989\,\mathrm{TFLOP/s}}{3.35\,\mathrm{TB/s}} \approx 295\,\mathrm{FLOP/Byte}$
   * H100 FP32 CUDA Core Ridge: $\frac{67\,\mathrm{TFLOP/s}}{3.35\,\mathrm{TB/s}} \approx 20\,\mathrm{FLOP/Byte}$
3. **Weight Arithmetic Intensity**:
   $$
   \mathrm{AI}_{\mathrm{weights}} =
   \frac{2 \cdot N_{\mathrm{weights}} \cdot B}{N_{\mathrm{weights}} \cdot b_w}
   = \frac{2B}{b_w}
   \quad (\mathrm{BF16}: b_w=2,\ \mathrm{AI}_{\mathrm{weights}} = B)
   $$
4. **Attention Arithmetic Intensity**:
   $$
   \mathrm{AI}_{\mathrm{attn}} =
   \frac{4 \cdot d_k \cdot H_q}{2 \cdot d_k \cdot H_{kv} \cdot b}
   = \frac{2g}{b},
   \qquad g = \frac{H_q}{H_{kv}}
   $$

   $\mathrm{AI}_{\mathrm{attn}}$ is strictly invariant to context length $S$ and batch size $B$.
5. **Crossover Context Length ($S^*$ holdout)**:
   $$
   S^* = \frac{N_{\mathrm{params}} \cdot b_w}
   {B \cdot 2L \cdot H_{kv} \cdot d_k \cdot b}
   $$

   For LLaMA-3.1-8B at $B=8$, $S^* \approx 14{,}313\ \mathrm{tokens}$.

### Novelty Overheads Modeled
* **Metadata Inflation**: Per-group FP16 scale and zero-point ($4\,\mathrm{Bytes}$ per group $G$): $b_{\mathrm{elem}} = \frac{\mathrm{bits}}{8} + \frac{4}{G}$ ($+25\%$ overhead at INT4 $G=32$).
* **CUDA-Core Dequantization & Softmax**: $\mathrm{FLOPs}_{\mathrm{CUDA}} = L \cdot e \cdot q_{\mathrm{tokens}} \cdot c_{\mathrm{deq}} + L \cdot H_q \cdot S \cdot c_{\mathrm{softmax}}$.
* **Quest Dynamic Page Indexing**: $\mathrm{IndexBytes} = \frac{S}{\mathrm{PageSize}} \cdot 2 \cdot H_{kv} \cdot d_k \cdot 2\,\mathrm{Bytes}$.
* **Active MoE Routed Expert Touch**: $E_{\mathrm{touched}} = E \cdot \left(1 - \left(1 - \frac{k}{E}\right)^B\right)$.

---

## 4. Empirical Validation Highlights

| Metric | Measured Benchmark | Framework Prediction | Error (MAPE) | Status |
|---|---|---|---|---|
| **HBM Memory Traffic** | NVIDIA Nsight Compute (`ncu`) | $\mathrm{Bytes}_{\mathrm{HBM}}$ model | **$\mathbf{3.3\%}$** | Verified across batch & context sweeps |
| **Decode Latency (TPOT)** | Real GPU PyTorch / vLLM runs | Calibrated model ($bw_{\mathrm{eff}} = 0.604$) | **$\mathbf{12.9\%}$** | Beats $\le 15\%$ course requirement |
| **Layerwise Breakdown** | LLM-Viewer (arXiv:2402.16363) | `layerwise_dataframe()` | **$\mathbf{0.000\%}$** | Exact parity on KV bytes & dense weights |
| **Unit Test Invariants** | Analytical formulas & Deck invariants | `tests/test_engine.py` | **$26 / 26$ passed** | Invariants, capacity limits, MoE touch |

---

## 5. Repository Structure

```
KV-Roofline/
├── README.md                           # Master project documentation & findings
├── PRESENTATION.md                     # Complete slide-by-slide presentation briefing & defense Q&A
├── DIRECTORY_STRUCTURE.md              # Detailed structural blueprint
├── requirements.txt                    # Base dependencies (numpy, scipy, pandas, matplotlib)
├── requirements-gpu.txt                # GPU execution dependencies (torch, vllm, transformers)
│
├── kvroofline/                         # Core Python analytical modeling library
│   ├── __init__.py                     # Package entry point
│   ├── models.py                       # 6 LLM architectures (LLaMA-3.1, LLaMA-3, DeepSeek-V2, Qwen-2.5)
│   ├── hardware.py                     # Hardware specs (H100, H200, B200, A100, L4, RTX 4090)
│   ├── compression.py                  # Quantization, metadata bloat, eviction policies (Quest, Streaming)
│   ├── sram.py                         # SRAM hit rate h(C) under pinning, access skew & sharing
│   ├── engine.py                       # Two-group hierarchical roofline evaluation engine
│   ├── layerwise.py                    # Layer-by-layer breakdown matching LLM-Viewer
│   ├── cost.py                         # Sourced area (SRAM mm², HBM PHY) and energy (pJ/bit) models
│   ├── quality.py                      # Empirical quality table loader and gating logic
│   ├── dse.py                          # Multi-objective sweeps, 3D Pareto fronts & iso-area ranking
│   └── plots.py                        # Publication plotting routines
│
├── data/                               # Empirical datasets & calibration parameters
│   ├── calib.json                      # Calibrated execution parameters (bw_eff = 0.6038, t_fixed ~ 0)
│   ├── measurements.csv                # Real empirical decode latency (TPOT) benchmarking dataset
│   ├── hbm_bytes.csv                   # Nsight Compute profiled DRAM traffic ground truth
│   └── quality_table.csv               # Empirical quality degradation lookup table across contexts
│
├── scripts/                            # Core CLI runners and validation tools
│   ├── explore.py                      # Core thesis experiments (E1 crossover S*, E4 roofline)
│   ├── run_dse.py                      # Full design-space exploration CLI runner
│   ├── validate.py                     # Calibration fitting and held-out MAPE validation
│   ├── layerwise_analysis.py           # Per-layer breakdown comparing with LLM-Viewer
│   ├── sensitivity.py                  # 11-parameter technology perturbation analysis
│   ├── measure_decode.py               # HuggingFace decode latency (TPOT) & NVML energy profiling
│   ├── measure_vllm.py                 # vLLM throughput benchmarking (FP8 and BF16 KV)
│   ├── measure_quality.py              # Perplexity evaluation for simulated KV quantization
│   ├── ncu_hbm_bytes.sh                # NVIDIA Nsight Compute DRAM traffic profiling script
│   └── parse_ncu.py                    # Parser converting NCU output to CSV
│
├── extensions/                         # Advanced research workstreams
│   ├── cuda_ceiling_study.py           # Workstream 1: Secondary CUDA ceiling & bottleneck oscillation
│   ├── frontier_schemes.py             # Workstream 2: 2025-2026 frontier compression schemes evaluation
│   ├── sram_budget_sweep.py            # Workstream 2: Eviction budget vs pinned SRAM capacity sweep
│   ├── hypervolume_and_validation.py   # Workstream 3: 3D Pareto hypervolume indicator & quality audit
│   ├── check_crossover.py              # Analytical verification of crossover context S* on DeepSeek-V2
│   ├── inspect_dse.py                  # Introspection script for DSE signatures and dataframe schema
│   └── outputs/                        # Curated single-format outputs (PNG plots, CSV datasets, summaries)
│
├── results/                            # Full DSE sweeps and Pareto outputs across target hardware
│   ├── llama3.1-8b_h100/               # LLaMA-3.1-8B on NVIDIA H100 SXM5
│   ├── llama3-70b_h100/                # LLaMA-3-70B on 8x H100 (TP=8)
│   ├── qwen2.5-7b_h100/                # Qwen-2.5-7B on NVIDIA H100 SXM5
│   ├── deepseek-v2-lite_h100/          # DeepSeek-V2-Lite (MLA+MoE) on NVIDIA H100
│   ├── llama3.1-8b_rtx4090/            # LLaMA-3.1-8B on consumer NVIDIA RTX 4090
│   └── explore/                        # Foundational exploratory plots
│
├── docs/                               # Comprehensive reports, presentation, literature review, and paper
│   ├── FINAL_REPORT.md                 # Complete deliverables report & hardware recommendations
│   ├── FINAL_PAPER.tex                 # Publication-ready IEEEtran conference paper source
│   ├── FINAL_PAPER.pdf                 # Compiled PDF from FINAL_PAPER.tex
│   ├── references.bib                  # Sourced and corrected BibTeX citations
│   ├── PROJECT_IDEA.md                 # Conceptual summary and plain-language guide
│   ├── amcas_lit_rev.pdf               # Mid-submission systematic literature review
│   ├── roofline-kv-cache-intro copy.pptx # Introductory presentation slide deck
│   └── figures/                        # High-resolution figures for paper and presentation
│
└── tests/                              # Unit testing suite
    └── test_engine.py                  # 26 unit tests asserting physical invariants
```

---

## 6. Quick Start

### Installation & Environment Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q tests               # Runs 26 unit tests verifying physical invariants
```

### Running Core Experiments
```bash
# 1. Core thesis experiments (KV share, S*, CUDA roof, SRAM hit rate):
python scripts/explore.py

# 2. Layerwise breakdown & LLM-Viewer reproduction:
python scripts/layerwise_analysis.py --compare-llm-viewer

# 3. Model calibration & held-out MAPE evaluation:
python scripts/validate.py --meas data/measurements.csv --hbm data/hbm_bytes.csv --fit --holdout 4

# 4. Full Design-Space Exploration (DSE) sweep:
python scripts/run_dse.py --model llama3.1-8b --hw h100 --quality-csv data/quality_table.csv --calib data/calib.json
```

### Running Advanced Research Workstreams
```bash
# Workstream 1: Secondary CUDA ceiling & bottleneck oscillation
python extensions/cuda_ceiling_study.py

# Workstream 2: 2025-2026 frontier compression schemes & decomposition
python extensions/frontier_schemes.py

# Workstream 3: 3D Pareto hypervolume & LLM-Viewer cross-check
python extensions/hypervolume_and_validation.py
```

GPU-specific measurement scripts are available under `scripts/` and require the
optional dependencies in `requirements-gpu.txt`.
