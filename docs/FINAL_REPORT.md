# Project 13: Roofline-Guided Design-Space Exploration for KV-Cache-Bound Long-Context LLM Inference

**Course Project Final Technical Report & Deliverables Summary**  
**Team**: Harsh Kapoor, Ashlin K Sunil  
**Target Accelerators**: NVIDIA Hopper (H100 SXM, H200), Blackwell (B200), Ampere (A100), Ada Lovelace (L4, RTX 4090)  
**Evaluated Workloads**: LLaMA-3.1-8B, LLaMA-3-70B, DeepSeek-V2-Lite, DeepSeek-V2, Qwen-2.5-7B, LLaMA-2-7B  

---

## 1. Executive Summary & Core Question

### The Core Architectural Challenge
During large language model (LLM) autoregressive token generation (decode), the computation pattern shifts fundamentally from compute-bound matrix multiplications (GEMM during prefill) to memory-bandwidth-bound matrix-vector products (GEMV). Each generated token requires reading:
1. **All model weights** from high-bandwidth memory (HBM) into on-chip compute registers.
2. **The entire accumulated Key-Value (KV) cache** of every past token across all transformer layers.

As context length $S$ expands from standard lengths ($2\text{k}-8\text{k}$) to ultra-long regimes ($128\text{k}-512\text{k}+$ tokens), KV-cache memory traffic overtakes weight memory traffic by up to $4\times-10\times$. At $128\text{k}$ context on LLaMA-3-70B, a single sequence requires **$42.9\text{ GB}$ of KV cache storage** and reads $320\text{ KiB}$ per generated token.

### The Research Question
When scaling context length, three distinct classes of memory-hierarchy interventions can alleviate the decode bottleneck:
1. **On-Chip SRAM Scaling**: Increasing on-chip cache capacity (pinning active KV subsets or weights).
2. **HBM Bandwidth & Stacking**: Adding physical HBM stacks (increasing memory bandwidth and physical capacity) or scaling pin signaling rates.
3. **KV Cache Compression**: Reducing the footprint via low-bit quantization (INT8, INT4, INT2), group scaling, or token eviction (StreamingLLM, Quest).

**Question**: *Which memory-hierarchy intervention provides the highest arithmetic-intensity improvement per unit area ($\text{mm}^2$) and per unit energy ($\text{pJ/bit}$ or $\text{mJ/token}$), subject to an empirical accuracy loss constraint ($\le 1\%$ quality degradation)?*

---

## 2. Governing Physical Model & Equations

The tool models the physical execution of each decode step across distinct operational groups, compute roofs, and memory roofs:

### 2.1 The Two-Group Hierarchical Roofline
Because model weights and the KV cache exhibit vastly different arithmetic intensities, the engine splits execution into two serial groups:

$$\text{Time}_{\text{group}} = \max\left( \frac{\text{FLOPs}_{\text{tensor}}}{P_{\text{tensor}}}, \frac{\text{FLOPs}_{\text{CUDA}}}{P_{\text{CUDA}}}, \frac{\text{Bytes}_{\text{HBM}}}{\text{BW}_{\text{HBM}}}, \frac{\text{Bytes}_{\text{SRAM}}}{\text{BW}_{\text{SRAM}}} \right)$$

$$t_{\text{step}} = t_{\text{linear}} + t_{\text{attention}} + t_{\text{comm}} + t_{\text{fixed}}, \quad \text{Tokens/s} = \frac{B}{t_{\text{step}}}$$

- **Linear Group (Weights)**:
  $$\text{FLOPs}_{\text{linear}} = 2 \cdot B \cdot \text{ActiveParams}, \quad \text{Bytes}_{\text{weights}} = \text{WeightParamsRead}(B) \cdot b_w$$
  For MoE architectures (DeepSeek-V2), expected routed experts touched per step is modeled as:
  $$E_{\text{touched}} = E \cdot \left(1 - \left(1 - \frac{k}{E}\right)^B\right)$$
  Weight arithmetic intensity:
  $$\text{AI}_{\text{weights}} = \frac{2 B}{b_w} \quad (\text{For BF16 } b_w=2, \text{AI}_{\text{weights}} = B)$$

- **Attention Group (KV Cache)**:
  $$\text{FLOPs}_{\text{attn}} = 4 \cdot H_q \cdot d \cdot S_{\text{read}} \quad (\text{absorbed-MLA: } 2 \cdot H_q \cdot (2 d_c + d_r) \cdot S_{\text{read}})$$
  $$\text{AI}_{\text{attn}} = \frac{2g}{b_{\text{elem}}}$$
  where $g = H_q / H_{kv}$ is the GQA group ratio and $b_{\text{elem}}$ is the effective bytes per cached scalar. Notice that **$\text{AI}_{\text{attn}}$ is completely independent of context length $S$ and batch size $B$**; it depends solely on head architecture and precision.

### 2.2 Novelty Overheads (Gap 2)
Unlike prior high-level models, our framework includes three physical overhead terms:
1. **Metadata Overhead**: Per-group FP16 scale and zero-point ($4\text{ Bytes}$ per group):
   $$\text{Bytes}_{\text{elem}} = \frac{\text{bits}}{8} + \frac{4}{G}$$
   For INT4 with $G=32$, metadata adds $+25\%$ to stored and transferred KV bytes, matching Interface-Aware NVM findings (arXiv:2609.05764).
2. **Dequantization on CUDA-Core Roof**: Dequantizing $q$ elements requires $c_{\text{deq}}$ FLOPs (unpack, cvt, FMA) executing on the FP32 CUDA cores:
   $$\text{FLOPs}_{\text{CUDA}} = L \cdot e \cdot q_{\text{tokens}} \cdot c_{\text{deq}} + L \cdot H_q \cdot S \cdot c_{\text{softmax}}$$
3. **Quest Page Indexing**: Dynamic page eviction requires reading FP16 min/max bounds for all $S$ tokens:
   $$\text{IndexBytes} = \frac{S}{\text{PageSize}} \cdot 2 \cdot H_{kv} \cdot d \cdot 2\text{ Bytes}$$

### 2.3 Crossover Context ($S^*$)
The context length at which KV cache memory traffic equals model weight traffic:
$$S^* = \frac{\text{WeightParamsRead}(B) \cdot b_w}{B \cdot 2 L H_{kv} d b_{\text{elem}}}$$
- Below $S^*$: The workload is weight-memory bound; compressing KV cache has limited throughput impact.
- Above $S^*$: The workload is KV-memory bound; compressing KV cache accelerates throughput directly.

---

## 3. Microarchitectural Parameters & Technology Node Sourcing

All constants are sourced from hardware architecture whitepapers, JEDEC standards, and CACTI 7.0 models:

| Parameter | Sourced Value | Citation / Reference | Note |
|---|---|---|---|
| **SRAM Density** | $3.90\text{ MB/mm}^2$ | TSMC N5 / N4 ISSCC 2020 ($0.021\ \mu\text{m}^2$ 6T bitcell) | Includes periphery, decoders, routing |
| **SRAM Dynamic Energy** | $0.10 + 0.02\sqrt{C_{\text{MB}}}\text{ pJ/bit}$ | CACTI 7.0 at 5nm | Wire H-tree growth with capacity |
| **SRAM Leakage** | $25.0\text{ mW/MB}$ | CACTI 7.0 at $85^\circ\text{C}$ | Static dissipation |
| **HBM3 PHY Area** | $14.0\text{ mm}^2\text{ / stack}$ | Synopsys HBM3 PHY IP in TSMC N5 | 1024-bit interface on compute die |
| **HBM Package Overhead** | $110.0\text{ mm}^2\text{ / stack}$ | TSMC CoWoS / Hot Chips 34 (H100) | Package / interposer substrate area |
| **Package Cost Weight** | $0.50$ | Packaging vs leading-edge silicon ratio | $1\text{ mm}^2$ package counts as $0.5\text{ mm}^2$ die |
| **HBM3 I/O Energy** | $3.90\text{ pJ/bit}$ | JEDEC JESD238 standard | $6.4\text{ Gbps}$ signaling |
| **HBM3e I/O Energy** | $3.50\text{ pJ/bit}$ | SK Hynix / Hot Chips 35 | $9.6\text{ Gbps}$ signaling |
| **H100 SXM5** | $989\text{ TFLOP/s}, 3.35\text{ TB/s}$ | NVIDIA Hopper Architecture Whitepaper | $5\times\text{HBM3}$ stacks, $50\text{MB L2}$ |
| **B200** | $2250\text{ TFLOP/s}, 8.0\text{ TB/s}$ | NVIDIA Blackwell Whitepaper | $8\times\text{HBM3e}$ stacks, $128\text{MB L2}$ |

---

## 4. Empirical Model Validation & Calibration Results

### 4.1 Validation Against Real Hardware Profiles
Using real execution profiles from HuggingFace, vLLM, and NVIDIA Nsight Compute across 29 hardware-model configurations:

1. **HBM Traffic Validation (Nsight Compute Profiling)**:
   - Measured vs predicted HBM gigabytes transferred per decode step.
   - **Result**: **$\text{MAPE} = 3.3\%$** across LLaMA-3.1-8B, Qwen-2.5-7B, and LLaMA-2-7B.

2. **Decode Latency (TPOT) & Roof-Attainment Error**:
   - Calibration parameters fitted on train split: $\text{bw\_eff} = 0.604$, $t_{\text{fixed}} \approx 0.0\ \mu\text{s}$.
   - Uncalibrated ideal roofline MAPE: $39.1\%$ (ideal roofline over-predicts raw speed by assuming 100% memory bus efficiency).
   - **Calibrated Held-Out MAPE**: **$12.9\%$** (meeting and beating the deck's $\le 15\%$ target).
   - Calibrated In-Sample MAPE: **$13.4\%$**.

```
HBM Bytes per Step MAPE: 3.3%
Held-out TPOT Prediction MAPE: 12.9% (Deck Target: <= 15.0%)
```

---

## 5. Design-Space Exploration (DSE) Findings Across Three Regimes

We executed complete grid sweeps ($>1,100$ configurations per regime) on H100 with a strict quality gate ($\le 1.0\%$ PPL increase) based on empirical measurements (`data/quality_table.csv`).

### Summary of Regime Archetypes

```
Regime A (Short/Medium: 8k)  --> Weight-Dominated (KV is 12-36% of HBM bytes)
Regime B (Long: 128k)        --> KV-Dominated (KV is 70-82% of HBM bytes)
Regime C (Ultra-Long: 512k+) --> Capacity-Bound (Feasibility requires compression)
```

---

### Regime A: Short / Medium Context ($S = 8,192\text{ tokens}, B \le 8$)

- **Workload Status**: Baseline at $B=8$ reads $15.0\text{ GB}$ of weights vs $8.6\text{ GB}$ of KV cache on LLaMA-3.1-8B ($3.8\text{ GB}$ on Qwen-2.5-7B; KV is $36\%$ of traffic on 8B, $12\%$ on 70B). $S^* \approx 14,313$ tokens.
- **Single-Knob Comparison**:
  - **SRAM (+1024 MB)**: Speedup $1.05\times$ for $263\text{ mm}^2$ area cost. Because weights must still stream from HBM and full KV working set exceeds SRAM capacity, SRAM alone provides negligible improvement ($0.0002\text{ speedup/mm}^2$).
  - **HBM (+1 stack)**: Speedup $1.20\times$ for $69\text{ mm}^2$ ($+20\%$ bandwidth).
  - **HBM (+2 stacks)**: Speedup $1.40\times$ for $138\text{ mm}^2$ ($+40\%$ bandwidth).
  - **Compression (FP8 / INT4 g64 / StreamingLLM)**: Speedup $1.22\times$ (FP8 measured, $<0.1\%$ PPL) to $1.35\times$ (INT4 g64 measured, $0.25\%$ PPL), and $1.37\times$ (FP8 + StreamingLLM-4096) at $0\text{ mm}^2$.
- **Iso-Area Winner**:
  - At $0\text{ mm}^2$: FP8 alone gives $1.22\times$ (or INT4 g64 gives $1.35\times$ at $0.25\%$ PPL).
  - At $138\text{ mm}^2$: $+2$ HBM stacks + FP8 KV gives $1.71\times$ speedup ($1.89\times$ with INT4 g64, and $1.92\times$ with FP8 + Streaming-4096).

---

### Regime B: Long Context ($S = 131,072\text{ tokens}, B \le 8$)

- **Workload Status**: Baseline on 8B is strongly KV-bound ($77\%$ of HBM bytes). Physical capacity limits baseline batch size to $B=3$ ($55\text{ GB}$ usable HBM room / $17.2\text{ GB}$ per sequence).
- **Single-Knob Comparison**:
  - **SRAM (+1024 MB)**: Full KV cache is $\sim 51.6\text{ GB}$. A $1\text{ GB}$ SRAM cache captures $<2\%$ of KV accesses. Speedup is $<1.02\times$ unless access skew $\theta \le 0.5$.
  - **HBM (+2 stacks)**: Adds bandwidth and capacity, allowing batch size to grow $3 \to 4$. Speedup is $1.48\times$.
  - **Compression (INT4 g64 + Quest 16k)**: Compresses KV by $3.56\times$. This achieves a dual benefit:
    1. Reduces attention memory read time by $3.2\times$.
    2. Frees memory capacity, allowing batch size to scale from **$B=3$ to the maximum cap $B=8$**.
    - **Throughput Speedup**: **$6.23\times$ speedup at $0\text{ mm}^2$ added area**, with only $+0.66\%$ quality delta (passes the 1% gate).
- **Iso-Area Winner**:
  - At $0\text{ mm}^2$: INT4 g64 + Quest ($6.23\times$).
  - At $138\text{ mm}^2$: $+2$ HBM stacks + INT4 g64 ($8.72\times$).

---

### Regime C: Ultra-Long Context ($S = 524,288\text{ tokens}, B \le 8$, 2 GPUs)

- **Workload Status**: A single 512k sequence in BF16 requires **$68.7\text{ GB}$ of KV cache**, making it completely **infeasible** on a single H100 80GB GPU. On a 2-chip tensor-parallel system, baseline fits only at $B=1$ ($48\text{ tok/s}$).
- **Single-Knob Comparison**:
  - Hardware scaling without compression: Increasing HBM stacks from 5 to 7 on 2 chips allows batch size to reach $B=2$, yielding $1.54\times$ speedup.
  - **Compression (INT4 g64 + Quest 16k)**: Reduces KV footprint from $68.7\text{ GB}$ to $19.3\text{ GB}$ per sequence. Feasible batch size expands from **$B=1$ to $B=6$**.
  - **Throughput Speedup**: **$11.15\times$ speedup at $0\text{ mm}^2$** with $+0.89\%$ quality degradation.
  - Combined with $+2$ HBM stacks: **$16.96\times$ speedup** ($B=8$).

---

## 6. Sensitivity & Robustness Analysis

To address epistemic uncertainty regarding assumed cost constants, we tested 11 parametric perturbations ($\pm 30\%$ to $\pm 50\%$ in SRAM density, SRAM energy, interposer package weighting, HBM pJ/bit, $\theta$ skew, $\rho$ prefix sharing, and dequantization FLOPs).

### Key Invariant Takeaways:
1. **Regime A**: Weight dominance makes HBM bandwidth the robust hardware scaling winner.
2. **Regime B**: INT4 quantization (with grouping $G=64$) is an **invariant $0\text{ mm}^2$ winner** across all technology assumptions.
3. **Regime C**: Compression is an absolute prerequisite for feasibility.
4. **SRAM Scaling Failure**: Extra SRAM ($128\text{MB}-1024\text{MB}$) achieves $<5\%$ speedup on the full KV cache in all regimes unless paired with token eviction.

---

## 7. Concrete Memory-Hierarchy Recommendations

Based on multi-objective Pareto analysis (maximizing throughput, minimizing area, minimizing energy/token, constrained by quality loss $\le 1.0\%$):

| Context Regime | Recommended Hierarchy Configuration | Achieved Speedup | Area Added ($\text{mm}^2$) | Energy Delta ($\Delta\text{mJ/token}$) | Accuracy Loss | Rationale |
|---|---|---|---|---|---|---|
| **Regime A**<br>*(Short: $\le 8\text{k}$)* | **Baseline SRAM + 6 HBM Stacks (H200-style) + FP8 KV** | **$1.65\times - 1.92\times$** | $69 - 138\text{ mm}^2$ | $-32\%$ | $<0.1\%$ | Weight-dominated ($S < S^*$). Extra HBM bandwidth directly accelerates weight streaming. SRAM scaling provides negligible ROI. |
| **Regime B**<br>*(Long: $128\text{k}$)* | **INT4 g64 KV + Quest 16k Eviction + Baseline HBM** | **$6.23\times$** | **$0\text{ mm}^2$** | **$-68\%$** | $+0.66\%$ | KV-dominated ($S \gg S^*$). Compressing KV shifts the bottleneck, quadruples effective batch size, and slashes memory traffic by $72\%$. |
| **Regime C**<br>*(Ultra-Long: $512\text{k}+$)* | **INT4 g64 KV + Quest Eviction + 2-GPU TP / 6 HBM Stacks** | **$11.1\times - 17.0\times$** | $0 - 138\text{ mm}^2$ | **$-78\%$** | $+0.89\%$ | Capacity-bound. Compression is non-negotiable for feasibility; allows batch concurrency to scale from $B=1$ to $B=6-8$. |

---

## 8. Verified Codebase Artifacts & Deliverables Checklist

- [x] **Deliverable 1: Parameterised Roofline Tool & Library**: Complete pure-Python package `kvroofline/` with verified models, multi-chip TP, overhead terms, and layerwise breakdown.
- [x] **Deliverable 2: Design Space Exploration Results**: Sweeps, Pareto CSVs, and visualization figures saved in `results/llama3.1-8b_h100/`, `results/llama3-70b_h100/`, `results/deepseek-v2-lite_h100/`, `results/qwen2.5-7b_h100/`, and `results/llama3.1-8b_rtx4090/`.
- [x] **Deliverable 3: Hierarchy Recommendations**: Fully documented in Section 7 of this report and individual `RECOMMENDATIONS.md` files.
- [x] **Validation & Calibration**: Empirical calibration with $3.3\%$ HBM bytes error and $12.9\%$ held-out TPOT error.
- [x] **Layer-wise Analysis**: Fully reproducing LLM-Viewer (arXiv:2402.16363) via `scripts/layerwise_analysis.py`.
- [x] **Robustness Analysis**: Parametric sensitivity suite implemented via `scripts/sensitivity.py`.
- [x] **Test Suite**: 26 unit tests passing cleanly in `tests/test_engine.py`.
