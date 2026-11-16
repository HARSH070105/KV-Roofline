# Project Presentation Guide: KV-Roofline

**Project #13**: *Roofline-Guided Design-Space Exploration for KV-Cache-Bound Long-Context LLM Inference*  
**Team**: Harsh Kapoor, Ashlin K. Sunil, Samyak Soni, Allu Joanna Preetika  
**Date**: November 2026

---

## Quick Reference: 5 Core Numbers You Must Know Cold
1. **Validation Accuracy**: **$3.3\%$ MAPE** on HBM memory bytes (Nsight Compute) and **$12.9\%$ held-out MAPE** on decode latency (TPOT).
2. **H100 Ridges**: Tensor Core Ridge = **$295\text{ FLOP/B}$**; FP32 CUDA Core Ridge = **$20\text{ FLOP/B}$**.
3. **Weight vs Attention AI**: $\text{AI}_{\text{weights}} = \frac{2B}{b_w}$ (scales with batch); $\text{AI}_{\text{attn}} = \frac{2g}{b_{\text{elem}}}$ (**strictly invariant** to context and batch).
4. **Regime B Headline**: INT4 g64 KV + Quest 16k eviction yields **$\mathbf{6.23\times}$ speedup at $\mathbf{0\text{ mm}^2}$ added area** ($+0.66\%$ quality loss), expanding batch concurrency from $B=3 \to 8$.
5. **Hardware-Software Co-Design Gain**: **$+32.9\% \text{ to } +184.2\%$** hypervolume expansion over the union of isolated hardware and software point designs.

---

## Slide-by-Slide Presentation Blueprint

### Slide 1: Title & The Core Problem
* **Slide Title**: Roofline-Guided Design-Space Exploration for KV-Cache-Bound Long-Context LLM Inference
* **Visual**: Context scaling curve showing memory traffic shifting from weights to KV cache; idle Tensor Core utilization icon (<10%).
* **Key Bullet Points**:
  * Autoregressive LLM decode is memory-bandwidth bound (GEMV).
  * In long context ($128\text{k}$--$512\text{k}+$), the accumulated KV cache dwarfs model weights ($42.9\text{ GB/seq}$ on 70B).
  * Accelerators operate at single-digit arithmetic intensity ($1$--$8\text{ FLOP/Byte}$), stranding $>90\%$ of compute units.
  * **Core Question**: Which memory-hierarchy intervention buys the most arithmetic intensity per unit area ($\text{mm}^2$) and energy ($\text{pJ/bit}$): more SRAM, more HBM bandwidth, or KV compression?
* **Speaker Script**:
  > *"Good morning. Today we present our work on roofline-guided design-space exploration for long-context LLM decode. While prefill prompt processing saturates GPU Tensor Cores, token generation is severely memory-bandwidth bound. Every single token generated requires fetching all model weights and the entire accumulated KV cache history. As context lengths scale from 8k to 128k and 512k tokens, KV cache streaming overtakes weights by up to 10x, stranding over 90% of GPU compute units in idle stall cycles. Our project builds the first unified, hardware-aware framework that evaluates SRAM scaling, HBM bandwidth, and KV compression on a normalized silicon area and energy scale."*

---

### Slide 2: Identified Research Gaps in Prior Work
* **Slide Title**: Motivation: 4 Unresolved Research Gaps
* **Visual**: A 2x2 grid contrasting software-only papers vs hardware-only point designs.
* **Key Bullet Points**:
  * **Gap 1: Software-Hardware Disconnect**: Software papers (KIVI, Quest) benchmark on fixed commercial GPUs; hardware papers evaluate static point designs.
  * **Gap 2: Missing Physical Overheads**: Prior tools (e.g. LLMCompass) omit quantization metadata bloat ($+20\%$--$25\%$), CUDA-core dequantization FLOPs, and page-indexing costs.
  * **Gap 3: Proliferation of Point Designs**: PIM and custom accelerators lack continuous sensitivity curves.
  * **Gap 4: Regime-Blind Prescriptions**: Solutions that work at $4\text{k}$ context fail at $128\text{k}$, requiring regime-dependent co-design.
* **Speaker Script**:
  > *"When surveying existing literature, we identified four major gaps. First, a deep disconnect exists between software and hardware: algorithmic papers benchmark compression on static GPUs, while hardware papers evaluate bespoke accelerators without normalizing against die area or energy. Second, existing modeling tools like LLMCompass omit critical real-world overheads: metadata inflation, which adds 25% to stored bytes at INT4, and runtime dequantization on CUDA cores. Third, PIM architectures are static point designs. Finally, prescriptions in literature have been regime-blind: what works at 4k tokens fails completely at 128k. We set out to bridge these gaps."*

---

### Slide 3: Theoretical Formulation: Two-Group Hierarchical Roofline
* **Slide Title**: Governing Analytical Model & Overhead Formulations
* **Visual**: The two-group serial execution diagram: Linear Group (weights) + Attention Group (KV cache) with Tensor, CUDA, and HBM roofs.
* **Key Bullet Points**:
  * Serial execution: $t_{\text{step}} = t_{\text{linear}} + t_{\text{attention}} + t_{\text{comm}} + t_{\text{fixed}}$.
  * $t_{\text{group}} = \max\left( \frac{\text{FLOPs}_{\text{tensor}}}{P_{\text{tensor}}}, \frac{\text{FLOPs}_{\text{CUDA}}}{P_{\text{CUDA}}}, \frac{\text{Bytes}_{\text{HBM}}}{BW_{\text{HBM}}}, \frac{\text{Bytes}_{\text{SRAM}}}{BW_{\text{SRAM}}} \right)$.
  * Attention intensity is strictly invariant: $\text{AI}_{\text{attn}} = \frac{2g}{b_{\text{elem}}}$ (independent of $S$ and $B$).
  * Crossover context length $S^* = \frac{N_{\text{params}} \cdot b_w}{B \cdot 2L \cdot H_{kv} \cdot d_k \cdot b}$ ($14,313\text{ tokens}$ on 8B, $B=8$).
  * Modeled overheads: FP16 metadata ($4\text{ B/group}$), CUDA-core dequantization ($c_{\text{deq}}$), Quest page indexing, and MoE active expert touch ($E_{\text{touched}} = E(1 - (1 - k/E)^B)$).
* **Speaker Script**:
  > *"Because weights and the KV cache have fundamentally different arithmetic intensities, we model decode execution as two serial groups: the Linear group and the Attention group. Each group is constrained by its own compute and memory roofs. Notice a fundamental mathematical property: attention arithmetic intensity, 2g over b, is completely invariant to context length and batch size. It depends only on head architecture and precision. We also explicitly derive the crossover context S*, the exact point where KV traffic exceeds weight traffic, and model physical overheads including metadata bloat, runtime dequantization on FP32 CUDA cores, Quest page indexing, and MoE expert touch."*

---

### Slide 4: Empirical Hardware Validation
* **Slide Title**: Empirical Calibration & Hardware Validation
* **Visual**: Predicted vs. Measured correlation scatter plot ([`docs/figures/fig_roofline.png`](docs/figures/fig_roofline.png)).
* **Key Bullet Points**:
  * **HBM Memory Traffic**: **$3.3\%$ MAPE** against NVIDIA Nsight Compute hardware DRAM counters across batch sizes and contexts.
  * **Decode Latency (TPOT)**: **$12.9\%$ held-out MAPE** against real GPU decode runs (calibrated $bw_{\text{eff}} = 0.604$, beating the $\le 15\%$ target).
  * **Layerwise Parity**: **$0.000\%$ error** on KV bytes and $<0.07\%$ on dense weights vs. reference LLM-Viewer formulas.
  * **Test Suite**: 26 automated unit tests validating all physical invariants and capacity constraints.
* **Speaker Script**:
  > *"To ensure our simulator reflects real hardware rather than idealized theory, we validated it against real GPU profiling across 29 configurations. Using NVIDIA Nsight Compute to capture raw DRAM byte transfers, our memory model achieves a 3.3% Mean Absolute Percentage Error. For decode latency, an uncalibrated ideal roofline has 39% error due to assuming 100% bus utilization. By fitting an empirical bus efficiency factor of 0.604 on our training set, we achieve 12.9% MAPE on held-out GPU measurements, easily beating our 15% target. Furthermore, our per-layer FLOP and byte breakdowns match published LLM-Viewer numbers with 0.000% error on KV bytes."*

---

### Slide 5: Multi-Regime Design-Space Exploration Results
* **Slide Title**: Architectural Recommendations Across Context Regimes
* **Visual**: The 3-regime comparison table (Regimes A, B, and C).
* **Key Bullet Points**:
  * **Regime A ($\le 8\text{k}$, Short/Medium)**: Weight-bound ($S < S^*$). Winner: **Baseline SRAM + 6 HBM Stacks (H200-style) + FP8 KV** ($1.71\times - 1.92\times$ speedup, $138\text{ mm}^2$). SRAM scaling gives $<1.05\times$ for $263\text{ mm}^2$.
  * **Regime B ($128\text{k}$, Long)**: KV-bound ($77\%$ of memory traffic). Winner: **INT4 g64 KV + Quest 16k Eviction + Baseline HBM** ($\mathbf{6.23\times}$ speedup at $\mathbf{0\text{ mm}^2}$, $+0.66\%$ quality loss). Unlocks batch concurrency $B: 3 \to 8$.
  * **Regime C ($512\text{k}+$, Ultra-Long, 2 GPUs)**: Capacity-bound. BF16 KV ($68.7\text{ GB/seq}$) is infeasible on 1 GPU. Winner: **INT4 g64 KV + Quest 16k + 2-GPU TP** ($\mathbf{11.15\times}$ speedup at $0\text{ mm}^2$, $\mathbf{16.96\times}$ with +2 HBM). Compression is strictly mandatory for execution.
* **Speaker Script**:
  > *"Here are our primary findings across the three context regimes. In Regime A, context is below S*, meaning weights dominate memory traffic. Here, SRAM expansion provides almost zero ROI, less than 1.05x speedup for 263 mm² of die area. Instead, adding HBM bandwidth and light FP8 quantization gives up to 1.92x speedup. In Regime B at 128k context, the KV cache becomes 77% of all memory traffic. Here, INT4 quantization with group size 64 combined with Quest dynamic page eviction delivers a massive 6.23x speedup at zero added die area while staying well within our 1% quality gate. In Regime C at 512k context, a single sequence requires 68.7 GB of KV cache in BF16, which is physically impossible to run on one GPU. Here, compression is not an optional speedup—it is strictly mandatory for physical feasibility, delivering 11.15x speedup on 2 GPUs and up to 16.96x when combined with 2 extra HBM stacks."*

---

### Slide 6: Three Core Architectural Takeaways
* **Slide Title**: Architectural Insights for Accelerator Designers
* **Visual**: Decomposition stacked bar chart ([`docs/figures/fig_decomposition.png`](docs/figures/fig_decomposition.png)) and phase oscillation map ([`docs/figures/fig_phase_oscillation.png`](docs/figures/fig_phase_oscillation.png)).
* **Key Bullet Points**:
  1. **Why SRAM Alone Fails**: $1\text{ GB}$ of SRAM ($263\text{ mm}^2$) caches $<2\%$ of a $52\text{ GB}$ working set. It only works if eviction shrinks the working set into SRAM first.
  2. **The Conditional CUDA Ceiling**: Critical threshold $c_{\text{deq}}^* \approx \text{AI}_{\text{ridge, CUDA}} \cdot b_{\text{elem}}$. Standard INT4 ($c_{\text{deq}}=6$) is safe on H100 ($c^*=11.2$), but binds for transform coding ($c_{\text{deq}}=12$) or on NVIDIA B200 ($\text{AI}_{\text{ridge}} \le 10$).
  3. **Capacity-Driven Batch Expansion**: Compressing KV footprint frees VRAM, expanding batch concurrency from $B=3 \to 8$. Most throughput gain is concurrency expansion, not just per-token latency.
* **Speaker Script**:
  > *"We highlight three critical architectural takeaways for hardware architects. First, why does on-chip SRAM scaling alone fail? Because token decode streams the whole cache once per token. A 1 GB SRAM cache covers less than 2% of a 52 GB working set. SRAM scaling only pays off when token eviction shrinks the working set first. Second, does low-bit quantization hit the CUDA-core compute roof? We derived the exact mathematical threshold: c_deq equals the CUDA ridge times bytes per element. On H100, INT4 attention remains safely memory-bound at 6 FLOPs per element. But on next-generation architectures like B200 where the CUDA ridge drops below 10, low-bit attention collides with the CUDA roof, producing identical throughput across bitwidths. Third, compression gains are primarily concurrency gains: shrinking the KV cache allows batch size to expand from 3 to 8, multiplying throughput while keeping latency bounded."*

---

### Slide 7: Research Extensions & Co-Design Hypervolume
* **Slide Title**: Workstreams 1--3 & Hardware-Software Co-Design
* **Visual**: 3D Pareto Hypervolume comparison bar chart ([`docs/figures/fig_hypervolume.png`](docs/figures/fig_hypervolume.png)).
* **Key Bullet Points**:
  * **Workstream 1 (Bottleneck Oscillation)**: BF16 ($23.6\%$ attn) $\to$ INT4 KV ($14.1\%$ attn, exposes weights) $\to$ INT4 KV + W8 ($28.4\%$ attn, swings back).
  * **Workstream 2 (Frontier Schemes)**: Benchmark of 2025--2026 schemes (AATC transform coding, PM-KVQ, Expected Attention). AATC is CUDA-bound at fixed batch ($2.18\times$).
  * **Workstream 3 (3D Hypervolume)**: Co-design expands the 3D Pareto hypervolume by **$+32.9\% \text{ to } +184.2\%$** over the union of isolated hardware and software point designs.
* **Speaker Script**:
  > *"In our extended research workstreams, we validated three deeper phenomena. In Workstream 1, we mapped bottleneck oscillation: compressing KV traffic drops attention's share of runtime to 14%, exposing weight reads as the primary bottleneck, while quantizing weights swings the bottleneck right back to attention. In Workstream 2, we evaluated 2025-2026 frontier schemes, proving that transform coding like AATC becomes CUDA-bound due to transform overhead. Finally, in Workstream 3, we computed 3D Pareto hypervolume indicators across Throughput, Die Area, and Energy. Hardware-Software Co-Design achieves a 33% to 184% hypervolume expansion over the union of isolated hardware or software designs, proving that memory hierarchy and compression must be designed together."*

---

### Slide 8: Conclusion & Summary
* **Slide Title**: Conclusion & Future Work
* **Visual**: Summary recommendation matrix and repo QR / GitHub link.
* **Key Bullet Points**:
  * Built, calibrated, and validated the first two-group hierarchical roofline framework for long-context LLM decode.
  * Verified to $3.3\%$ MAPE on memory traffic and $12.9\%$ MAPE on decode latency.
  * Demonstrated that memory hierarchy investments must be regime-aware: HBM bandwidth in short contexts, INT4 + eviction in long contexts, and multi-GPU tensor parallelism in ultra-long contexts.
  * Publication-ready IEEE conference paper and open-source codebase completed.
* **Speaker Script**:
  > *"In conclusion, we have built and validated an analytical framework that resolves the memory-hierarchy trade-offs for long-context LLM inference. With 3.3% MAPE on memory traffic and 12.9% on decode latency, our framework provides clear, actionable guidance: scale HBM bandwidth for short contexts, deploy INT4 quantization with dynamic eviction for long contexts, and combine compression with multi-GPU tensor parallelism for ultra-long horizons. Thank you, and we welcome your questions."*

---

## Defense Q&A: 6 Toughest Questions & Expert Answers

#### Q1: "Why does an uncalibrated roofline have 39% error, and is a 12.9% error acceptable?"
* **Answer**: *"The theoretical roofline assumes 100% bus utilization and zero driver latency. On real hardware, memory controllers, bank conflicts, DRAM burst alignment, and CUDA kernel launch overheads cap real-world bus efficiency to around 60% of theoretical peak. By fitting a single empirical bandwidth efficiency parameter ($bw_{\text{eff}} = 0.604$) on training data, our model achieved 12.9% held-out MAPE on unseen configurations, easily meeting our course milestone requirement of $\le 15\%$."*

#### Q2: "Why does token eviction leave attention arithmetic intensity unchanged?"
* **Answer**: *"Attention arithmetic intensity is $\text{FLOPs} / \text{Bytes}$. When evicting tokens to a budget $S_{\text{budget}}$, attention computation scales as $4 \cdot d_k \cdot H_q \cdot S_{\text{budget}}$ and memory read scales as $2 \cdot d_k \cdot H_{kv} \cdot b \cdot S_{\text{budget}}$. The $S_{\text{budget}}$ terms cancel out in the ratio, leaving $\text{AI}_{\text{attn}} = 2g/b$, exactly identical to dense attention. Eviction moves execution down the bandwidth roof, reducing latency and freeing VRAM capacity, rather than moving execution to the right."*

#### Q3: "Why doesn't INT4 attention hit the CUDA-core roof on NVIDIA H100?"
* **Answer**: *"Because attention matrix multiplications ($QK^T$ and $AV$) run on Tensor Cores ($989\text{ TFLOP/s}$), not CUDA cores. Only unpacking, scale multiplication, and softmax run on FP32 CUDA cores ($67\text{ TFLOP/s}$). At standard dequantization overhead ($c_{\text{deq}} = 6\text{ FLOP/element}$), CUDA execution time is only $72\%$ of HBM read time. Attention only flips to CUDA-bound if dequantization costs exceed $11.2\text{ FLOP/elem}$ (such as in transform coding) or on B200 where the CUDA ridge drops below $10\text{ FLOP/B}$."*

#### Q4: "Why do you treat quality as a constraint instead of an objective in the Pareto front?"
* **Answer**: *"If quality were a standard objective, extreme but unusable configurations—such as 2-bit quantization with a 20% perplexity explosion—would appear on the Pareto front simply because they have high throughput and zero added die area. In real production serving, accuracy degradation must be strictly bounded. By applying an empirical quality gate ($\Delta\text{PPL} \le 1.0\%$) before Pareto extraction, we guarantee that all recommended configurations are deployable."*

#### Q5: "How did you model the area of an extra HBM stack?"
* **Answer**: *"We sourced the physical silicon area from Synopsys TSMC N5 datasheets, which report an HBM3 PHY area of $14.0\text{ mm}^2$ per 1024-bit interface. In addition, an extra HBM stack requires TSMC CoWoS interposer and substrate allocation of $110.0\text{ mm}^2$. Because interposer packaging costs roughly half as much as high-density silicon die area per unit area, we apply a standard $0.50\times$ packaging weight, giving an effective area cost of $14.0 + 0.5 \times 110.0 = 69.0\text{ mm}^2$ per additional HBM stack."*

#### Q6: "Why is Hardware-Software Co-Design better than just applying compression in software?"
* **Answer**: *"Software compression alone cannot change memory bus bandwidth, and hardware scaling alone cannot bypass physical VRAM capacity limits without prohibitive die area costs. In Regime B, INT4 compression plus eviction unlocks batch expansion ($B: 3 \to 8$), but adding $+2$ HBM stacks increases bandwidth, scaling speedup from $6.23\times$ to $8.72\times$. Our 3D Pareto hypervolume calculation shows that co-design achieves a $33\%$ to $184\%$ hypervolume gain over the union of isolated hardware or software point designs."*
