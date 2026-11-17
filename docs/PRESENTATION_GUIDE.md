# KV-Roofline Presentation Guide

This guide accompanies `KV_Roofline_10_Slide_Deck.pptx`. It explains the project in
simple terms and provides a speaking guide for a presentation of approximately
10 minutes.

## 1. The project in one sentence

KV-Roofline studies why long-context large language model (LLM) inference becomes
slow and expensive, then uses a roofline model to identify whether hardware
changes, KV-cache compression, token eviction, or multi-GPU scaling will help
most.

## 2. Basic background

### What is an LLM?

An LLM generates text one token at a time. A token may be a word, part of a word,
or punctuation. During generation, the model repeatedly predicts the next token
using all the information it has already processed.

### What are prefill and decode?

- **Prefill** processes the input prompt. Many tokens can be processed together,
  so the GPU can perform large matrix operations efficiently.
- **Decode** generates the output. Usually one new token is generated at a time,
  and the model must repeatedly access its weights and previous attention data.

The project focuses on **decode**, especially when the prompt and generated
history are very long.

### What is the KV cache?

Attention uses two intermediate values for previous tokens:

- **K (key)**: information used to compare the current token with previous tokens.
- **V (value)**: information retrieved after those comparisons.

The model stores these values in the **KV cache** so it does not recompute them
for every new token. The cache grows as the context grows.

At long context lengths, the KV cache can become very large. For example, a
70-billion-parameter model at 128k context can require approximately 42.9 GB of
BF16 KV storage for one sequence.

## 3. The central problem

The main problem is not only the number of arithmetic operations. It is the
amount of data that must move between memory and the compute units.

Modern GPUs contain powerful Tensor Cores and CUDA cores, but those units may
wait for data when memory bandwidth is the limiting resource. This is called a
**memory-bound** workload.

During decode, the system may need to:

1. Read model weights from high-bandwidth memory (HBM).
2. Read the accumulated KV cache.
3. Perform attention and other calculations.
4. Repeat the process for every generated token.

As context length increases, the KV cache read can become larger than the weight
read. At that point, improving compute throughput alone does not solve the
problem.

## 4. Important terms

### HBM

**High-Bandwidth Memory (HBM)** is fast external memory attached to the GPU.
It stores model weights and KV-cache data. More HBM bandwidth can help when the
workload is limited by memory transfers, but adding HBM may require additional
physical area and packaging resources.

### SRAM

**SRAM** is small, fast on-chip memory. It can reduce some external-memory
traffic when the requested data fits in the cache. However, a small SRAM cache
does not automatically solve a large streaming KV-cache workload.

### Arithmetic intensity

Arithmetic intensity is:

```text
arithmetic intensity = useful FLOPs / bytes moved
```

Low arithmetic intensity means that the system moves a lot of data for relatively
few operations. Decode commonly operates at low arithmetic intensity and is
therefore vulnerable to memory-bandwidth limits.

### Roofline model

The roofline model compares the performance available from compute units with
the performance available from memory bandwidth. The lower limit determines the
actual performance.

In this project, the model considers:

- Tensor-Core compute,
- CUDA-core compute,
- HBM bandwidth,
- SRAM bandwidth,
- communication overhead,
- fixed runtime overheads.

### Quantization

Quantization stores values with fewer bits. For example, INT4 uses four bits per
value instead of the 16 bits used by BF16. This reduces storage and memory
traffic, but it can introduce accuracy loss and additional dequantization work.

### Token eviction

Token eviction removes or deprioritizes some old KV-cache entries while keeping
the most useful information. Quest is an example of a page-based eviction
strategy. Eviction reduces the active working set and can increase the feasible
batch size.

### Batch size

The batch size is the number of sequences processed together. A smaller KV
footprint can allow more sequences to fit in GPU memory. This means compression
can improve throughput not only by reducing bytes per token, but also by
increasing concurrency.

## 5. Slide-by-slide speaking guide

### Slide 1 — Introduction

Introduce the title, team, and research topic. Explain that the project asks how
to make long-context LLM generation more efficient. State the main message:
there is no single best optimization for every context length.

### Slide 2 — What changes during inference?

Explain prefill and decode in plain language. Prefill handles many prompt tokens
at once. Decode generates one token at a time and repeatedly reads the model
weights and KV cache. Define the KV cache before discussing why it becomes a
problem.

### Slide 3 — Bottlenecks

Introduce arithmetic intensity as work divided by data movement. Explain the
three main bottlenecks:

1. Weight streaming at shorter contexts.
2. KV-cache streaming at longer contexts.
3. GPU memory capacity and CUDA-core dequantization overhead.

The crossover point, written as `S*`, is where KV traffic becomes larger than
weight traffic.

### Slide 4 — What the project did

Describe the project pipeline:

1. Model the workload.
2. Represent hardware and memory limits.
3. Explore compression, eviction, SRAM, HBM, and scaling choices.
4. Compare predictions with measurements.

Mention the main validation results: 3.3% HBM traffic MAPE, 12.9% held-out
TPOT MAPE, and 26 passing tests.

### Slide 5 — Main methodology

This slide explains the analytical model used to estimate decode time. The
important modeling choice is to separate the decode step into two groups with
different data dependencies:

- The **linear group** contains the model's feed-forward and projection layers.
  These operations repeatedly read the model weights. Increasing batch size
  allows the same weights to be reused across more sequences, so the arithmetic
  intensity of the weight path generally increases with batch size.
- The **attention group** contains the query-key and attention-value operations.
  These operations read the KV cache. Their traffic grows with the number of
  tokens retained in the context and with the number of active sequences.
- **Communication and fixed costs** represent tensor-parallel communication,
  kernel launches, synchronization, and other work that is not captured by the
  two main memory streams.

The model adds the group times:

```text
t_step = t_linear + t_attention + t_communication + t_fixed
```

For each group, the time is the maximum of the relevant compute and memory
times. In other words, the group is limited by whichever resource finishes last:

```text
t_group = max(Tensor-Core time, CUDA-core time,
              HBM time, SRAM time)
```

This is the key reason the model is more useful than a FLOP-only estimate. A
configuration can have enough arithmetic capacity but still be slow because
weights or KV data cannot arrive from HBM quickly enough. Conversely, after
compression reduces HBM traffic, CUDA-core dequantization may become the new
limiting resource.

Point out the practical overheads shown on the slide:

- Quantization scales and zero-points add metadata bytes.
- Dequantization and softmax use CUDA-core FLOPs.
- Quest page indexing adds bookkeeping traffic.
- MoE models may touch several routed experts as batch size increases.

The model therefore evaluates physical effects that an idealized compression
ratio would miss.

**Suggested wording:** “We do not assume that reducing bytes automatically
produces the same proportional speedup. We calculate the time for each resource
and let the active roof determine the result.”

### Slide 6 — Experiments and validation

This slide explains how the design space was explored and how the model was
checked against hardware. The experiments vary one or more of the factors that
can change the active bottleneck:

- Hardware bandwidth and SRAM capacity.
- BF16, FP8, and INT4 representations.
- Quantization group size.
- Quest eviction budgets.
- Context length and batch size.
- One-GPU and two-GPU execution.

Explain the purpose of each group of experiments:

1. **Hardware experiments** ask whether extra HBM bandwidth or on-chip SRAM
   provides meaningful return for its area cost.
2. **Precision experiments** compare the traffic reduction from FP8 and INT4
   with their metadata, quality, and dequantization overheads.
3. **Eviction experiments** test whether retaining a smaller, more useful KV
   working set can reduce traffic and allow a larger batch.
4. **Scaling experiments** test whether tensor parallelism is needed when one
   GPU cannot hold the KV cache.

The validation process has two separate measurements:

- **HBM traffic:** the predicted bytes are compared with NVIDIA Nsight Compute
  DRAM counters. This tests whether the byte-counting model represents actual
  memory movement.
- **Decode latency:** predicted time per output token (TPOT) is compared with
  real GPU decode measurements. This tests the complete roofline model,
  including effective bandwidth and fixed overhead.

The bandwidth efficiency parameter is calibrated on a subset of measurements,
then the final error is reported on held-out configurations. This distinction
matters: a held-out result tests whether the model generalizes rather than only
memorizing the calibration data.

The reported results are:

- 3.3% MAPE for HBM traffic.
- 12.9% held-out MAPE for TPOT.
- 26 automated tests covering equations, capacity limits, SRAM behavior,
  expert touch, and design-space exploration.

**Suggested wording:** “The traffic result validates the physical byte model;
the held-out TPOT result validates whether those bytes and roof limits translate
into useful latency predictions.”

### Slide 7 — Results and plots

This slide contains the first set of final results plots. Start with the
validation plot and explain that the closer the predicted and measured values
are, the better the model captures the hardware behavior. The 3.3% HBM traffic
MAPE indicates that the model's accounting of weights, KV data, metadata, and
other traffic is close to the Nsight measurements.

Then use the decomposition plot to explain that total decode time is made of
multiple components. A large attention portion indicates that the KV path is
important; a large linear portion indicates that weight streaming is important.
The plot is not only showing “fast versus slow”—it is showing *why* a
configuration is fast or slow.

Emphasize bottleneck movement:

1. With BF16 KV, attention can dominate because the cache is large.
2. INT4 KV reduces attention bytes, so the linear weight path may become more
   visible.
3. If the compression method requires expensive transforms or unpacking,
   CUDA-core time can grow and limit the benefit.

This is called **bottleneck oscillation**: after one bottleneck is improved, a
different bottleneck becomes dominant. It is why the project evaluates all
resource roofs together instead of ranking techniques only by their nominal
compression ratio.

**Suggested wording:** “The important result is not just that compression helps.
It is that compression changes the identity of the bottleneck, so every
optimization must be evaluated together with the work it introduces.”

### Slide 8 — Regime-specific recommendations

This slide turns the experiments into practical design recommendations. Explain
that the same accelerator should not be optimized identically for every context
length.

#### Regime A: short and medium context

- At up to about 8k tokens, the workload is commonly **weight-bound**.
- The KV cache is not yet large enough to dominate all memory traffic.
- Extra HBM bandwidth and FP8 KV provide approximately 1.71x–1.92x speedup.
- Large SRAM expansion alone has weak return because the working set is still
  streamed and most data does not remain reusable in SRAM.

#### Regime B: long context

- Around 128k tokens, the workload becomes **KV-bound**.
- BF16 KV consumes substantial memory and limits the batch size.
- INT4 group-size-64 KV reduces the stored and transferred bytes.
- Quest eviction keeps a bounded set of useful tokens or pages.
- Together they provide approximately 6.23x speedup and increase batch
  concurrency from 3 to 8, while staying within the reported 1% quality gate.
- The zero added silicon area is important: this improvement is primarily a
  software and representation change, not a new SRAM block.

#### Regime C: ultra-long context

- At 512k tokens or more, the problem becomes **capacity-bound**.
- A BF16 KV cache may not fit on one GPU at all, regardless of peak compute.
- INT4 KV and Quest eviction reduce the footprint enough to make execution
  feasible.
- Two-GPU tensor parallelism supplies additional memory capacity and distributes
  the workload.
- The reported configuration reaches approximately 11.15x speedup.

The headline lesson is that the best intervention follows the active constraint:
HBM for weight-bound cases, compression and eviction for KV-bound cases, and
distributed execution for capacity-bound cases.

**Suggested wording:** “At short context we buy bandwidth. At long context we
reduce the KV working set. At ultra-long context we must also distribute that
working set across GPUs.”

### Slide 9 — Summary

Use this slide to connect the detailed results to the project question. The
framework does not claim that one technology always wins. It identifies the
dominant bottleneck at a particular context length and then ranks interventions
under area, energy, quality, and capacity constraints.

Walk through the conclusions:

1. **Decode is a memory-hierarchy problem.** Peak FLOPs do not determine
   performance when the GPU is waiting for HBM data.
2. **Context changes the bottleneck.** Weight traffic matters more at shorter
   contexts; KV traffic and capacity dominate at longer contexts.
3. **Compression is also a concurrency technique.** It reduces bytes per token
   and frees VRAM, allowing more sequences to run together.
4. **Co-design is necessary.** A compression technique can expose weight or
   CUDA limits, while hardware bandwidth cannot by itself solve a capacity
   problem.
5. **The model is quantitatively useful.** The 3.3% HBM traffic MAPE and 12.9%
   held-out TPOT MAPE show that the framework can support design decisions.

End with the practical recommendation matrix:

```text
short context  -> HBM bandwidth / light KV precision
long context   -> INT4 KV + token eviction
ultra-long     -> compression + eviction + multi-GPU scaling
```

### Slide 10 — Thanks and references

Use the final slide as both the closing and the handoff to questions. Thank the
audience, then point to the four supporting resources:

- The repository contains the analytical implementation, data, tests, and
  generated figures.
- The literature review explains the prior work and motivates the research gap.
- The final paper gives the formal equations, assumptions, and results.
- The bibliography provides the sources for the roofline model, KV-cache
  compression, eviction methods, and hardware assumptions.

If questions begin immediately, the most useful one-sentence answer is:

> The project shows that long-context decode should be optimized according to
> its active bottleneck: bandwidth at shorter contexts, KV compression and
> eviction at long contexts, and distributed capacity at ultra-long contexts.

For technical questions, refer back to the distinction between **traffic** and
**latency**: the 3.3% result concerns predicted HBM bytes, while the 12.9%
result concerns held-out time per output token.

## 6. Timing plan

| Slides | Topic | Suggested time |
|---|---|---:|
| 1 | Introduction | 1:00 |
| 2–3 | Problem and bottlenecks | 2:00 |
| 4 | Project overview | 1:00 |
| 5–6 | Methodology and experiments | 2:00 |
| 7–8 | Results and recommendations | 2:00 |
| 9 | Summary | 1:00 |
| 10 | Thanks and references | 1:00 |
| **Total** |  | **10:00** |

## 7. One-minute emergency version

If time is very limited, say:

> Long-context LLM decode is often memory-bound because every generated token
> must read model weights and an increasingly large KV cache. We built a
> two-group hierarchical roofline model that represents weight work, attention
> work, HBM, SRAM, CUDA-core overheads, and communication. The model matches HBM
> traffic with 3.3% MAPE and held-out decode latency with 12.9% MAPE. Our
> experiments show that shorter contexts benefit from HBM bandwidth, long
> contexts benefit from INT4 KV compression and token eviction, and ultra-long
> contexts require compression plus multi-GPU scaling. The main conclusion is
> that the best solution depends on the active bottleneck and must be chosen
> through hardware–software co-design.

## 8. Common questions

### Why not just add more compute?

The workload is often waiting for memory rather than computation. More compute
units do not help if the data cannot arrive quickly enough.

### Why does SRAM alone have limited benefit?

The KV cache can be tens of gigabytes, while a practical SRAM budget is much
smaller. If the workload streams through data that is not reused, most requests
still go to HBM.

### Why is INT4 not always the answer?

INT4 reduces bytes, but it can introduce quality loss, metadata overhead, and
dequantization work. If dequantization becomes the CUDA bottleneck, the speedup
will be less than the raw byte reduction suggests.

### Why does compression improve throughput?

Compression reduces traffic and lowers KV-cache capacity requirements. The freed
memory can allow a larger batch, so more sequences can be served concurrently.

### What is the main contribution?

The main contribution is a unified, hardware-aware design-space exploration
framework that compares memory, compute, capacity, quality, and area trade-offs
across context regimes.

## 9. Supporting project files

- [10-slide presentation](./KV_Roofline_10_Slide_Deck.pptx)
- [Project presentation blueprint](../PRESENTATION.md)
- [Final paper PDF](./FINAL_PAPER.pdf)
- [Literature review](./amcas_lit_rev.pdf)
- [Repository README](../README.md)
