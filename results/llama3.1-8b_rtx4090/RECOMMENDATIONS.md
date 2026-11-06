# DSE results: llama3.1-8b on rtx4090

Quality source: **measured table data/quality_table.csv**; gate = 1.0%; theta = 1.0; rho = 0.0; batch mode = max; dequant FLOPs/elem = 6.0; attention unit = tensor.

## A_short  (S = 4,096, chips = 1)

**Baseline** (BF16 KV, no extras): B = 4 (max feasible 8), TPOT = 28.10 ms, 142 tok/s, dominant = `linear:hbm`, AI = 4.0 FLOP/B (ridge 164), KV share of HBM bytes = 12%, S* = 28,628 tokens.

### Best gated config per knob family

| family | config | B | speedup | +mm2 | dE mJ/tok | dAI | d(AI/ridge) | quality % | bound |
|---|---|---|---|---|---|---|---|---|---|
| SRAM+HBM+COMP | +1024 MB SRAM (pins weights), +2 HBM stack, KV4g128+R128 | 4 | 3.53 | 401 | -192 | 1.13 | 0.0699 | 0.45 | linear:hbm |
| HBM+COMP | +2 HBM stack, KV4g128+R128 | 4 | 3.29 | 138 | -223 | 0.795 | 0.0637 | 0.45 | linear:hbm |
| SRAM+HBM | +1024 MB SRAM (pins KV), +2 HBM stack | 4 | 3.19 | 401 | -157 | 0.256 | 0.0538 | 0 | linear:hbm |
| HBM | +2 HBM stack | 4 | 3 | 138 | -194 | 0 | 0.0491 | 0 | linear:hbm |
| SRAM+COMP | +1024 MB SRAM (pins weights), KV4g128+R128 | 4 | 1.18 | 263 | 74.9 | 1.13 | 0.00692 | 0.45 | linear:hbm |
| COMP | KV4g128+R128 | 4 | 1.1 | 0 | -47 | 0.795 | 0.00486 | 0.45 | linear:hbm |
| SRAM | +1024 MB SRAM (pins KV) | 4 | 1.06 | 263 | 138 | 0.256 | 0.00156 | 0 | linear:hbm |
| baseline | baseline | 4 | 1 | 0 | 0 | 0 | 0 | 0 | linear:hbm |

### Iso-area winners (best tokens/s under an added-area budget, quality-gated)

| budget mm2 | config | B | speedup | used mm2 | mJ/token | quality % | measured? |
|---|---|---|---|---|---|---|---|
| 0 | KV4g128+R128 | 4 | 1.1 | 0 | 507 | 0.45 | True |
| 25 | +64 MB SRAM (pins KV), KV4g128+R128 | 4 | 1.1 | 16.4 | 515 | 0.45 | True |
| 50 | +128 MB SRAM (pins KV), KV4g128+R128 | 4 | 1.11 | 32.8 | 523 | 0.45 | True |
| 100 | +64 MB SRAM (pins KV), +1 HBM stack, KV4g128+R128 | 4 | 2.21 | 85.4 | 378 | 0.45 | True |
| 200 | +128 MB SRAM (pins weights), +2 HBM stack, KV4g128+R128 | 4 | 3.32 | 171 | 335 | 0.45 | True |
| 400 | +512 MB SRAM (pins weights), +2 HBM stack, KV4g128+R128 | 4 | 3.41 | 269 | 347 | 0.45 | True |

* Best single-knob **SRAM** efficiency: +1024 MB SRAM (pins KV) -> speedup 1.064x for 263 mm2 (2.43e-04 speedup/mm2; dAI/mm2 = 9.74e-04, d(AI/ridge)/mm2 = 5.95e-06).
* Best single-knob **HBM** efficiency: +1 HBM stack -> speedup 2.000x for 69 mm2 (1.45e-02 speedup/mm2; dAI/mm2 = 0.00e+00, d(AI/ridge)/mm2 = 3.56e-04).

## B_medium  (S = 16,384, chips = 1)

**Baseline** (BF16 KV, no extras): B = 2 (max feasible 2), TPOT = 31.63 ms, 63 tok/s, dominant = `linear:hbm`, AI = 2.5 FLOP/B (ridge 164), KV share of HBM bytes = 22%, S* = 57,256 tokens.

### Best gated config per knob family

| family | config | B | speedup | +mm2 | dE mJ/tok | dAI | d(AI/ridge) | quality % | bound |
|---|---|---|---|---|---|---|---|---|---|
| SRAM+HBM+COMP | +1024 MB SRAM (pins weights), +2 HBM stack, KV16+streaming4096 | 2 | 3.85 | 401 | -506 | -0.171 | 0.0269 | 0.61 | linear:hbm |
| HBM+COMP | +2 HBM stack, KV16+streaming4096 | 2 | 3.6 | 138 | -572 | -0.317 | 0.0242 | 0.61 | linear:hbm |
| SRAM+HBM | +1024 MB SRAM (pins weights), +2 HBM stack | 2 | 3.17 | 401 | -347 | 0.138 | 0.0326 | 0 | linear:hbm |
| HBM | +2 HBM stack | 2 | 3 | 138 | -436 | 0 | 0.0301 | 0 | linear:hbm |
| SRAM+COMP | +1024 MB SRAM (pins weights), KV16+streaming4096 | 2 | 1.28 | 263 | 44.6 | -0.171 | -0.00104 | 0.61 | linear:hbm |
| COMP | KV16+streaming4096 | 2 | 1.2 | 0 | -209 | -0.317 | -0.00194 | 0.61 | linear:hbm |
| SRAM | +1024 MB SRAM (pins KV) | 2 | 1.06 | 263 | 321 | 0.138 | 0.000844 | 0 | linear:hbm |
| baseline | baseline | 2 | 1 | 0 | 0 | 0 | 0 | 0 | linear:hbm |

### Iso-area winners (best tokens/s under an added-area budget, quality-gated)

| budget mm2 | config | B | speedup | used mm2 | mJ/token | quality % | measured? |
|---|---|---|---|---|---|---|---|
| 0 | KV16+streaming4096 | 2 | 1.2 | 0 | 1.03e+03 | 0.61 | True |
| 25 | +64 MB SRAM (pins KV), KV16+streaming4096 | 2 | 1.21 | 16.4 | 1.05e+03 | 0.61 | True |
| 50 | +128 MB SRAM (pins KV), KV16+streaming4096 | 2 | 1.21 | 32.8 | 1.07e+03 | 0.61 | True |
| 100 | +64 MB SRAM (pins KV), +1 HBM stack, KV16+streaming4096 | 2 | 2.41 | 85.4 | 768 | 0.61 | True |
| 200 | +128 MB SRAM (pins weights), +2 HBM stack, KV16+streaming4096 | 2 | 3.63 | 171 | 678 | 0.61 | True |
| 400 | +512 MB SRAM (pins KV), +2 HBM stack, KV16+streaming4096 | 2 | 3.72 | 269 | 704 | 0.61 | True |

* Best single-knob **SRAM** efficiency: +1024 MB SRAM (pins KV) -> speedup 1.056x for 263 mm2 (2.14e-04 speedup/mm2; dAI/mm2 = 5.26e-04, d(AI/ridge)/mm2 = 3.22e-06).
* Best single-knob **HBM** efficiency: +1 HBM stack -> speedup 2.000x for 69 mm2 (1.45e-02 speedup/mm2; dAI/mm2 = 0.00e+00, d(AI/ridge)/mm2 = 2.18e-04).

## C_long  (S = 32,768, chips = 1)

**Baseline** (BF16 KV, no extras): B = 1 (max feasible 1), TPOT = 31.63 ms, 32 tok/s, dominant = `linear:hbm`, AI = 1.7 FLOP/B (ridge 164), KV share of HBM bytes = 22%, S* = 114,512 tokens.

### Best gated config per knob family

| family | config | B | speedup | +mm2 | dE mJ/tok | dAI | d(AI/ridge) | quality % | bound |
|---|---|---|---|---|---|---|---|---|---|
| SRAM+HBM+COMP | +1024 MB SRAM (pins weights), +2 HBM stack, KV16+streaming4096 | 1 | 3.99 | 401 | -1.06e+03 | -0.494 | 0.0115 | 0.92 | linear:hbm |
| HBM+COMP | +2 HBM stack, KV16+streaming4096 | 1 | 3.73 | 138 | -1.19e+03 | -0.572 | 0.0101 | 0.92 | linear:hbm |
| SRAM+HBM | +1024 MB SRAM (pins KV), +2 HBM stack | 1 | 3.17 | 401 | -695 | 0.0944 | 0.0223 | 0 | linear:hbm |
| HBM | +2 HBM stack | 1 | 3 | 138 | -872 | 0 | 0.0205 | 0 | linear:hbm |
| SRAM+COMP | +1024 MB SRAM (pins weights), KV16+streaming4096 | 1 | 1.33 | 263 | -3.03 | -0.494 | -0.00302 | 0.92 | linear:hbm |
| COMP | KV16+streaming4096 | 1 | 1.24 | 0 | -487 | -0.572 | -0.00349 | 0.92 | linear:hbm |
| SRAM | +1024 MB SRAM (pins KV) | 1 | 1.06 | 263 | 642 | 0.0944 | 0.000577 | 0 | linear:hbm |
| baseline | baseline | 1 | 1 | 0 | 0 | 0 | 0 | 0 | linear:hbm |

### Iso-area winners (best tokens/s under an added-area budget, quality-gated)

| budget mm2 | config | B | speedup | used mm2 | mJ/token | quality % | measured? |
|---|---|---|---|---|---|---|---|
| 0 | KV16+streaming4096 | 1 | 1.24 | 0 | 1.99e+03 | 0.92 | True |
| 25 | +64 MB SRAM (pins KV), KV16+streaming4096 | 1 | 1.25 | 16.4 | 2.02e+03 | 0.92 | True |
| 50 | +128 MB SRAM (pins KV), KV16+streaming4096 | 1 | 1.25 | 32.8 | 2.05e+03 | 0.92 | True |
| 100 | +64 MB SRAM (pins KV), +1 HBM stack, KV16+streaming4096 | 1 | 2.5 | 85.4 | 1.48e+03 | 0.92 | True |
| 200 | +128 MB SRAM (pins weights), +2 HBM stack, KV16+streaming4096 | 1 | 3.76 | 171 | 1.3e+03 | 0.92 | True |
| 400 | +512 MB SRAM (pins weights), +2 HBM stack, KV16+streaming4096 | 1 | 3.86 | 269 | 1.35e+03 | 0.92 | True |

* Best single-knob **SRAM** efficiency: +1024 MB SRAM (pins KV) -> speedup 1.056x for 263 mm2 (2.14e-04 speedup/mm2; dAI/mm2 = 3.60e-04, d(AI/ridge)/mm2 = 2.20e-06).
* Best single-knob **HBM** efficiency: +1 HBM stack -> speedup 2.000x for 69 mm2 (1.45e-02 speedup/mm2; dAI/mm2 = 0.00e+00, d(AI/ridge)/mm2 = 1.49e-04).
