# DSE results: llama3-70b on h100

Quality source: **measured table data/quality_table.csv**; gate = 1.0%; theta = 0.5; rho = 0.0; batch mode = max; dequant FLOPs/elem = 6.0; attention unit = tensor.

## A_short_medium  (S = 8,192, chips = 8)

**Baseline** (BF16 KV, no extras): B = 8 (max feasible 158), TPOT = 10.65 ms, 751 tok/s, dominant = `linear:hbm`, AI = 8.1 FLOP/B (ridge 295), KV share of HBM bytes = 12%, S* = 53,062 tokens.

### Best gated config per knob family

| family | config | B | speedup | +mm2 | dE mJ/tok | dAI | d(AI/ridge) | quality % | bound |
|---|---|---|---|---|---|---|---|---|---|
| SRAM+HBM+COMP | +1024 MB SRAM (pins weights), +2 HBM stack, KV8+R128+streaming4096 | 8 | 1.55 | 401 | -280 | 0.901 | 0.0153 | 0.96 | linear:hbm |
| HBM+COMP | +2 HBM stack, KV8+R128+streaming4096 | 8 | 1.48 | 138 | -403 | 0.447 | 0.0131 | 0.96 | linear:hbm |
| SRAM+HBM | +1024 MB SRAM (pins KV), +2 HBM stack | 8 | 1.44 | 401 | -175 | 0.607 | 0.0139 | 0 | linear:hbm |
| HBM | +2 HBM stack | 8 | 1.36 | 138 | -282 | 0 | 0.011 | 0 | linear:hbm |
| SRAM+COMP | +1024 MB SRAM (pins weights), KV8+R128+streaming4096 | 8 | 1.15 | 263 | 23.8 | 0.901 | 0.00305 | 0.96 | linear:hbm |
| COMP | KV8+R128+streaming4096 | 8 | 1.09 | 0 | -147 | 0.447 | 0.00151 | 0.96 | linear:hbm |
| SRAM | +1024 MB SRAM (pins KV) | 8 | 1.07 | 263 | 154 | 0.607 | 0.00206 | 0 | linear:hbm |
| baseline | baseline | 8 | 1 | 0 | 0 | 0 | 0 | 0 | linear:hbm |

### Iso-area winners (best tokens/s under an added-area budget, quality-gated)

| budget mm2 | config | B | speedup | used mm2 | mJ/token | quality % | measured? |
|---|---|---|---|---|---|---|---|
| 0 | KV8+R128+streaming4096 | 8 | 1.09 | 0 | 1.6e+03 | 0.96 | False |
| 25 | +64 MB SRAM (pins KV), KV8+R128+streaming4096 | 8 | 1.1 | 16.4 | 1.61e+03 | 0.96 | False |
| 50 | +128 MB SRAM (pins KV), KV8+R128+streaming4096 | 8 | 1.1 | 32.8 | 1.62e+03 | 0.96 | False |
| 100 | +64 MB SRAM (pins KV), +1 HBM stack, KV8+R128+streaming4096 | 8 | 1.3 | 85.4 | 1.46e+03 | 0.96 | False |
| 200 | +128 MB SRAM (pins KV), +2 HBM stack, KV8+R128+streaming4096 | 8 | 1.49 | 171 | 1.36e+03 | 0.96 | False |
| 400 | +512 MB SRAM (pins weights), +2 HBM stack, KV8+R128+streaming4096 | 8 | 1.51 | 269 | 1.41e+03 | 0.96 | False |

* Best single-knob **SRAM** efficiency: +64 MB SRAM (pins KV) -> speedup 1.010x for 16 mm2 (5.85e-04 speedup/mm2; dAI/mm2 = 5.19e-03, d(AI/ridge)/mm2 = 1.76e-05).
* Best single-knob **HBM** efficiency: +1 HBM stack -> speedup 1.180x for 69 mm2 (2.62e-03 speedup/mm2; dAI/mm2 = 0.00e+00, d(AI/ridge)/mm2 = 7.99e-05).

## B_long  (S = 131,072, chips = 8)

**Baseline** (BF16 KV, no extras): B = 8 (max feasible 9), TPOT = 30.08 ms, 266 tok/s, dominant = `attn:hbm`, AI = 8.2 FLOP/B (ridge 295), KV share of HBM bytes = 71%, S* = 53,062 tokens.

### Best gated config per knob family

| family | config | B | speedup | +mm2 | dE mJ/tok | dAI | d(AI/ridge) | quality % | bound |
|---|---|---|---|---|---|---|---|---|---|
| SRAM+HBM+COMP | +1024 MB SRAM (pins weights), +2 HBM stack, KV16+quest4096 | 8 | 3.73 | 401 | -3.34e+03 | -0.862 | 0.00706 | 0.88 | linear:hbm |
| HBM+COMP | +2 HBM stack, KV16+quest4096 | 8 | 3.57 | 138 | -3.49e+03 | -1.21 | 0.00539 | 0.88 | linear:hbm |
| SRAM+COMP | +1024 MB SRAM (pins weights), KV16+quest4096 | 8 | 2.75 | 263 | -2.98e+03 | -0.862 | -0.00292 | 0.88 | linear:hbm |
| COMP | KV16+quest4096 | 8 | 2.63 | 0 | -3.18e+03 | -1.21 | -0.00411 | 0.88 | linear:hbm |
| SRAM+HBM | +1024 MB SRAM (pins KV), +2 HBM stack | 8 | 1.52 | 401 | -663 | 0.842 | 0.0151 | 0 | attn:hbm |
| HBM | +2 HBM stack | 8 | 1.38 | 138 | -842 | 0 | 0.0111 | 0 | attn:hbm |
| SRAM | +1024 MB SRAM (pins KV) | 8 | 1.1 | 263 | 294 | 0.842 | 0.00285 | 0 | attn:hbm |
| baseline | baseline | 8 | 1 | 0 | 0 | 0 | 0 | 0 | attn:hbm |

### Iso-area winners (best tokens/s under an added-area budget, quality-gated)

| budget mm2 | config | B | speedup | used mm2 | mJ/token | quality % | measured? |
|---|---|---|---|---|---|---|---|
| 0 | KV16+quest4096 | 8 | 2.63 | 0 | 1.88e+03 | 0.88 | True |
| 25 | +64 MB SRAM (pins weights), KV16+quest4096 | 8 | 2.63 | 16.4 | 1.89e+03 | 0.88 | True |
| 50 | +128 MB SRAM (pins weights), KV16+quest4096 | 8 | 2.64 | 32.8 | 1.9e+03 | 0.88 | True |
| 100 | +64 MB SRAM (pins weights), +1 HBM stack, KV16+quest4096 | 8 | 3.11 | 85.4 | 1.71e+03 | 0.88 | True |
| 200 | +128 MB SRAM (pins weights), +2 HBM stack, KV16+quest4096 | 8 | 3.59 | 171 | 1.59e+03 | 0.88 | True |
| 400 | +512 MB SRAM (pins weights), +2 HBM stack, KV16+quest4096 | 8 | 3.65 | 269 | 1.64e+03 | 0.88 | True |

* Best single-knob **SRAM** efficiency: +64 MB SRAM (pins KV) -> speedup 1.014x for 16 mm2 (8.31e-04 speedup/mm2; dAI/mm2 = 7.05e-03, d(AI/ridge)/mm2 = 2.39e-05).
* Best single-knob **HBM** efficiency: +1 HBM stack -> speedup 1.193x for 69 mm2 (2.80e-03 speedup/mm2; dAI/mm2 = 0.00e+00, d(AI/ridge)/mm2 = 8.08e-05).

## C_ultra_long  (S = 524,288, chips = 8)

**Baseline** (BF16 KV, no extras): B = 2 (max feasible 2), TPOT = 30.02 ms, 67 tok/s, dominant = `attn:hbm`, AI = 6.5 FLOP/B (ridge 295), KV share of HBM bytes = 71%, S* = 212,248 tokens.

### Best gated config per knob family

| family | config | B | speedup | +mm2 | dE mJ/tok | dAI | d(AI/ridge) | quality % | bound |
|---|---|---|---|---|---|---|---|---|---|
| SRAM+HBM+COMP | +1024 MB SRAM (pins weights), +2 HBM stack, KV4g64+R128+quest16384 | 8 | 10.9 | 401 | -1.77e+04 | 0.475 | 0.011 | 0.89 | linear:hbm |
| HBM+COMP | +2 HBM stack, KV4g64+R128+quest16384 | 8 | 10.6 | 138 | -1.79e+04 | 0.237 | 0.00988 | 0.89 | linear:hbm |
| SRAM+COMP | +1024 MB SRAM (pins weights), KV4g64+R128+quest16384 | 8 | 7.99 | 263 | -1.72e+04 | 0.475 | 0.00161 | 0.89 | linear:hbm |
| COMP | KV4g64+R128+quest16384 | 8 | 7.73 | 0 | -1.75e+04 | 0.237 | 0.000802 | 0.89 | linear:hbm |
| SRAM+HBM | +1024 MB SRAM (pins KV), +2 HBM stack | 3 | 1.68 | 401 | -4.21e+03 | 1.32 | 0.015 | 0 | attn:hbm |
| HBM | +2 HBM stack | 3 | 1.54 | 138 | -4.97e+03 | 0.672 | 0.0119 | 0 | attn:hbm |
| SRAM | +1024 MB SRAM (pins KV) | 2 | 1.1 | 263 | 1.17e+03 | 0.661 | 0.00224 | 0 | attn:hbm |
| baseline | baseline | 2 | 1 | 0 | 0 | 0 | 0 | 0 | attn:hbm |

### Iso-area winners (best tokens/s under an added-area budget, quality-gated)

| budget mm2 | config | B | speedup | used mm2 | mJ/token | quality % | measured? |
|---|---|---|---|---|---|---|---|
| 0 | KV4g64+R128+quest16384 | 8 | 7.73 | 0 | 2.58e+03 | 0.89 | True |
| 25 | +64 MB SRAM (pins weights), KV4g64+R128+quest16384 | 8 | 7.75 | 16.4 | 2.6e+03 | 0.89 | True |
| 50 | +128 MB SRAM (pins weights), KV4g64+R128+quest16384 | 8 | 7.76 | 32.8 | 2.62e+03 | 0.89 | True |
| 100 | +64 MB SRAM (pins weights), +1 HBM stack, KV4g64+R128+quest16384 | 8 | 9.19 | 85.4 | 2.35e+03 | 0.89 | True |
| 200 | +128 MB SRAM (pins weights), +2 HBM stack, KV4g64+R128+quest16384 | 8 | 10.6 | 171 | 2.19e+03 | 0.89 | True |
| 400 | +512 MB SRAM (pins weights), +2 HBM stack, KV4g64+R128+quest16384 | 8 | 10.8 | 269 | 2.27e+03 | 0.89 | True |

* Best single-knob **SRAM** efficiency: +64 MB SRAM (pins KV) -> speedup 1.014x for 16 mm2 (8.33e-04 speedup/mm2; dAI/mm2 = 5.54e-03, d(AI/ridge)/mm2 = 1.88e-05).
* Best single-knob **HBM** efficiency: +1 HBM stack -> speedup 1.329x for 69 mm2 (4.76e-03 speedup/mm2; dAI/mm2 = 9.74e-03, d(AI/ridge)/mm2 = 1.03e-04).
