# DSE results: qwen2.5-7b on h100

Quality source: **measured table data/quality_table.csv**; gate = 1.0%; theta = 1.0; rho = 0.0; batch mode = max; dequant FLOPs/elem = 6.0; attention unit = tensor.

## A_short_medium  (S = 8,192, chips = 1)

**Baseline** (BF16 KV, no extras): B = 8 (max feasible 118), TPOT = 8.83 ms, 906 tok/s, dominant = `linear:hbm`, AI = 7.8 FLOP/B (ridge 295), KV share of HBM bytes = 21%, S* = 30,845 tokens.

### Best gated config per knob family

| family | config | B | speedup | +mm2 | dE mJ/tok | dAI | d(AI/ridge) | quality % | bound |
|---|---|---|---|---|---|---|---|---|---|
| SRAM+HBM+COMP | +1024 MB SRAM (pins weights), +2 HBM stack, KV8+R128+streaming4096 | 8 | 1.78 | 401 | -46.6 | 1.57 | 0.018 | 0.96 | linear:hbm |
| HBM+COMP | +2 HBM stack, KV8+R128+streaming4096 | 8 | 1.66 | 138 | -54.7 | 0.928 | 0.015 | 0.96 | linear:hbm |
| SRAM+HBM | +1024 MB SRAM (pins KV), +2 HBM stack | 8 | 1.49 | 401 | -20.6 | 0.475 | 0.0129 | 0 | linear:hbm |
| HBM | +2 HBM stack | 8 | 1.4 | 138 | -31.8 | 0 | 0.0106 | 0 | linear:hbm |
| SRAM+COMP | +1024 MB SRAM (pins weights), KV8+R128+streaming4096 | 8 | 1.27 | 263 | -15.2 | 1.57 | 0.0053 | 0.96 | linear:hbm |
| COMP | KV8+R128+streaming4096 | 8 | 1.18 | 0 | -27.9 | 0.928 | 0.00314 | 0.96 | linear:hbm |
| SRAM | +1024 MB SRAM (pins KV) | 8 | 1.06 | 263 | 17.1 | 0.475 | 0.00161 | 0 | linear:hbm |
| baseline | baseline | 8 | 1 | 0 | 0 | 0 | 0 | 0 | linear:hbm |

### Iso-area winners (best tokens/s under an added-area budget, quality-gated)

| budget mm2 | config | B | speedup | used mm2 | mJ/token | quality % | measured? |
|---|---|---|---|---|---|---|---|
| 0 | KV8+R128+streaming4096 | 8 | 1.18 | 0 | 159 | 0.96 | False |
| 25 | +64 MB SRAM (pins KV), KV8+R128+streaming4096 | 8 | 1.19 | 16.4 | 160 | 0.96 | False |
| 50 | +128 MB SRAM (pins KV), KV8+R128+streaming4096 | 8 | 1.2 | 32.8 | 161 | 0.96 | False |
| 100 | +64 MB SRAM (pins KV), +1 HBM stack, KV8+R128+streaming4096 | 8 | 1.43 | 85.4 | 144 | 0.96 | False |
| 200 | +128 MB SRAM (pins KV), +2 HBM stack, KV8+R128+streaming4096 | 8 | 1.67 | 171 | 134 | 0.96 | False |
| 400 | +512 MB SRAM (pins KV), +2 HBM stack, KV8+R128+streaming4096 | 8 | 1.72 | 269 | 137 | 0.96 | False |

* Best single-knob **SRAM** efficiency: +1024 MB SRAM (pins KV) -> speedup 1.061x for 263 mm2 (2.31e-04 speedup/mm2; dAI/mm2 = 1.81e-03, d(AI/ridge)/mm2 = 6.13e-06).
* Best single-knob **HBM** efficiency: +1 HBM stack -> speedup 1.200x for 69 mm2 (2.90e-03 speedup/mm2; dAI/mm2 = 0.00e+00, d(AI/ridge)/mm2 = 7.68e-05).

## B_long  (S = 131,072, chips = 1)

**Baseline** (BF16 KV, no extras): B = 7 (max feasible 7), TPOT = 32.99 ms, 212 tok/s, dominant = `attn:hbm`, AI = 7.1 FLOP/B (ridge 295), KV share of HBM bytes = 79%, S* = 35,251 tokens.

### Best gated config per knob family

| family | config | B | speedup | +mm2 | dE mJ/tok | dAI | d(AI/ridge) | quality % | bound |
|---|---|---|---|---|---|---|---|---|---|
| SRAM+HBM+COMP | +1024 MB SRAM (pins weights), +2 HBM stack, KV16+quest4096 | 8 | 5.7 | 401 | -613 | -0.303 | 0.00812 | 0.88 | linear:hbm |
| HBM+COMP | +2 HBM stack, KV16+quest4096 | 8 | 5.41 | 138 | -627 | -0.654 | 0.00646 | 0.88 | linear:hbm |
| SRAM+COMP | +1024 MB SRAM (pins weights), KV4g64+R128+quest16384 | 8 | 4.01 | 263 | -562 | 2.87 | 0.00973 | 0.66 | linear:hbm |
| COMP | KV4g64+R128+quest16384 | 8 | 3.81 | 0 | -583 | 2.36 | 0.00801 | 0.66 | linear:hbm |
| SRAM+HBM | +1024 MB SRAM (pins KV), +2 HBM stack | 8 | 1.46 | 401 | -77.8 | 0.293 | 0.011 | 0 | attn:hbm |
| HBM | +2 HBM stack | 8 | 1.44 | 138 | -153 | 0.192 | 0.0105 | 0 | attn:hbm |
| SRAM | +1024 MB SRAM (pins KV) | 7 | 1.02 | 263 | 108 | 0.11 | 0.000373 | 0 | attn:hbm |
| baseline | baseline | 7 | 1 | 0 | 0 | 0 | 0 | 0 | attn:hbm |

### Iso-area winners (best tokens/s under an added-area budget, quality-gated)

| budget mm2 | config | B | speedup | used mm2 | mJ/token | quality % | measured? |
|---|---|---|---|---|---|---|---|
| 0 | KV4g64+R128+quest16384 | 8 | 3.81 | 0 | 215 | 0.66 | True |
| 25 | +64 MB SRAM (pins weights), KV4g64+R128+quest16384 | 8 | 3.82 | 16.4 | 216 | 0.66 | True |
| 50 | +128 MB SRAM (pins weights), KV4g64+R128+quest16384 | 8 | 3.83 | 32.8 | 217 | 0.66 | True |
| 100 | +64 MB SRAM (pins weights), +1 HBM stack, KV16+quest4096 | 8 | 4.65 | 85.4 | 186 | 0.88 | True |
| 200 | +128 MB SRAM (pins weights), +2 HBM stack, KV16+quest4096 | 8 | 5.44 | 171 | 172 | 0.88 | True |
| 400 | +512 MB SRAM (pins weights), +2 HBM stack, KV16+quest4096 | 8 | 5.55 | 269 | 177 | 0.88 | True |

* Best single-knob **SRAM** efficiency: +1024 MB SRAM (pins KV) -> speedup 1.016x for 263 mm2 (5.94e-05 speedup/mm2; dAI/mm2 = 4.19e-04, d(AI/ridge)/mm2 = 1.42e-06).
* Best single-knob **HBM** efficiency: +1 HBM stack -> speedup 1.233x for 69 mm2 (3.37e-03 speedup/mm2; dAI/mm2 = 2.78e-03, d(AI/ridge)/mm2 = 8.06e-05).

## C_ultra_long  (S = 524,288, chips = 1)

**Baseline** (BF16 KV, no extras): B = 1 (max feasible 1), TPOT = 21.84 ms, 46 tok/s, dominant = `attn:hbm`, AI = 5.1 FLOP/B (ridge 295), KV share of HBM bytes = 68%, S* = 246,757 tokens.

### Best gated config per knob family

| family | config | B | speedup | +mm2 | dE mJ/tok | dAI | d(AI/ridge) | quality % | bound |
|---|---|---|---|---|---|---|---|---|---|
| SRAM+HBM+COMP | +1024 MB SRAM (pins weights), +2 HBM stack, KV4g64+R128+quest16384 | 8 | 16.3 | 401 | -3.37e+03 | 1.1 | 0.0122 | 0.89 | attn:hbm |
| HBM+COMP | +2 HBM stack, KV4g64+R128+quest16384 | 8 | 15.8 | 138 | -3.39e+03 | 0.898 | 0.0112 | 0.89 | attn:hbm |
| SRAM+COMP | +1024 MB SRAM (pins weights), KV4g64+R128+quest16384 | 6 | 10.2 | 263 | -3.25e+03 | 0.317 | 0.00107 | 0.89 | linear:hbm |
| COMP | KV4g64+R128+quest16384 | 6 | 9.82 | 0 | -3.29e+03 | 0.11 | 0.000373 | 0.89 | linear:hbm |
| SRAM+HBM | +1024 MB SRAM (pins KV), +2 HBM stack | 2 | 1.69 | 401 | -802 | 1.06 | 0.012 | 0 | attn:hbm |
| HBM | +2 HBM stack | 2 | 1.67 | 138 | -1.1e+03 | 0.975 | 0.0116 | 0 | attn:hbm |
| SRAM | +1024 MB SRAM (pins KV) | 1 | 1.02 | 263 | 469 | 0.122 | 0.000412 | 0 | attn:hbm |
| baseline | baseline | 1 | 1 | 0 | 0 | 0 | 0 | 0 | attn:hbm |

### Iso-area winners (best tokens/s under an added-area budget, quality-gated)

| budget mm2 | config | B | speedup | used mm2 | mJ/token | quality % | measured? |
|---|---|---|---|---|---|---|---|
| 0 | KV4g64+R128+quest16384 | 6 | 9.82 | 0 | 377 | 0.89 | True |
| 25 | +64 MB SRAM (pins weights), KV4g64+R128+quest16384 | 6 | 9.84 | 16.4 | 379 | 0.89 | True |
| 50 | +128 MB SRAM (pins weights), KV4g64+R128+quest16384 | 6 | 9.86 | 32.8 | 382 | 0.89 | True |
| 100 | +64 MB SRAM (pins weights), +1 HBM stack, KV4g64+R128+quest16384 | 8 | 13.6 | 85.4 | 298 | 0.89 | True |
| 200 | +128 MB SRAM (pins weights), +2 HBM stack, KV4g64+R128+quest16384 | 8 | 15.9 | 171 | 276 | 0.89 | True |
| 400 | +512 MB SRAM (pins weights), +2 HBM stack, KV4g64+R128+quest16384 | 8 | 16.1 | 269 | 287 | 0.89 | True |

* Best single-knob **SRAM** efficiency: +1024 MB SRAM (pins KV) -> speedup 1.024x for 263 mm2 (9.04e-05 speedup/mm2; dAI/mm2 = 4.64e-04, d(AI/ridge)/mm2 = 1.57e-06).
* Best single-knob **HBM** efficiency: +1 HBM stack -> speedup 1.428x for 69 mm2 (6.20e-03 speedup/mm2; dAI/mm2 = 1.41e-02, d(AI/ridge)/mm2 = 1.08e-04).
