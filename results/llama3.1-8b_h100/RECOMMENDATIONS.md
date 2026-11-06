# DSE results: llama3.1-8b on h100

Quality source: **measured table data/quality_table.csv**; gate = 1.0%; theta = 1.0; rho = 0.0; batch mode = max; dequant FLOPs/elem = 6.0; attention unit = tensor.

## A_short_medium  (S = 8,192, chips = 1)

**Baseline** (BF16 KV, no extras): B = 8 (max feasible 51), TPOT = 11.65 ms, 687 tok/s, dominant = `linear:hbm`, AI = 6.6 FLOP/B (ridge 295), KV share of HBM bytes = 36%, S* = 14,314 tokens.

### Best gated config per knob family

| family | config | B | speedup | +mm2 | dE mJ/tok | dAI | d(AI/ridge) | quality % | bound |
|---|---|---|---|---|---|---|---|---|---|
| SRAM+HBM+COMP | +1024 MB SRAM (pins KV), +2 HBM stack, KV8+R128+streaming4096 | 8 | 2.04 | 401 | -83.2 | 2.71 | 0.0217 | 0.96 | linear:hbm |
| HBM+COMP | +2 HBM stack, KV8+R128+streaming4096 | 8 | 1.92 | 138 | -93.7 | 2.15 | 0.0191 | 0.96 | linear:hbm |
| SRAM+HBM | +1024 MB SRAM (pins KV), +2 HBM stack | 8 | 1.46 | 401 | -24.3 | 0.298 | 0.0103 | 0 | linear:hbm |
| SRAM+COMP | +1024 MB SRAM (pins KV), KV8+R128+streaming4096 | 8 | 1.46 | 263 | -47.1 | 2.71 | 0.00917 | 0.96 | linear:hbm |
| HBM | +2 HBM stack | 8 | 1.4 | 138 | -42 | 0 | 0.0089 | 0 | linear:hbm |
| COMP | KV8+R128+streaming4096 | 8 | 1.37 | 0 | -63.1 | 2.15 | 0.00729 | 0.96 | linear:hbm |
| SRAM | +1024 MB SRAM (pins KV) | 8 | 1.05 | 263 | 26.1 | 0.298 | 0.00101 | 0 | linear:hbm |
| baseline | baseline | 8 | 1 | 0 | 0 | 0 | 0 | 0 | linear:hbm |

### Iso-area winners (best tokens/s under an added-area budget, quality-gated)

| budget mm2 | config | B | speedup | used mm2 | mJ/token | quality % | measured? |
|---|---|---|---|---|---|---|---|
| 0 | KV8+R128+streaming4096 | 8 | 1.37 | 0 | 183 | 0.96 | False |
| 25 | +64 MB SRAM (pins KV), KV8+R128+streaming4096 | 8 | 1.38 | 16.4 | 184 | 0.96 | False |
| 50 | +128 MB SRAM (pins KV), KV8+R128+streaming4096 | 8 | 1.38 | 32.8 | 185 | 0.96 | False |
| 100 | +64 MB SRAM (pins KV), +1 HBM stack, KV8+R128+streaming4096 | 8 | 1.65 | 85.4 | 166 | 0.96 | False |
| 200 | +128 MB SRAM (pins KV), +2 HBM stack, KV8+R128+streaming4096 | 8 | 1.93 | 171 | 153 | 0.96 | False |
| 400 | +512 MB SRAM (pins KV), +2 HBM stack, KV8+R128+streaming4096 | 8 | 1.98 | 269 | 157 | 0.96 | False |

* Best single-knob **SRAM** efficiency: +1024 MB SRAM (pins KV) -> speedup 1.045x for 263 mm2 (1.73e-04 speedup/mm2; dAI/mm2 = 1.14e-03, d(AI/ridge)/mm2 = 3.85e-06).
* Best single-knob **HBM** efficiency: +1 HBM stack -> speedup 1.200x for 69 mm2 (2.90e-03 speedup/mm2; dAI/mm2 = 0.00e+00, d(AI/ridge)/mm2 = 6.45e-05).

## B_long  (S = 131,072, chips = 1)

**Baseline** (BF16 KV, no extras): B = 3 (max feasible 3), TPOT = 32.88 ms, 91 tok/s, dominant = `attn:hbm`, AI = 3.8 FLOP/B (ridge 295), KV share of HBM bytes = 77%, S* = 38,171 tokens.

### Best gated config per knob family

| family | config | B | speedup | +mm2 | dE mJ/tok | dAI | d(AI/ridge) | quality % | bound |
|---|---|---|---|---|---|---|---|---|---|
| SRAM+HBM+COMP | +1024 MB SRAM (pins weights), +2 HBM stack, KV4g64+R128+quest16384 | 8 | 9.04 | 401 | -1.55e+03 | 4.95 | 0.0286 | 0.66 | linear:hbm |
| HBM+COMP | +2 HBM stack, KV4g64+R128+quest16384 | 8 | 8.72 | 138 | -1.57e+03 | 4.64 | 0.0272 | 0.66 | linear:hbm |
| SRAM+COMP | +1024 MB SRAM (pins weights), KV4g64+R128+quest16384 | 8 | 6.46 | 263 | -1.49e+03 | 4.95 | 0.0168 | 0.66 | linear:hbm |
| COMP | KV4g64+R128+quest16384 | 8 | 6.23 | 0 | -1.52e+03 | 4.64 | 0.0157 | 0.66 | linear:hbm |
| SRAM+HBM | +1024 MB SRAM (pins KV), +2 HBM stack | 4 | 1.5 | 401 | -228 | 0.277 | 0.00647 | 0 | attn:hbm |
| HBM | +2 HBM stack | 4 | 1.48 | 138 | -399 | 0.227 | 0.00623 | 0 | attn:hbm |
| SRAM | +1024 MB SRAM (pins KV) | 3 | 1.02 | 263 | 251 | 0.0595 | 0.000202 | 0 | attn:hbm |
| baseline | baseline | 3 | 1 | 0 | 0 | 0 | 0 | 0 | attn:hbm |

### Iso-area winners (best tokens/s under an added-area budget, quality-gated)

| budget mm2 | config | B | speedup | used mm2 | mJ/token | quality % | measured? |
|---|---|---|---|---|---|---|---|
| 0 | KV4g64+R128+quest16384 | 8 | 6.23 | 0 | 307 | 0.66 | True |
| 25 | +64 MB SRAM (pins weights), KV4g64+R128+quest16384 | 8 | 6.24 | 16.4 | 309 | 0.66 | True |
| 50 | +128 MB SRAM (pins weights), KV4g64+R128+quest16384 | 8 | 6.25 | 32.8 | 311 | 0.66 | True |
| 100 | +64 MB SRAM (pins weights), +1 HBM stack, KV4g64+R128+quest16384 | 8 | 7.49 | 85.4 | 279 | 0.66 | True |
| 200 | +128 MB SRAM (pins weights), +2 HBM stack, KV4g64+R128+quest16384 | 8 | 8.75 | 171 | 259 | 0.66 | True |
| 400 | +512 MB SRAM (pins weights), +2 HBM stack, KV4g64+R128+quest16384 | 8 | 8.87 | 269 | 268 | 0.66 | True |

* Best single-knob **SRAM** efficiency: +1024 MB SRAM (pins KV) -> speedup 1.016x for 263 mm2 (5.96e-05 speedup/mm2; dAI/mm2 = 2.27e-04, d(AI/ridge)/mm2 = 7.68e-07).
* Best single-knob **HBM** efficiency: +1 HBM stack -> speedup 1.272x for 69 mm2 (3.94e-03 speedup/mm2; dAI/mm2 = 3.29e-03, d(AI/ridge)/mm2 = 5.07e-05).

## C_ultra_long  (S = 524,288, chips = 2)

**Baseline** (BF16 KV, no extras): B = 1 (max feasible 1), TPOT = 21.00 ms, 48 tok/s, dominant = `attn:hbm`, AI = 3.5 FLOP/B (ridge 295), KV share of HBM bytes = 82%, S* = 114,512 tokens.

### Best gated config per knob family

| family | config | B | speedup | +mm2 | dE mJ/tok | dAI | d(AI/ridge) | quality % | bound |
|---|---|---|---|---|---|---|---|---|---|
| SRAM+HBM+COMP | +1024 MB SRAM (pins weights), +2 HBM stack, KV4g64+R128+quest16384 | 8 | 17.6 | 401 | -6.43e+03 | 1.11 | 0.01 | 0.89 | attn:hbm |
| HBM+COMP | +2 HBM stack, KV4g64+R128+quest16384 | 8 | 17 | 138 | -6.48e+03 | 0.94 | 0.0092 | 0.89 | attn:hbm |
| SRAM+COMP | +1024 MB SRAM (pins weights), KV4g64+R128+quest16384 | 6 | 11.7 | 263 | -6.26e+03 | 0.763 | 0.00259 | 0.89 | attn:hbm |
| COMP | KV4g64+R128+quest16384 | 6 | 11.1 | 0 | -6.33e+03 | 0.567 | 0.00192 | 0.89 | attn:hbm |
| SRAM+HBM | +1024 MB SRAM (pins KV), +2 HBM stack | 2 | 1.56 | 401 | -1.1e+03 | 0.395 | 0.00661 | 0 | attn:hbm |
| HBM | +2 HBM stack | 2 | 1.54 | 138 | -1.72e+03 | 0.343 | 0.00636 | 0 | attn:hbm |
| SRAM | +1024 MB SRAM (pins weights) | 1 | 1.02 | 263 | 896 | 0.0878 | 0.000297 | 0 | attn:hbm |
| baseline | baseline | 1 | 1 | 0 | 0 | 0 | 0 | 0 | attn:hbm |

### Iso-area winners (best tokens/s under an added-area budget, quality-gated)

| budget mm2 | config | B | speedup | used mm2 | mJ/token | quality % | measured? |
|---|---|---|---|---|---|---|---|
| 0 | KV4g64+R128+quest16384 | 6 | 11.1 | 0 | 629 | 0.89 | True |
| 25 | +64 MB SRAM (pins weights), KV4g64+R128+quest16384 | 6 | 11.2 | 16.4 | 633 | 0.89 | True |
| 50 | +128 MB SRAM (pins weights), KV4g64+R128+quest16384 | 6 | 11.2 | 32.8 | 638 | 0.89 | True |
| 100 | +64 MB SRAM (pins weights), +1 HBM stack, KV4g64+R128+quest16384 | 8 | 14.6 | 85.4 | 523 | 0.89 | True |
| 200 | +128 MB SRAM (pins weights), +2 HBM stack, KV4g64+R128+quest16384 | 8 | 17 | 171 | 485 | 0.89 | True |
| 400 | +512 MB SRAM (pins weights), +2 HBM stack, KV4g64+R128+quest16384 | 8 | 17.3 | 269 | 503 | 0.89 | True |

* Best single-knob **SRAM** efficiency: +1024 MB SRAM (pins weights) -> speedup 1.025x for 263 mm2 (9.41e-05 speedup/mm2; dAI/mm2 = 3.34e-04, d(AI/ridge)/mm2 = 1.13e-06).
* Best single-knob **HBM** efficiency: +1 HBM stack -> speedup 1.324x for 69 mm2 (4.70e-03 speedup/mm2; dAI/mm2 = 4.97e-03, d(AI/ridge)/mm2 = 5.45e-05).
