# DSE results: deepseek-v2-lite on h100

Quality source: **measured table data/quality_table.csv**; gate = 1.0%; theta = 1.0; rho = 0.0; batch mode = max; dequant FLOPs/elem = 6.0; attention unit = tensor.

## A_short_medium  (S = 8,192, chips = 1)

**Baseline** (BF16 KV, no extras): B = 8 (max feasible 155), TPOT = 9.83 ms, 814 tok/s, dominant = `linear:hbm`, AI = 5.1 FLOP/B (ridge 295), KV share of HBM bytes = 10%, S* = 71,864 tokens.

### Best gated config per knob family

| family | config | B | speedup | +mm2 | dE mJ/tok | dAI | d(AI/ridge) | quality % | bound |
|---|---|---|---|---|---|---|---|---|---|
| SRAM+HBM+COMP | +1024 MB SRAM (pins weights), +2 HBM stack, KV8+R128+streaming4096 | 8 | 1.6 | 401 | -37 | -0.867 | 0.00276 | 0.96 | linear:hbm |
| HBM+COMP | +2 HBM stack, KV8+R128+streaming4096 | 8 | 1.52 | 138 | -48.9 | -1.1 | 0.00165 | 0.96 | linear:hbm |
| SRAM+HBM | +1024 MB SRAM (pins KV), +2 HBM stack | 8 | 1.48 | 401 | -21.9 | 0.275 | 0.00818 | 0 | linear:hbm |
| HBM | +2 HBM stack | 8 | 1.4 | 138 | -35.4 | 0 | 0.00687 | 0 | linear:hbm |
| SRAM+COMP | +1024 MB SRAM (pins weights), KV8+R128+streaming4096 | 8 | 1.15 | 263 | 1.71 | -0.867 | -0.00294 | 0.96 | linear:hbm |
| COMP | KV8+R128+streaming4096 | 8 | 1.08 | 0 | -16.1 | -1.1 | -0.00373 | 0.96 | linear:hbm |
| SRAM | +1024 MB SRAM (pins KV) | 8 | 1.05 | 263 | 20.2 | 0.275 | 0.000933 | 0 | linear:hbm |
| baseline | baseline | 8 | 1 | 0 | 0 | 0 | 0 | 0 | linear:hbm |

### Iso-area winners (best tokens/s under an added-area budget, quality-gated)

| budget mm2 | config | B | speedup | used mm2 | mJ/token | quality % | measured? |
|---|---|---|---|---|---|---|---|
| 0 | KV8+R128+streaming4096 | 8 | 1.08 | 0 | 190 | 0.96 | False |
| 25 | +64 MB SRAM (pins KV), KV8+R128+streaming4096 | 8 | 1.09 | 16.4 | 191 | 0.96 | False |
| 50 | +128 MB SRAM (pins KV), KV8+R128+streaming4096 | 8 | 1.09 | 32.8 | 192 | 0.96 | False |
| 100 | +64 MB SRAM (pins KV), +1 HBM stack, KV8+R128+streaming4096 | 8 | 1.3 | 85.4 | 172 | 0.96 | False |
| 200 | +128 MB SRAM (pins KV), +2 HBM stack, KV8+R128+streaming4096 | 8 | 1.53 | 171 | 159 | 0.96 | False |
| 400 | +512 MB SRAM (pins weights), +2 HBM stack, KV8+R128+streaming4096 | 8 | 1.56 | 269 | 163 | 0.96 | False |

* Best single-knob **SRAM** efficiency: +1024 MB SRAM (pins KV) -> speedup 1.054x for 263 mm2 (2.07e-04 speedup/mm2; dAI/mm2 = 1.05e-03, d(AI/ridge)/mm2 = 3.55e-06).
* Best single-knob **HBM** efficiency: +1 HBM stack -> speedup 1.200x for 69 mm2 (2.90e-03 speedup/mm2; dAI/mm2 = 0.00e+00, d(AI/ridge)/mm2 = 4.98e-05).

## B_long  (S = 131,072, chips = 1)

**Baseline** (BF16 KV, no extras): B = 8 (max feasible 9), TPOT = 24.95 ms, 321 tok/s, dominant = `attn:hbm`, AI = 20.4 FLOP/B (ridge 295), KV share of HBM bytes = 65%, S* = 71,864 tokens.

### Best gated config per knob family

| family | config | B | speedup | +mm2 | dE mJ/tok | dAI | d(AI/ridge) | quality % | bound |
|---|---|---|---|---|---|---|---|---|---|
| SRAM+HBM+COMP | +1024 MB SRAM (pins weights), +2 HBM stack, KV16+quest4096 | 8 | 3.96 | 401 | -384 | -16.4 | -0.0503 | 0.88 | linear:hbm |
| HBM+COMP | +2 HBM stack, KV16+quest4096 | 8 | 3.74 | 138 | -396 | -16.6 | -0.0514 | 0.88 | linear:hbm |
| SRAM+COMP | +1024 MB SRAM (pins weights), KV16+quest4096 | 8 | 2.83 | 263 | -344 | -16.4 | -0.0557 | 0.88 | linear:hbm |
| COMP | KV16+quest4096 | 8 | 2.67 | 0 | -362 | -16.6 | -0.0564 | 0.88 | linear:hbm |
| SRAM+HBM | +1024 MB SRAM (pins KV), +2 HBM stack | 8 | 1.43 | 401 | -41.8 | 0.422 | 0.0296 | 0 | attn:hbm |
| HBM | +2 HBM stack | 8 | 1.4 | 138 | -89.9 | 0 | 0.0276 | 0 | attn:hbm |
| SRAM | +1024 MB SRAM (pins KV) | 8 | 1.02 | 263 | 68.6 | 0.422 | 0.00143 | 0 | attn:hbm |
| baseline | baseline | 8 | 1 | 0 | 0 | 0 | 0 | 0 | attn:hbm |

### Iso-area winners (best tokens/s under an added-area budget, quality-gated)

| budget mm2 | config | B | speedup | used mm2 | mJ/token | quality % | measured? |
|---|---|---|---|---|---|---|---|
| 0 | KV16+quest4096 | 8 | 2.67 | 0 | 194 | 0.88 | True |
| 25 | +64 MB SRAM (pins weights), KV16+quest4096 | 8 | 2.68 | 16.4 | 196 | 0.88 | True |
| 50 | +128 MB SRAM (pins weights), KV16+quest4096 | 8 | 2.69 | 32.8 | 197 | 0.88 | True |
| 100 | +64 MB SRAM (pins weights), +1 HBM stack, KV16+quest4096 | 8 | 3.22 | 85.4 | 176 | 0.88 | True |
| 200 | +128 MB SRAM (pins weights), +2 HBM stack, KV16+quest4096 | 8 | 3.77 | 171 | 162 | 0.88 | True |
| 400 | +512 MB SRAM (pins weights), +2 HBM stack, KV16+quest4096 | 8 | 3.85 | 269 | 167 | 0.88 | True |

* Best single-knob **SRAM** efficiency: +1024 MB SRAM (pins KV) -> speedup 1.021x for 263 mm2 (7.89e-05 speedup/mm2; dAI/mm2 = 1.61e-03, d(AI/ridge)/mm2 = 5.44e-06).
* Best single-knob **HBM** efficiency: +1 HBM stack -> speedup 1.200x for 69 mm2 (2.90e-03 speedup/mm2; dAI/mm2 = 0.00e+00, d(AI/ridge)/mm2 = 2.00e-04).

## C_ultra_long  (S = 524,288, chips = 1)

**Baseline** (BF16 KV, no extras): B = 2 (max feasible 2), TPOT = 19.73 ms, 101 tok/s, dominant = `attn:hbm`, AI = 25.0 FLOP/B (ridge 295), KV share of HBM bytes = 82%, S* = 117,925 tokens.

### Best gated config per knob family

| family | config | B | speedup | +mm2 | dE mJ/tok | dAI | d(AI/ridge) | quality % | bound |
|---|---|---|---|---|---|---|---|---|---|
| SRAM+HBM+COMP | +1024 MB SRAM (pins weights), +2 HBM stack, KV4g64+R128+quest16384 | 8 | 12.4 | 401 | -1.61e+03 | -15.3 | -0.0386 | 0.89 | linear:hbm |
| HBM+COMP | +2 HBM stack, KV4g64+R128+quest16384 | 8 | 11.8 | 138 | -1.63e+03 | -15.8 | -0.0411 | 0.89 | linear:hbm |
| SRAM+COMP | +1024 MB SRAM (pins weights), KV4g64+R128+quest16384 | 8 | 8.87 | 263 | -1.57e+03 | -15.3 | -0.0518 | 0.89 | linear:hbm |
| COMP | KV4g64+R128+quest16384 | 8 | 8.4 | 0 | -1.59e+03 | -15.8 | -0.0535 | 0.89 | linear:hbm |
| SRAM+HBM | +1024 MB SRAM (pins KV), +2 HBM stack | 4 | 1.48 | 401 | -180 | 1.35 | 0.0403 | 0 | attn:hbm |
| HBM | +2 HBM stack | 4 | 1.46 | 138 | -336 | 1 | 0.0386 | 0 | attn:hbm |
| SRAM | +1024 MB SRAM (pins KV) | 2 | 1.03 | 263 | 208 | 0.658 | 0.00223 | 0 | attn:hbm |
| baseline | baseline | 2 | 1 | 0 | 0 | 0 | 0 | 0 | attn:hbm |

### Iso-area winners (best tokens/s under an added-area budget, quality-gated)

| budget mm2 | config | B | speedup | used mm2 | mJ/token | quality % | measured? |
|---|---|---|---|---|---|---|---|
| 0 | KV4g64+R128+quest16384 | 8 | 8.4 | 0 | 202 | 0.89 | True |
| 25 | +64 MB SRAM (pins weights), KV4g64+R128+quest16384 | 8 | 8.42 | 16.4 | 203 | 0.89 | True |
| 50 | +128 MB SRAM (pins weights), KV4g64+R128+quest16384 | 8 | 8.45 | 32.8 | 205 | 0.89 | True |
| 100 | +64 MB SRAM (pins weights), +1 HBM stack, KV4g64+R128+quest16384 | 8 | 10.1 | 85.4 | 183 | 0.89 | True |
| 200 | +128 MB SRAM (pins weights), +2 HBM stack, KV4g64+R128+quest16384 | 8 | 11.8 | 171 | 170 | 0.89 | True |
| 400 | +512 MB SRAM (pins weights), +2 HBM stack, KV4g64+R128+quest16384 | 8 | 12.1 | 269 | 175 | 0.89 | True |

* Best single-knob **SRAM** efficiency: +1024 MB SRAM (pins KV) -> speedup 1.026x for 263 mm2 (1.00e-04 speedup/mm2; dAI/mm2 = 2.51e-03, d(AI/ridge)/mm2 = 8.49e-06).
* Best single-knob **HBM** efficiency: +1 HBM stack -> speedup 1.229x for 69 mm2 (3.32e-03 speedup/mm2; dAI/mm2 = 8.88e-03, d(AI/ridge)/mm2 = 2.82e-04).
