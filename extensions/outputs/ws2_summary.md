# Workstream 2 summary (auto-generated)

Assumptions: see header of frontier_schemes.py. Expected Attention and StreamingLLM are identical in the engine (quality differs only).

Schemes with unmeasured quality (placeholder deltas, do not quote as gate-passing): AATC_Transform, ExpectedAttn_16k, Hybrid_AATC_ExpectedAttn16k, PM_KVQ_Mixed, StreamingLLM_16k

## Top 3 schemes by system throughput (quality NOT considered; unmeasured quality entries are placeholders)

| hw   | model            | regime   | scheme                      |   B_eff |   tokens_per_s |   speedup_total |   speedup_fixed_batch |   speedup_batch_expansion |   energy_mj_per_token |
|:-----|:-----------------|:---------|:----------------------------|--------:|---------------:|----------------:|----------------------:|--------------------------:|----------------------:|
| b200 | qwen2.5-7b       | C        | Hybrid_AATC_ExpectedAttn16k |       8 |        5514.16 |           11.16 |                 11.16 |                      1.00 |                120.85 |
| b200 | deepseek-v2-lite | C        | Hybrid_INT4_Quest16k        |       8 |        5184.53 |           11.46 |                 11.46 |                      1.00 |                138.30 |
| b200 | qwen2.5-7b       | C        | ExpectedAttn_16k            |       8 |        4920.85 |            9.96 |                  9.96 |                      1.00 |                142.42 |
| b200 | qwen2.5-7b       | C        | StreamingLLM_16k            |       8 |        4920.85 |            9.96 |                  9.96 |                      1.00 |                142.42 |
| b200 | deepseek-v2-lite | C        | Hybrid_AATC_ExpectedAttn16k |       8 |        4720.48 |           10.44 |                 10.44 |                      1.00 |                144.03 |
| b200 | deepseek-v2-lite | C        | StreamingLLM_16k            |       8 |        4239.01 |            9.37 |                  9.37 |                      1.00 |                167.53 |
| b200 | qwen2.5-7b       | A        | PM_KVQ_Mixed                |       8 |        4201.50 |            1.17 |                  1.17 |                      1.00 |                 94.61 |
| b200 | llama3.1-8b      | C        | Hybrid_AATC_ExpectedAttn16k |       8 |        4196.94 |           19.34 |                 11.65 |                      1.66 |                156.61 |
| b200 | qwen2.5-7b       | A        | KIVI_INT2_g32               |       8 |        4186.97 |            1.17 |                  1.17 |                      1.00 |                 94.31 |
| b200 | qwen2.5-7b       | A        | KIVI_INT4_g32               |       8 |        4186.97 |            1.17 |                  1.17 |                      1.00 |                 95.93 |
| b200 | llama3.1-8b      | A        | PM_KVQ_Mixed                |       8 |        3665.77 |            1.35 |                  1.35 |                      1.00 |                109.47 |
| b200 | llama3.1-8b      | A        | KIVI_INT2_g32               |       8 |        3640.58 |            1.34 |                  1.34 |                      1.00 |                108.79 |
| b200 | llama3.1-8b      | A        | Hybrid_INT4_Quest16k        |       8 |        3640.58 |            1.34 |                  1.34 |                      1.00 |                111.56 |
| b200 | deepseek-v2-lite | A        | PM_KVQ_Mixed                |       8 |        3465.16 |            1.07 |                  1.07 |                      1.00 |                111.99 |
| b200 | deepseek-v2-lite | A        | KIVI_INT4_g32               |       8 |        3459.79 |            1.07 |                  1.07 |                      1.00 |                112.71 |
| b200 | deepseek-v2-lite | A        | KIVI_INT2_g32               |       8 |        3459.79 |            1.07 |                  1.07 |                      1.00 |                111.83 |
| b200 | llama3.1-8b      | C        | StreamingLLM_16k            |       8 |        3441.53 |           15.86 |                 10.32 |                      1.54 |                206.62 |
| b200 | llama3.1-8b      | C        | ExpectedAttn_16k            |       8 |        3441.53 |           15.86 |                 10.32 |                      1.54 |                206.62 |
| b200 | qwen2.5-7b       | B        | Hybrid_AATC_ExpectedAttn16k |       8 |        3426.88 |            3.97 |                  3.97 |                      1.00 |                110.35 |
| b200 | deepseek-v2-lite | B        | Hybrid_INT4_Quest16k        |       8 |        3346.60 |            2.64 |                  2.64 |                      1.00 |                119.12 |
| b200 | qwen2.5-7b       | B        | Hybrid_INT4_Quest16k        |       8 |        3191.59 |            3.70 |                  3.70 |                      1.00 |                126.55 |
| b200 | deepseek-v2-lite | B        | Hybrid_AATC_ExpectedAttn16k |       8 |        3146.91 |            2.48 |                  2.48 |                      1.00 |                122.01 |
| b200 | qwen2.5-7b       | B        | StreamingLLM_16k            |       8 |        2966.99 |            3.44 |                  3.44 |                      1.00 |                132.15 |
| b200 | deepseek-v2-lite | B        | ExpectedAttn_16k            |       8 |        2927.29 |            2.31 |                  2.31 |                      1.00 |                133.73 |
| h100 | qwen2.5-7b       | C        | Hybrid_AATC_ExpectedAttn16k |       8 |        2923.09 |           14.86 |                  7.93 |                      1.87 |                144.94 |
| b200 | llama3.1-8b      | B        | Hybrid_AATC_ExpectedAttn16k |       8 |        2529.26 |            6.02 |                  6.02 |                      1.00 |                144.55 |
| h100 | deepseek-v2-lite | C        | Hybrid_INT4_Quest16k        |       8 |        2430.33 |           13.20 |                  8.91 |                      1.48 |                173.63 |
| h100 | deepseek-v2-lite | C        | Hybrid_AATC_ExpectedAttn16k |       8 |        2418.93 |           13.13 |                  8.88 |                      1.48 |                174.85 |
| h100 | llama3.1-8b      | C        | Hybrid_AATC_ExpectedAttn16k |       8 |        2396.81 |           30.70 |                  4.82 |                      6.37 |                182.00 |
| h100 | qwen2.5-7b       | C        | ExpectedAttn_16k            |       8 |        2279.35 |           11.59 |                  6.90 |                      1.68 |                180.14 |
| h100 | qwen2.5-7b       | C        | StreamingLLM_16k            |       8 |        2279.35 |           11.59 |                  6.90 |                      1.68 |                180.14 |
| b200 | llama3.1-8b      | B        | Hybrid_INT4_Quest16k        |       8 |        2244.31 |            5.34 |                  5.34 |                      1.00 |                181.39 |
| b200 | llama3.1-8b      | B        | ExpectedAttn_16k            |       8 |        1994.13 |            4.75 |                  4.75 |                      1.00 |                194.78 |
| h100 | deepseek-v2-lite | C        | StreamingLLM_16k            |       8 |        1928.29 |           10.47 |                  7.60 |                      1.38 |                213.11 |
| h100 | qwen2.5-7b       | A        | KIVI_INT2_g32               |       8 |        1802.85 |            1.20 |                  1.20 |                      1.00 |                122.20 |
| h100 | qwen2.5-7b       | A        | PM_KVQ_Mixed                |       8 |        1785.66 |            1.19 |                  1.19 |                      1.00 |                123.20 |
| h100 | qwen2.5-7b       | A        | KVQuant_INT4_g64            |       8 |        1761.75 |            1.17 |                  1.17 |                      1.00 |                125.56 |
| h100 | qwen2.5-7b       | B        | Hybrid_AATC_ExpectedAttn16k |       8 |        1632.86 |            4.65 |                  4.14 |                      1.12 |                137.83 |
| h100 | llama3.1-8b      | A        | KIVI_INT2_g32               |       8 |        1605.51 |            1.41 |                  1.41 |                      1.00 |                139.55 |
| h100 | llama3.1-8b      | A        | PM_KVQ_Mixed                |       8 |        1574.65 |            1.38 |                  1.38 |                      1.00 |                141.84 |
| h100 | llama3.1-8b      | C        | StreamingLLM_16k            |       8 |        1561.69 |           20.00 |                  4.46 |                      4.48 |                262.92 |
| h100 | llama3.1-8b      | C        | ExpectedAttn_16k            |       8 |        1561.69 |           20.00 |                  4.46 |                      4.48 |                262.92 |
| h100 | llama3.1-8b      | A        | KVQuant_INT4_g64            |       8 |        1532.72 |            1.35 |                  1.35 |                      1.00 |                147.22 |
| h100 | deepseek-v2-lite | A        | KIVI_INT2_g32               |       8 |        1468.26 |            1.09 |                  1.09 |                      1.00 |                145.50 |
| h100 | deepseek-v2-lite | A        | PM_KVQ_Mixed                |       8 |        1462.04 |            1.08 |                  1.08 |                      1.00 |                146.04 |
| h100 | deepseek-v2-lite | A        | KVQuant_INT4_g64            |       8 |        1453.28 |            1.08 |                  1.08 |                      1.00 |                147.32 |
| h100 | deepseek-v2-lite | B        | Hybrid_INT4_Quest16k        |       8 |        1407.03 |            2.65 |                  2.65 |                      1.00 |                155.45 |
| h100 | deepseek-v2-lite | B        | Hybrid_AATC_ExpectedAttn16k |       8 |        1402.43 |            2.64 |                  2.64 |                      1.00 |                156.12 |
| h100 | qwen2.5-7b       | B        | Hybrid_INT4_Quest16k        |       8 |        1335.99 |            3.80 |                  3.45 |                      1.10 |                165.33 |
| h100 | llama3.1-8b      | B        | Hybrid_AATC_ExpectedAttn16k |       8 |        1329.59 |            8.80 |                  3.93 |                      2.24 |                173.83 |
| h100 | qwen2.5-7b       | B        | ExpectedAttn_16k            |       8 |        1239.07 |            3.53 |                  3.23 |                      1.09 |                173.18 |
| h100 | deepseek-v2-lite | B        | StreamingLLM_16k            |       8 |        1222.53 |            2.30 |                  2.30 |                      1.00 |                175.22 |
| h100 | llama3.1-8b      | B        | Hybrid_INT4_Quest16k        |       8 |         939.56 |            6.22 |                  3.31 |                      1.88 |                236.65 |
| h100 | llama3.1-8b      | B        | ExpectedAttn_16k            |       8 |         833.52 |            5.52 |                  3.11 |                      1.78 |                254.89 |

## Best scheme that PASSES the 1% gate with MEASURED quality

| hw   | model            | regime   | scheme               |   B_eff |   tokens_per_s |   speedup_total |   speedup_fixed_batch |   speedup_batch_expansion |   energy_mj_per_token |
|:-----|:-----------------|:---------|:---------------------|--------:|---------------:|----------------:|----------------------:|--------------------------:|----------------------:|
| b200 | deepseek-v2-lite | C        | Hybrid_INT4_Quest16k |       8 |        5184.53 |           11.46 |                 11.46 |                      1.00 |                138.30 |
| b200 | qwen2.5-7b       | A        | Hybrid_INT4_Quest16k |       8 |        4186.97 |            1.17 |                  1.17 |                      1.00 |                 95.53 |
| b200 | llama3.1-8b      | A        | Hybrid_INT4_Quest16k |       8 |        3640.58 |            1.34 |                  1.34 |                      1.00 |                111.56 |
| b200 | qwen2.5-7b       | C        | Hybrid_INT4_Quest16k |       8 |        3568.43 |            7.22 |                  7.22 |                      1.00 |                203.70 |
| b200 | deepseek-v2-lite | A        | Hybrid_INT4_Quest16k |       8 |        3459.79 |            1.07 |                  1.07 |                      1.00 |                112.49 |
| b200 | deepseek-v2-lite | B        | Hybrid_INT4_Quest16k |       8 |        3346.60 |            2.64 |                  2.64 |                      1.00 |                119.12 |
| b200 | qwen2.5-7b       | B        | Hybrid_INT4_Quest16k |       8 |        3191.59 |            3.70 |                  3.70 |                      1.00 |                126.55 |
| h100 | deepseek-v2-lite | C        | Hybrid_INT4_Quest16k |       8 |        2430.33 |           13.20 |                  8.91 |                      1.48 |                173.63 |
| b200 | llama3.1-8b      | B        | Hybrid_INT4_Quest16k |       8 |        2244.31 |            5.34 |                  5.34 |                      1.00 |                181.39 |
| b200 | llama3.1-8b      | C        | Hybrid_INT4_Quest16k |       8 |        2151.73 |            9.91 |                  7.41 |                      1.34 |                345.35 |
| h100 | qwen2.5-7b       | A        | KVQuant_INT4_g64     |       8 |        1761.75 |            1.17 |                  1.17 |                      1.00 |                125.56 |
| h100 | qwen2.5-7b       | C        | Hybrid_INT4_Quest16k |       8 |        1611.52 |            8.19 |                  5.53 |                      1.48 |                259.03 |
| h100 | llama3.1-8b      | A        | Hybrid_INT4_Quest16k |       8 |        1532.72 |            1.35 |                  1.35 |                      1.00 |                145.63 |
| h100 | deepseek-v2-lite | A        | KVQuant_INT4_g64     |       8 |        1453.28 |            1.08 |                  1.08 |                      1.00 |                147.32 |
| h100 | deepseek-v2-lite | B        | Hybrid_INT4_Quest16k |       8 |        1407.03 |            2.65 |                  2.65 |                      1.00 |                155.45 |
| h100 | qwen2.5-7b       | B        | Hybrid_INT4_Quest16k |       8 |        1335.99 |            3.80 |                  3.45 |                      1.10 |                165.33 |
| h100 | llama3.1-8b      | B        | Hybrid_INT4_Quest16k |       8 |         939.56 |            6.22 |                  3.31 |                      1.88 |                236.65 |
| h100 | llama3.1-8b      | C        | Hybrid_INT4_Quest16k |       6 |         861.47 |           11.03 |                  3.89 |                      2.84 |                483.58 |
