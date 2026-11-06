#!/usr/bin/env bash
# HBM bytes moved in ONE decode step, via Nsight Compute (deck slide 12: "Nsight Compute for HBM bytes").
#
#   bash scripts/ncu_hbm_bytes.sh <model-id> <kvroofline-key> <hw> <S> <B>
#   e.g. bash scripts/ncu_hbm_bytes.sh meta-llama/Llama-3.1-8B-Instruct llama3.1-8b h100 8192 2
#
# ncu replays every kernel (slow, 10-100x), so we profile exactly ONE NVTX-marked decode step.
# Then parse_ncu.py sums dram bytes and writes hbm_gb_per_step into data/hbm_bytes.csv, which
# scripts/validate.py compares to the model's hbm_gb_per_step_chip * n_chips.
set -euo pipefail
MODEL_ID=$1; KEY=$2; HW=$3; S=$4; B=$5
OUT=data/ncu_${KEY}_S${S}_B${B}.csv
mkdir -p data
ncu --nvtx --nvtx-include "decode_step/" \
    --metrics dram__bytes_read.sum,dram__bytes_write.sum \
    --csv --log-file "$OUT" \
    python scripts/measure_decode.py --model-id "$MODEL_ID" --key "$KEY" --hw "$HW" --S "$S" --B "$B" \
        --steps 1 --warmup 1 --nvtx --out /tmp/ncu_run_measure.csv
python scripts/parse_ncu.py "$OUT" --key "$KEY" --hw "$HW" --S "$S" --B "$B" --out data/hbm_bytes.csv
