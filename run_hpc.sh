#!/usr/bin/env bash
# ==============================================================================
# kvroofline: Turn-key HPC & Multi-GPU Benchmark Execution Script
# Team: Harsh Kapoor, Samyak Soni, Allu Joanna Preetika, Ashlin K Sunil
#
# Usage:
#   bash run_hpc.sh                 # Automatic hardware detection & full DSE
#   bash run_hpc.sh --quick         # Run validation & core experiments only
#   bash run_hpc.sh --benchmark-gpu # Execute live HuggingFace/vLLM GPU profiling
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment if present
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "================================================================================"
echo "kvroofline: Long-Context KV-Cache Roofline DSE (HPC Execution)"
echo "================================================================================"

# 1. Environment & Python check
PYTHON_BIN="python3"
if ! command -v python3 &> /dev/null; then
    if command -v python &> /dev/null; then
        PYTHON_BIN="python"
    else
        echo "Error: Python 3 is required but not found in PATH." >&2
        exit 1
    fi
fi
echo "[1/6] Using Python: $($PYTHON_BIN --version)"

# 2. Hardware Detection
HW_KEY="h100"
GPU_COUNT=1
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 || true)
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l || echo 1)
    echo "[2/6] Detected $GPU_COUNT NVIDIA GPU(s): $GPU_NAME"
    if [[ "$GPU_NAME" =~ "H100" ]]; then
        HW_KEY="h100"
    elif [[ "$GPU_NAME" =~ "H200" ]]; then
        HW_KEY="h200"
    elif [[ "$GPU_NAME" =~ "A100" ]]; then
        HW_KEY="a100-80"
    elif [[ "$GPU_NAME" =~ "B200" ]]; then
        HW_KEY="b200"
    elif [[ "$GPU_NAME" =~ "4090" ]]; then
        HW_KEY="rtx4090"
    elif [[ "$GPU_NAME" =~ "L4" ]]; then
        HW_KEY="l4"
    else
        echo "  Unrecognized GPU model; defaulting hardware preset to 'h100'."
        HW_KEY="h100"
    fi
else
    echo "[2/6] No active NVIDIA GPU found in current shell. Running analytical modeling for '$HW_KEY'."
fi
echo "  Active hardware preset: $HW_KEY (Chips: $GPU_COUNT)"

# 3. Unit Tests Verification
echo "[3/6] Running test suite to verify physical invariants..."
$PYTHON_BIN -m pytest -q tests

# 4. Optional Live GPU Measurement (if requested and torch available)
DO_GPU_BENCH=0
QUICK_MODE=0
for arg in "$@"; do
    case "$arg" in
        --benchmark-gpu) DO_GPU_BENCH=1 ;;
        --quick) QUICK_MODE=1 ;;
    esac
done

if [ "$DO_GPU_BENCH" -eq 1 ]; then
    echo "[4/6] Running live GPU profiling on hardware..."
    if $PYTHON_BIN -c "import torch; assert torch.cuda.is_available()" &> /dev/null; then
        echo "  Running HuggingFace decode latency benchmark across contexts..."
        $PYTHON_BIN scripts/measure_decode.py --selftest
        echo "  Running simulated KV quantization perplexity evaluation..."
        $PYTHON_BIN scripts/measure_quality.py --selftest
    else
        echo "  PyTorch CUDA not available. Skipping live GPU profiling."
    fi
else
    echo "[4/6] Skipping live GPU measurements (pass --benchmark-gpu to execute live profiling)."
fi

# 5. Core Experiments & Layerwise Reproduction
echo "[5/6] Executing core experiments & LLM-Viewer baseline reproduction..."
$PYTHON_BIN scripts/explore.py --hw "$HW_KEY"
$PYTHON_BIN scripts/layerwise_analysis.py --compare-llm-viewer
$PYTHON_BIN scripts/validate.py --hbm data/hbm_bytes.csv
$PYTHON_BIN scripts/validate.py --meas data/measurements.csv --calib data/calib.json
$PYTHON_BIN scripts/sensitivity.py --hw "$HW_KEY"

if [ "$QUICK_MODE" -eq 1 ]; then
    echo "Quick mode requested. Finished successfully."
    exit 0
fi

# 6. Full Design Space Exploration (DSE) Sweeps
echo "[6/6] Executing full DSE sweeps across models and context regimes..."
echo "  Sweeping LLaMA-3.1-8B on $HW_KEY..."
$PYTHON_BIN scripts/run_dse.py --model llama3.1-8b --hw "$HW_KEY" --quality-csv data/quality_table.csv --calib data/calib.json

echo "  Sweeping LLaMA-3-70B (8 chips) on $HW_KEY..."
$PYTHON_BIN scripts/run_dse.py --model llama3-70b --hw "$HW_KEY" --chips 8 --theta 0.5 --quality-csv data/quality_table.csv --calib data/calib.json

echo "  Sweeping DeepSeek-V2-Lite on $HW_KEY..."
$PYTHON_BIN scripts/run_dse.py --model deepseek-v2-lite --hw "$HW_KEY" --quality-csv data/quality_table.csv --calib data/calib.json

echo "  Sweeping Qwen-2.5-7B on $HW_KEY..."
$PYTHON_BIN scripts/run_dse.py --model qwen2.5-7b --hw "$HW_KEY" --quality-csv data/quality_table.csv --calib data/calib.json

echo "================================================================================"
echo "kvroofline execution completed successfully!"
echo "Outputs generated:"
echo "  - Technical Report:      docs/FINAL_REPORT.md"
echo "  - Exploration Figures:   results/explore/"
echo "  - Sweep CSVs & Pareto:   results/<model>_<hw>/"
echo "================================================================================"
