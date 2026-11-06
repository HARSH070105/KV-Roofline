#!/usr/bin/env bash
#SBATCH --job-name=kvroofline
#SBATCH --output=kvroofline_%j.log
#SBATCH --error=kvroofline_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --partition=gpu

# ==============================================================================
# Slurm Job Script for kvroofline Long-Context KV-Cache Roofline DSE
# ==============================================================================

echo "Job started on $(hostname) at $(date)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-none}"

# Load modules if on standard HPC environment (uncomment as appropriate)
# module load cuda/12.2 python/3.10

# Activate virtual environment if present
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run the turnkey execution script
bash run_hpc.sh --benchmark-gpu

echo "Job completed at $(date)"
