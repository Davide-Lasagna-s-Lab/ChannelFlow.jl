#!/bin/bash -l
#SBATCH --partition=a100
#SBATCH --account=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --nodelist=rose03
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --job-name=channel-512
#SBATCH --output=/scratch/dl1f12/channelflow-docs-final/512.log
set -euo pipefail
module load julia/1.12.4
export JULIA_DEPOT_PATH=/scratch/dl1f12/cheb-gpu/depot
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export CHANNEL_SIZES=512 CHANNEL_SAMPLES=100 CHANNEL_FFT_THREADS=1
export CHANNEL_SOURCE_COMMIT=fe2d4d3
cd /scratch/dl1f12/channelflow-docs-final/code
out=/scratch/dl1f12/channelflow-docs-final/512-results
mkdir -p "$out"
nvidia-smi > "$out/gpu-environment.txt"
# Each process owns its storage. Preserve a GPU capacity failure and still run CPU.
if julia --startup-file=no --project=test/cuda benchmarks/step.jl cuda "$out/gpu.csv" > "$out/gpu.log" 2>&1; then
    echo 'GPU completed' > "$out/status.txt"
else
    code=$?
    echo "GPU failed with exit $code; inspect gpu.log (no timing extrapolated)" > "$out/status.txt"
fi
julia --startup-file=no --project=test/cuda benchmarks/step.jl cpu "$out/cpu-1.csv" > "$out/cpu.log" 2>&1
echo 'CPU completed' >> "$out/status.txt"
