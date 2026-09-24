#!/bin/bash -l
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=channel-benchmark
#SBATCH --output=channel-benchmark-%j.log
set -euo pipefail
# Submit from the repository root after instantiating test/cuda on a login node.
# Adapt the module, partition and optional account to your cluster.
module load julia/1.12.4
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CHANNEL_SAMPLES=100 CHANNEL_SOURCE_COMMIT=$(git rev-parse HEAD)
mkdir -p benchmarks/results
julia --startup-file=no --project=test/cuda test/cuda/runtests.jl
for threads in 1 4; do
    CHANNEL_FFT_THREADS=$threads julia --startup-file=no --project=test/cuda \
      benchmarks/step.jl cpu benchmarks/results/xeon-$threads.csv
done
CHANNEL_SIZES=8,16,24,32,48,64,96,128,192,256 \
    julia --startup-file=no --project=test/cuda benchmarks/step.jl cuda benchmarks/results/a100.csv
CHANNEL_PROFILE_CSV=benchmarks/results/xeon-phases.csv \
    julia --startup-file=no --project=test/cuda benchmarks/profile.jl cpu > benchmarks/results/xeon-profile.txt
julia --startup-file=no --project=test/cuda benchmarks/profile.jl cuda > benchmarks/results/a100-profile.txt
