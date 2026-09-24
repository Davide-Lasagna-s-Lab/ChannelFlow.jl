#!/bin/bash -l
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --job-name=channel-benchmark
#SBATCH --output=channel-benchmark-%j.log
set -euo pipefail
# Submit from the repository root after instantiating test/cuda on a login node.
# Adapt the module, partition and optional account to your cluster.
module load julia/1.12.4
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CHANNEL_SAMPLES=100 CHANNEL_SOURCE_COMMIT=$(git rev-parse HEAD)
mkdir -p benchmarks/results/final
julia --startup-file=no --project=test/cuda test/cuda/runtests.jl
for threads in 1; do
    CHANNEL_SIZES=8,16,32,64,128,192,256,512 CHANNEL_FFT_THREADS=$threads julia --startup-file=no --project=test/cuda \
      benchmarks/step.jl cpu benchmarks/results/final/cpu-$threads.csv
done
CHANNEL_SIZES=8,16,32,64,128,192,256,512 \
    julia --startup-file=no --project=test/cuda benchmarks/step.jl cuda benchmarks/results/final/gpu.csv
# Profiles run separately, after all uninstrumented measurements.
for N in 64 128 256; do
    for threads in 1; do
        prefix=benchmarks/results/final/profile-cpu-$threads-N$N
        CHANNEL_PROFILE_N=$N CHANNEL_FFT_THREADS=$threads CHANNEL_PROFILE_CSV=$prefix.csv \
            julia --startup-file=no --project=test/cuda benchmarks/profile.jl cpu > "$prefix.txt" 2>&1
    done
    prefix=benchmarks/results/final/profile-gpu-N$N
    CHANNEL_PROFILE_N=$N CHANNEL_GPU_PROFILE_CSV=$prefix.csv \
        julia --startup-file=no --project=test/cuda benchmarks/profile.jl cuda > "$prefix.txt" 2>&1
done
