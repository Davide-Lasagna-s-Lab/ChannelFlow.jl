#!/bin/bash -l
#SBATCH --partition=a100
#SBATCH --account=normal
#SBATCH --gres=gpu:1
#SBATCH --nodelist=rose03
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=channel-cpp
#SBATCH --output=channel-cpp-%j.log
set -euo pipefail
# Request the same node as the Julia baseline; no GPU work is performed here.
module load gcc/13.3.0 fftw/3.3.10 julia/1.12.4
export JULIA_DEPOT_PATH=/scratch/dl1f12/cheb-gpu/depot
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cpp=/scratch/dl1f12/channelflow-cpp
cd /scratch/dl1f12/channelflow-docs-final/code
out="$cpp/results"
mkdir -p "$out"
cp "$cpp/environment.txt" "$out/"
printf 'Nx,Ny,Nz,padded,samples,min_seconds,mean_seconds\n' > "$out/cpu-serial.csv"
for N in 8 16 32 64 128 192 256; do
    # Seed creation and file input are not timed. Reuse the exact Julia seed.
    julia --startup-file=no --project=test/cuda validation/channelflow/seed.jl \
        "$N" "$cpp/seed.bin" "$cpp/reference.bin"
    "$cpp/step-release" "$N" 100 "$cpp/seed.bin" "$cpp/reference.bin" \
        >> "$out/cpu-serial.csv" 2>> "$out/parity.txt"
    if [[ $N == 64 || $N == 128 || $N == 256 ]]; then
        # gprof counts instrumented C++ calls; FFTW itself is not instrumented.
        (cd "$out"; "$cpp/step-profile" "$N" 20 "$cpp/seed.bin" > "profile-N$N-run.txt"
         gprof "$cpp/step-profile" gmon.out > "profile-N$N.txt"
         mv gmon.out "profile-N$N.gmon")
    fi
    rm "$cpp/seed.bin" "$cpp/reference.bin"
done
