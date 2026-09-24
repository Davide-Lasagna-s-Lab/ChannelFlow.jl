#!/bin/bash -l
#SBATCH --partition=a100
#SBATCH --account=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --job-name=channel-mkm590
#SBATCH --output=channel-mkm590-%j.log
set -euo pipefail
module load julia/1.12.4
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=4
# Submit from the ChannelFlow repository root. Instantiate test/cuda first.
julia --startup-file=no --project=test/cuda examples/mkm590/run.jl
