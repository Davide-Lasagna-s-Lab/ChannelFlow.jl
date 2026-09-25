#!/usr/bin/env bash
# Run from repository root after building upstream with build.sh.
set -euo pipefail
exe=${1:?Usage: run.sh /path/to/step-release}
out=validation/results/cpp
mkdir -p "$out"
for N in 16 32 64 128 192 256; do
    julia --startup-file=no --project=. validation/channelflow/seed.jl "$N" "$out/seed.bin" "$out/reference.bin"
    "$exe" "$N" 1 "$out/seed.bin" "$out/reference.bin" > "$out/N$N.csv" 2> "$out/N$N.txt"
done
rm "$out/seed.bin" "$out/reference.bin"
python3 validation/channelflow/summarize.py "$out"
