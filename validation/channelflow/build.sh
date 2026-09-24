#!/usr/bin/env bash
# Prerequisites: GCC, CMake, FFTW (double precision), Git, GNU gprof.
# Keep downloaded GPL upstream sources outside the Julia repository.
set -euo pipefail
root=${1:?Usage: build.sh scratch-directory fftw-prefix}
fftw=${2:?Supply the FFTW installation prefix}
mkdir -p "$root"
root=$(cd "$root" && pwd)
driver=$(cd "$(dirname "$0")" && pwd)/step.cpp
revision=ad37ef3022351d4e4a7a6c274c59a88605ad19e8
if [[ ! -d "$root/source/.git" ]]; then
    git clone https://github.com/epfl-ecps/channelflow.git "$root/source"
fi
git -C "$root/source" checkout "$revision"
if [[ ! -d "$root/eigen/.git" ]]; then
    git clone --depth 1 --branch 3.3.7 https://github.com/eigenteam/eigen-git-mirror.git "$root/eigen"
fi
for mode in release profile; do
    flags=-Wno-error=deprecated-declarations
    extra=()
    if [[ $mode == profile ]]; then
        flags="$flags -pg"
        extra=(-pg)
    fi
    build="$root/$mode"
    cmake -S "$root/source" -B "$build" \
        -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
        -DUSE_MPI=OFF -DWITH_NETCDF=OFF -DWITH_NSOLVER=OFF -DWITH_GTEST=OFF \
        -DCMAKE_CXX_FLAGS="$flags" -DEIGEN3_INCLUDE_DIR="$root/eigen" \
        -DFFTW_INCLUDE_DIR="$fftw/include" -DFFTW_LIBRARY="$fftw/lib/libfftw3.so"
    cmake --build "$build" --target chflow -j2
    g++ -O3 -DNDEBUG -std=c++11 "${extra[@]}" \
        -I "$root/source" -I "$build" -I "$root/eigen" -I "$fftw/include" \
        "$driver" "$build/channelflow/libchflow.a" \
        -L "$fftw/lib" -Wl,-rpath,"$fftw/lib" -lfftw3 -o "$root/step-$mode"
done
{
    git -C "$root/source" rev-parse HEAD
    g++ --version
    cmake --version
    ldd "$root/step-release"
} > "$root/environment.txt"
