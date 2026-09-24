# Serial C++ Channelflow comparison

This standalone client uses [upstream Channelflow](https://github.com/epfl-ecps/channelflow)
at `ad37ef3022351d4e4a7a6c274c59a88605ad19e8`. Upstream source is downloaded
separately; the Julia solver is unchanged.

## Matching the numerical problem

- Couette flow, walls y=±1, Lx=Lz=2π, ν=1/400, Δt=0.002.
- Three-stage CNRK2, rotational nonlinearity, tau correction, fixed zero pressure gradient.
- Julia resolved grid `(N,N+1,N)` corresponds to C++ physical grid
  `(3N/2,N+1,3N/2)` with `DealiasXZ`. Upstream retains |kx|,|kz|≤N/2−1.
  Neither run dealiasses y. Comparing equal constructor sizes would be unfair.
- `seed.jl` exports the same projected perturbation velocity and modified
  pressure used by the Julia timing driver. A second file contains a Julia
  one-step reference; the C++ driver reports physical-space maximum differences.
  Pressure is compared after removing a constant gauge offset.
  A preliminary N=32 check gives velocity max error 2.6e-15 and
  pressure max error 7.7e-12 after one step (double precision).
  This checks matching configuration, not general solver validation.
- Five warm-up steps and minimum/mean of 100 individual one-step calls.
  Field restoration, initialization, transforms for file exchange and disk I/O
  are outside timing. CNRK2's first-stage history coefficient is zero.
- Serial Release build, no MPI, one CPU/library thread, same Xeon node.
  Upstream `advance(fields,1)` includes its own temporary allocations; these
  are part of its public one-step interface and are included in the measurement.

## Build and run

With GCC, CMake and FFTW loaded:

```sh
bash validation/channelflow/build.sh /scratch/path/channelflow-cpp /path/to/fftw
```

`iridis.slurm` records the exact site paths used for the comparison. Submit it
with a dependency on the other timing/profiling jobs, so runs are sequential.
The build uses `-O3 -DNDEBUG` (CMake Release), with only deprecated-declaration
warnings made nonfatal for Eigen 3.3.7 on GCC 13. No numerical source patch is used.

## Profiling

A separate static build adds `-pg` for GNU gprof. It produces flat and call-graph
reports at N=64,128,256. Its timings must **not** be mixed with Release timings.
FFTW is dynamically linked and not instrumented: gprof's percentages cover the
profiled C++ executable, not a complete FFT-inclusive wall-time breakdown.
The reports also include initialization and warm-up; use them to locate C++
hotspots, not as exclusive step-phase percentages comparable to Julia's profiler.

Results are written under the external build directory's `results/`; publication
requires checking the parity output and matching hardware/software provenance first.

After retrieving the completed results, generate the serial-only comparison:

```sh
python validation/channelflow/plot.py results/cpu-serial.csv benchmarks/results/final/cpu-1.csv
```

Both panels use logarithmic axes. A C++/Julia ratio greater than one means Julia
is faster. Do not substitute the four-thread Julia measurements in this serial comparison.

## Numerical validation (independent of timing)

From the repository root, after the build:

```sh
bash validation/channelflow/run.sh /scratch/path/channelflow-cpp/step-release
python validation/plot.py
```

This regenerates one-step errors for N=8,16,32, preserving raw stdout/stderr in
`validation/results/cpp/` and summarizing errors in `validation/results/cpp.csv`.
Only the complete-step errors enter the validation figure, not the timing
columns from these short checks. Re-run both codes with the same seed files.
