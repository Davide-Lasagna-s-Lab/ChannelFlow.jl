# Complete DNS timestep benchmarks

See the [benchmark documentation](https://Davide-Lasagna-s-Lab.github.io/ChannelFlow.jl/benchmarks/)
for the CPU/A100 comparison, numerical configuration, measurement protocol,
interpretation and downloadable raw data. The [Markdown source](../docs/src/benchmarks.md)
is also readable offline.

`step.jl` measures a complete three-stage CNRK2 step, excluding setup and
monitoring, with explicit GPU synchronization. `CHANNEL_SAMPLES` controls the
sample count, `CHANNEL_SIZES` the resolved periodic sizes (Ny=N+1), and
`CHANNEL_FFT_THREADS` the FFTW/BLAS thread count. `iridis.slurm` is a cluster template.

The current complete sweep lives in `results/final/`. It uses the same numerical
source on CPU and GPU. Regenerate its figure and documentation table with:

```sh
python benchmarks/plot.py  # requires Matplotlib
```

For profiles of warmed steps:

```sh
CHANNEL_PROFILE_N=64 julia --project=. benchmarks/profile.jl cpu > cpu-profile.txt
CHANNEL_PROFILE_N=64 julia --project=test/cuda benchmarks/profile.jl cuda > gpu-profile.txt
```

Profiler instrumentation affects elapsed time. Use the unprofiled timing sweep
for throughput, and do not add overlapping host and device activity durations.

The scheduler template profiles N=64,128,256 after the timing sweep, using
1 and 4 FFT/BLAS threads on CPU and CUDA on GPU. `CHANNEL_PROFILE_STEPS` defaults
to 20 warmed steps; `CHANNEL_FFT_THREADS` also applies to CPU profiles. Each run
writes a full text log plus a CSV (`CHANNEL_PROFILE_CSV` on CPU,
`CHANNEL_GPU_PROFILE_CSV` on GPU). CPU runs additionally save folded call stacks.
The initial condition and numerical configuration match `step.jl`.

Generate a readable breakdown from these outputs:

```sh
python benchmarks/profile_report.py benchmarks/results/final
```

Read `results/final/profiling.md` for CPU sample shares and the most expensive
GPU kernels. GPU device-time shares do not include host overhead or idle gaps;
the full text log includes host API timings. CPU sample counts are estimates,
and library worker threads are not profiled as Julia call stacks.

For a matching serial C++ comparison, see [the upstream Channelflow driver](../validation/channelflow/README.md).
