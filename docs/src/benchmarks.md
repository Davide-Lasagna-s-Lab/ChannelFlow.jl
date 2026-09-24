# CPU and GPU benchmarks

These measurements time one **complete three-stage CNRK2 step**: nonlinear
products, Fourier/Chebyshev transforms, pressure–velocity coupling, influence/tau
correction and the mean-mode solve. Times are for the entire grid, not per mode
or per Helmholtz system. They measure the implementation documented by this site.

## Hardware and numerical configuration

| Setting | Value |
|---|---|
| CPU | Intel Xeon Gold 6336Y; four allocated host cores |
| GPU | NVIDIA A100 80 GB PCIe |
| Host | IRIDIS X, `rose03` |
| Software | Julia 1.12.4, CUDA.jl 6.4.0 |
| Precision | `Float64` / `ComplexF64` |
| Flow | Couette, rotational form, prescribed zero pressure gradient |
| Parameters | ``\nu=1/400``, ``\Delta t=0.002``, ``L_x=L_z=2\pi`` |
| Resolved grid | ``(N_x,N_y,N_z)=(N,N+1,N)`` |
| Nonlinear grid | 3/2 padding in x and z; unchanged y |
| CPU transforms | FFTW, one or four FFT/BLAS threads; one Julia thread |
| GPU transforms | cuFFT, including even-extension Chebyshev transforms |
| Sampling | minimum of 100 samples after five warm-up steps |

All series were measured sequentially in the same GPU-node allocation on
2026-09-24. The raw CSVs retain the exact solver source revision; the
[environment](assets/benchmarks/environment.txt) and
[dependency manifest](assets/benchmarks/Manifest.toml) record the software stack.

## Complete-step cost

![Complete timestep cost and CPU/GPU ratio](assets/benchmarks/timestep-cost.svg)

| Resolved grid | CPU, 1 thread [ms] | CPU, 4 FFT/BLAS threads [ms] | GPU [ms] | Faster measured CPU / GPU |
|---|---:|---:|---:|---:|
| `8×9×8` | 0.261 | 1.593 | 2.533 | 0.10× |
| `16×17×16` | 2.209 | 13.411 | 2.613 | 0.85× |
| `32×33×32` | 16.417 | 41.191 | 2.889 | 5.68× |
| `64×65×64` | 169.633 | 190.692 | 5.745 | 29.53× |
| `128×129×128` | 1629.038 | 1236.019 | 19.938 | 61.99× |
| `192×193×192` | — | — | 52.956 | — |
| `256×257×256` | — | — | 115.950 | — |

The GPU has a fixed launch/dispatch cost, so very small problems can run faster
on the CPU. Larger grids amortize that cost. The four-thread setting applies
to FFTW and BLAS; the remaining CPU kernels use SIMD and are not parallel Julia
loops. More library threads therefore need not improve small-grid timings.

Ratios compare this Julia code on the stated CPU and GPU, not C++ Channelflow.
CPU measurements stop at ``128\times129\times128``. The GPU-only sizes have no
reported CPU speedup. Grid sizes are resolved sizes; padded workspaces consume
additional memory. No extrapolation to the MKM590 production grid is implied.

## What is timed

1. Construct plans, factors and a smooth projected initial state outside timing.
2. Warm up five complete steps to exclude compilation.
3. Restore the same initial state before each sample, outside timing.
4. Synchronize the GPU before starting the timer and after the complete step.
5. Record the minimum of 100 timings. Raw data also retains their mean.

There is no monitor, diagnostic, disk output or full-field host transfer in the
timed step. These are fixed-gradient measurements: a fixed-flux problem adds
mean-mode reductions and should be measured separately for its own workload.

The `host_bytes` and `host_allocations` columns count Julia host allocations,
including CUDA bookkeeping; they do **not** measure device allocations or peak
memory. CPU measurements report 3,984 bytes and 75 allocations per step. GPU
host allocations are approximately 0.84–0.87 MB over this sweep. Device factors
and full-field workspaces are reused. Setup cost and peak device-memory capacity
are separate from the reported throughput.

## Reproduce

From the repository root in an environment with dependencies installed:

```sh
CHANNEL_SAMPLES=100 CHANNEL_SIZES=8,16,32,64,128 CHANNEL_FFT_THREADS=1 \
  julia --project=. benchmarks/step.jl cpu cpu-1.csv
CHANNEL_SAMPLES=100 CHANNEL_SIZES=8,16,32,64,128 CHANNEL_FFT_THREADS=4 \
  julia --project=. benchmarks/step.jl cpu cpu-4.csv
CHANNEL_SAMPLES=100 CHANNEL_SIZES=8,16,32,64,128,192,256 \
  julia --project=test/cuda benchmarks/step.jl cuda gpu.csv
```

Use the same node and run configurations sequentially. FFT planning, clock
frequencies and competing work can affect results. The minimum estimates warmed
best-case cost; it is not a confidence interval or a promise for every production
step. Timing instrumentation and sampling profilers also perturb execution.

The repository provides `benchmarks/iridis.slurm` as a scheduler template and
`benchmarks/plot.py` to regenerate the figure and this table from the saved CSVs.
The dependency manifest is provenance, not an environment to activate directly.

## Data and verification

- [CPU, one thread](assets/benchmarks/cpu-1.csv)
- [CPU, four FFT/BLAS threads](assets/benchmarks/cpu-4.csv)
- [A100](assets/benchmarks/gpu.csv)

The [validation suite](validation.md) separately checks accuracy and CPU/GPU
agreement. Benchmark completion at the GPU-only sizes is a timing/stability
smoke check, not a high-resolution turbulent validation. See the [developer
guide](contributing.md) for profiling and numerical-test entry points.
