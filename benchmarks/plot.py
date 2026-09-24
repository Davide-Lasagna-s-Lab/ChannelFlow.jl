"""Plot the final-source complete-step sweep and generate its documentation table."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

repo = Path(__file__).resolve().parents[1]
root = repo / "benchmarks/results/final"

def read(name):
    with (root / name).open() as f:
        return {int(r["Nx"]): r for r in csv.DictReader(f)}

cpu1, gpu = (read(name) for name in ("cpu-1.csv", "gpu.csv"))
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 200})
fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
for data, label, color, marker in (
    (cpu1, "CPU: 1 FFT/BLAS thread", "#1565c0", "o"),
    (gpu, "NVIDIA A100", "#c62828", "^"),
):
    x = sorted(data)
    axes[0].loglog(x, [1000*float(data[n]["min_seconds"]) for n in x],
                   color=color, marker=marker, markersize=4, label=label)
for data, label, color, marker in (
    (cpu1, "CPU: 1 FFT/BLAS thread", "#1565c0", "o"),
):
    x = sorted(data.keys() & gpu.keys())
    axes[1].loglog(x, [float(data[n]["min_seconds"])/float(gpu[n]["min_seconds"]) for n in x],
                    color=color, marker=marker, markersize=4, label=label)
axes[0].set_ylabel("Complete CNRK2 step [ms]")
axes[1].set_ylabel("CPU time / GPU time")
axes[1].axhline(1, color="0.5", lw=.8)
for ax in axes:
    ax.set_xlabel(r"$N_x=N_z$  ($N_y=N_x+1$)")
    ticks = sorted(gpu) if ax is axes[0] else sorted(cpu1)
    ax.set_xticks(ticks, labels=ticks)
    ax.minorticks_off()
    ax.grid(alpha=.2)
    ax.legend(frameon=False, fontsize=8)
for ext in ("svg", "png"):
    fig.savefig(root / ("timestep-cost."+ext))
plt.close(fig)

rows = []
for n in sorted(gpu):
    g = float(gpu[n]["min_seconds"])
    c = float(cpu1[n]["min_seconds"]) if n in cpu1 else None
    cost = f"{1000*c:.3f}" if c is not None else "—"
    speed = f"{c/g:.2f}×" if c is not None else "—"
    rows.append(f"| `{n}×{n+1}×{n}` | {cost} | {1000*g:.3f} | {speed} |")
table = "\n".join(rows)
text = r'''# CPU and GPU benchmarks

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
| CPU transforms | FFTW, one FFT/BLAS thread; one Julia thread |
| GPU transforms | cuFFT, including even-extension Chebyshev transforms |
| Sampling | minimum of 100 samples after five warm-up steps |

Measurements were taken sequentially on the same node on 2026-09-24, in two
allocations: N=8–128 in job `1643121`, and N=192–256 in job `1643303`. Each
grid size has paired CPU/GPU measurements from its allocation, using the same
solver source. The raw CSVs retain the source revision; the
[environment](assets/benchmarks/environment.txt) and
[dependency manifest](assets/benchmarks/Manifest.toml) record the software stack.

## Complete-step cost

![Complete timestep cost and CPU/GPU ratio](assets/benchmarks/timestep-cost.svg)

| Resolved grid | CPU, 1 thread [ms] | GPU [ms] | CPU / GPU |
|---|---:|---:|---:|
TABLE

The GPU has a fixed launch/dispatch cost, so very small problems can run faster
on the CPU. Larger grids amortize that cost. The CPU series uses one thread
throughout, including FFTW and BLAS; CPU kernels may still use SIMD.

Ratios compare this Julia code on the stated CPU and GPU, not C++ Channelflow.
All seven sizes include CPU and GPU measurements. At the largest size, the A100
takes approximately **116 ms** per step, versus **15.01 s** on the serial CPU:
approximately **130×** faster. A ``512\times513\times512`` extension is scheduled;
no extrapolated timing or speedup is included for that case. Its persistent
problem storage alone is estimated at about 72 GiB by cubic scaling from N=32,
excluding states and CUDA workspace. It may reach the A100 memory limit.
Grid sizes are resolved sizes; padded workspaces consume
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
CHANNEL_SAMPLES=100 CHANNEL_SIZES=8,16,32,64,128,192,256,512 CHANNEL_FFT_THREADS=1 \
  julia --project=. benchmarks/step.jl cpu cpu-1.csv
CHANNEL_SAMPLES=100 CHANNEL_SIZES=8,16,32,64,128,192,256,512 \
  julia --project=test/cuda benchmarks/step.jl cuda gpu.csv
```

Use the same node and run configurations sequentially. FFT planning, clock
frequencies and competing work can affect results. The minimum estimates warmed
best-case cost; it is not a confidence interval or a promise for every production
step. Timing instrumentation and sampling profilers also perturb execution.

The repository provides `benchmarks/iridis.slurm` as a scheduler template and
`benchmarks/plot.py` to regenerate the figure and this table from the saved CSVs.
The dependency manifest is provenance, not an environment to activate directly.

## Profiling

The scheduler template also profiles complete warmed steps at
``N=64,128,256``, using one FFT/BLAS thread on CPU and the A100.
These runs are separate from the throughput measurements above.

Each CPU case saves an exclusive category-count CSV, folded call stacks and a
text report with the flat profile and call tree. GPU cases save kernel durations
and call counts, plus the CUDA profiler's host and device activity report.
`CHANNEL_PROFILE_STEPS` controls the number of profiled steps (default: 20).
Generate a readable summary with:

```sh
python benchmarks/profile_report.py benchmarks/results/final
```

The generated `profiling.md` lists CPU sample shares and GPU kernel-time shares.
CPU sampling is approximate and does not sample library workers as Julia stacks.
GPU activity fractions exclude launch overhead and idle gaps; they are not
fractions of whole-step wall time. Do not add host and device durations together.

## Data and verification

- [CPU, one thread](assets/benchmarks/cpu-1.csv)
- [A100](assets/benchmarks/gpu.csv)

The [validation suite](validation.md) separately checks accuracy and CPU/GPU
agreement. Benchmark completion at the largest sizes is a timing/stability
smoke check, not a high-resolution turbulent validation. See the [developer
guide](contributing.md) for profiling and numerical-test entry points.
'''.replace('TABLE',table)
(repo / 'docs/src/benchmarks.md').write_text(text)
