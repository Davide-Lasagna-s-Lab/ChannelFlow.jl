# DNS step benchmark

Run from the package root, with its local dependencies available:

```sh
julia --startup-file=no --threads=1 --project=. perf/benchmark_step.jl
julia --startup-file=no --threads=1 --project=. perf/benchmark_step.jl --record --profile
julia --startup-file=no --threads=1 --project=. perf/benchmark_step.jl --cpu
```

`history.csv` is append-only: each run records a direct CNRK2 step and a
10-step `Flows.flow` propagation without a monitor. All values are **per
step**, including the latter case. Source and backend Git revisions, dirty
flags and SHA-256 source fingerprints distinguish committed code from local
experiments. Compilation, initialization, reset and explicit garbage
collection are outside the timed region; garbage collection triggered by
propagation itself is included. Nine samples follow three warm-up calls.
Times are milliseconds, memory is bytes, and allocation counts are heap
allocation events. The median and minimum are both retained because machine
load affects timings. Run on an otherwise idle machine for comparisons.

The fixed case follows the notebook's box and resolution: Couette flow,
Re=400, Lx=2π/1.14, Lz=2π/2.5, Ny=35, Nx=Nz=32, dt=0.025,
Float64, convective nonlinearity, one Julia thread and one BLAS thread.
A seeded random physical input (seed 42, scale 0.01) is transformed and
projected before timing. It is a reproducible performance input, not a
physical validation experiment. FFTW uses ESTIMATE to avoid costly planning;
keep this choice fixed across comparisons. The notebook is not modified.

`allocations.txt` reports one warmed step sampled with `Profile.Allocs` at
1%. Its counts/bytes are sampled values, not full totals; full totals come
from the timed measurements. `allocations-before.txt` preserves the initial
profile. Each entry is attributed to the first ChannelFlow or Chebyshev
source frame; inspect the indicated solve when investigating regressions.

## Measurements and changes

The initial working tree allocated 8,323,056 bytes in 110,001 allocations
per direct step. Removing function-valued real/imaginary broadcasts reduced
this to 5,638,176 bytes. Specializing the even/odd Chebyshev solves reduced
it to 1,642,464 bytes. Finally, wrapping modal views with the solver's known
polynomial degree eliminated the remaining direct-step allocations.
No factorization or numerical scheme was changed.

The pristine ChannelFlow commit `be56ce2` is also recorded separately
(`dirty_source=false`), using unchanged backend `90fd156`. Its direct-step
timing was unusually noisy; use the retained minimum and the 10-step case
alongside the initial working-tree measurement, rather than interpreting
its large median as an algorithmic speedup. The final working-tree rows
include the modified backend fingerprint. These are not new commits.

## Recorded result (2026-09-15, Julia 1.12.6, Apple M1)

| Case | Clean `be56ce2` median / minimum (ms) | Optimized working tree median / minimum (ms) | Bytes before → after | Allocations before → after |
|---|---:|---:|---:|---:|
| Direct step | 165.788 / 99.355 | 106.392 / 105.873 | 8,323,056 → 0 | 110,001 → 0 |
| Ten-step propagation, per step | 92.910 / 84.917 | 79.510 / 78.000 | 8,326,000 → 2,944 | 110,008 → 7 |

Single-step timings varied substantially between runs (the initial working
tree measured 85.958 ms, and an intermediate optimized run 79.556 ms).
Consequently the timing evidence is the more stable ten-step propagation:
about 14% lower median time, or 8% lower minimum. The allocation reduction
was consistent across runs. The residual 7 allocations / 2,944 bytes per
step belong to the propagation path; the direct solver measured zero.

Validation after these changes: 1,751 ChannelFlow interface/analytic checks,
1,767 Chebyshev backend checks, and 833 physics checks passed.

## CPU profiling and transform optimization (2026-09-16)

`--cpu` times individual kernels and samples 40 warmed steps with Julia's
CPU profiler. The kernel timings and flat profile (including native FFTW
frames) are saved to `cpu-profile.txt`. These kernel costs are diagnostic
per-call measurements, not additive timings of a complete step.
`cpu-before.txt` preserves the profile before these optimizations.
`--measure` selects FFTW.MEASURE; recorded case names acquire a `_measure`
suffix to distinguish them from the default ESTIMATE benchmark.

Before optimization, approximately 89% of sampled step stacks included the
nonlinear evaluation; transforms dominated. MEASURE alone reduced the
propagation cost from 76.74 to 71.01 ms/step. Representative kernel costs
with MEASURE were 21.11 ms per nonlinear evaluation, 1.58 ms per Stokes
solve, 0.23 ms per vector gradient, 0.82 ms per inverse Chebyshev transform
and 0.24 ms per inverse Fourier transform. Sampling percentages are
inclusive, and individual kernel timings are sensitive to CPU frequency.

Two changes target that cost without changing the resolved equations:

- Forward: truncate Fourier columns before the Chebyshev DCT, executing
  the DCT directly in the caller's output. Inverse: execute the DCT in a
  reusable resolved buffer before Fourier padding. These operations commute
  because they act on different axes. Each inverse plan now owns one extra
  resolved buffer (304,640 bytes for this case), allocated only at setup.
- Pad to `ceil(3N/2)` without forcing odd sizes: the benchmark uses 48 rather
  than 49 points in each periodic physical direction. The resolved grid and
  Nyquist filtering are unchanged. Recreate FFT plans and physical buffers
  after updating the package. With the retained Nyquist modes excluded,
  this padding still prevents quadratic aliases in the retained modes.

Moving the DCT reduced the ESTIMATE propagation measurement to 52.13
ms/step; removing forced odd padding reduced it further to 46.84 ms/step.
A dense complex-matrix DCT was also measured separately (0.53 versus
0.75 ms for the padded DCT kernel), but was not introduced: its extra
storage and different scaling with Ny were unnecessary for these changes.

The random benchmark input is sampled on the padded mesh, so changing that
mesh changes the particular random realization despite the fixed seed.
This is a throughput comparison, not a same-initial-state trajectory test.
Analytic transform tests (including the highest Chebyshev coefficient),
quadratic dealiasing tests, and physical regression tests check correctness
independently of performance measurements.

Final recorded ESTIMATE comparison (same resolved grid, one thread):

| Case | Before median / minimum (ms) | After median / minimum (ms) | After bytes / allocations |
|---|---:|---:|---:|
| Direct step | 77.497 / 76.907 | 47.763 / 47.049 | 0 / 0 |
| Ten-step propagation, per step | 76.738 / 76.521 | 48.953 / 47.296 | 2,944 / 7 |

The propagation median is 36% lower (1.57x throughput). Final validation:
1,754 interface/analytic tests and 833 physical tests passed. The Grid
minimum-Ny rejection test was removed to follow the intentionally relaxed
constructor contract. The source fingerprints and full records are in
`history.csv`; these results still describe uncommitted source changes.

## Recording subsequent commits

A local `.git/hooks/post-commit` hook invokes `perf/post-commit`, which runs
the benchmark and appends two rows after each commit. The history therefore
becomes modified after committing; include those rows in the following
commit. Git cannot include its own final hash in its own contents.
The hook does not stage or commit anything. It is installed in this checkout
only. For another checkout, install it explicitly and adjust the Julia
executable in `perf/post-commit` if necessary. Changes to the benchmark case
should start a new history file to preserve comparability.

## Detailed profiling report

See [profiling-summary.md](profiling-summary.md) for the phase breakdown,
nested kernel measurements and optimization priorities. Reproduce with
`julia --startup-file=no --threads=1 --project=. perf/profile_detailed.jl`.

## Scalar-field broadcasting correction (2026-09-16)

The custom PhysicalField/SpectralField materializers flattened the broadcast
and indexed `bc[i]` in a manual linear loop. This bypassed Julia's standard
broadcast preparation/copy path and incurred repeated axis/index work.
Instantiating the axes alone improved the microbenchmark but did not remove
most of the overhead. Both overrides have therefore been removed: Julia's
standard materialize!/copyto! now handles iteration and broadcast axes.
The compact broadcast expression in `dot!` is retained unchanged.

`perf/benchmark_convection.jl` preserves the old custom materializer as a
reference and compares it with native-array broadcast and production dot!.
It verifies exact equality on seeded random fields before timing. Eleven
batches of fifty calls on the 35×48×48 padded grid measured:

| Contraction implementation | Median ms/call | Julia bytes |
|---|---:|---:|
| Original custom materializer | 2.264 | 0 |
| Broadcast on parent arrays | 0.356 | 0 |
| Field broadcast using Julia's standard path | 0.362 | 0 |

The corrected field broadcast is about 6.3x faster and matches native-array
performance. An explicit SIMD contraction was also measured, but is not
retained: fixing the shared broadcast implementation improves other field
expressions and keeps the production code shorter.

Before/after complete-step measurements, ESTIMATE, one thread:

| Case | Baseline `0119238` median ms | Corrected median ms | Julia bytes after |
|---|---:|---:|---:|
| Direct step | 49.062 | 40.327 | 0 |
| Propagation, per step | 49.249 | 42.434 | 2,944 |

The propagation median fell by about 14%. Timings vary with machine load;
full source identities and subsequent post-commit measurements are retained
in `history.csv`. Intermediate explicit-loop measurements remain in that
history, identified by their different source hashes. Earlier detailed
profiles describe the pre-correction implementation.

New regression checks cover singleton-axis expansion, self-referential
assignment and rejection of incompatible dimensions for both scalar field
types. Reproduce the kernel comparison with:

```sh
julia --startup-file=no --threads=1 --project=. perf/benchmark_convection.jl
```

Validation of the correction: all 1,760 interface/analytic tests and all
833 physics tests passed.

## FFT copy and scaling passes (2026-09-16)

Forward normalisation, Chebyshev endpoint weights and Nyquist filtering now
share one traversal of the resolved spectrum. On the inverse path, the final
factor `1/2` is applied while copying the transformed resolved modes into the
padded buffer, removing one complete pass over the resolved workspace.

An attempted fusion of the inverse input copy with its endpoint weights was
discarded: isolated timings were 0.021 ms for the manual fused loop against
0.011 ms for `copyto!` followed by the two small endpoint updates. The retained
inverse-output fusion reduced that isolated phase from 0.047 to 0.027 ms per
scalar transform. The retained forward fusion reduced its isolated phase from
0.036 to 0.025 ms per transform.

On the complete ESTIMATE benchmark, the direct-step median changed from
39.885 to 38.735 ms. The ten-step propagation was noisy: its median changed
from 41.575 to 41.978 ms, while the minimum changed from 40.947 to 38.632 ms.
This is a small memory-pass optimization; interpret it from repeated runs and
source hashes in `history.csv`, rather than from one propagation median.

All 1,760 interface/analytic checks and all 833 physics checks passed with
the retained fusions.

## Adaptive Chebyshev transform (2026-09-16)

For `Ny ≤ 35`, the DCT-I is now represented by a cached complex Chebyshev
matrix and applied to all retained Fourier columns with one BLAS `gemm!`.
For larger wall-normal systems, the implementation keeps FFTW's DCT-I. This
threshold is empirical and intentionally conservative: with one thread,
representative isolated measurements were:

| Ny / Fourier columns | FFTW DCT-I ms | Dense BLAS ms |
|---|---:|---:|
| 9 / 544 | 0.040 | 0.023 |
| 17 / 544 | 0.065 | 0.061 |
| 35 / 544 | 0.336 | 0.236 |
| 49 / 1200 | 0.493 | 1.112 |
| 81 / 3136 | 2.040 | 6.657 |
| 129 / 6240 | 6.225 | 35.455 |

The inverse dense matrix incorporates the Chebyshev endpoint input weights;
the existing output scaling remains fused with Fourier padding. The forward
plan owns one additional resolved spectral workspace. `LinearAlgebra` is now
a runtime dependency because production calls BLAS directly.

A first implementation used `reshape` around every three-dimensional buffer:
it reduced the direct step to 31.85 ms but allocated 4,320 bytes in 90 events
per step. `SpectralMatrixView` now exposes the same contiguous storage to
`BLAS.gemm!` without allocating. Calling `BLAS.gemm!` explicitly is required:
generic `mul!` does not recognise the custom view as a strided BLAS matrix and
was substantially slower.

Final ESTIMATE measurements changed from 38.86 to 30.75 ms for the direct
step, and from 41.84 to 32.46 ms per propagated step. The direct step remains
at zero Julia heap allocations; propagation retains 2,944 bytes and seven
allocations per step from the Flows path. All 1,760 interface/analytic checks
and all 833 physics checks passed, including Waleffe cases at both `Ny=35`
(dense path) and `Ny=49` (FFTW fallback).
