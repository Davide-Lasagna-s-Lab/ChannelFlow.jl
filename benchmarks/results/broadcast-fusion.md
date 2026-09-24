# GPU transform fusion

Measured on 2026-09-24 on an NVIDIA A100 80 GB PCIe, IRIDIS X, Julia
1.12.4 and CUDA 6.4.0. The baseline uses the existing transform implementation;
the candidate adds the uncommitted `ext/transforms.jl` changes to `76cc7d0`.
The remote baseline differs from the local transform file only in field comments.

## Changes

- Prepare the complex even extension for the Chebyshev transform in one kernel,
  including inverse-transform endpoint doubling. Previously this used two array
  assignments and, for inverse transforms, two additional endpoint broadcasts.
- Combine forward normalization, Chebyshev endpoint weights and Fourier Nyquist
  filtering in one kernel, replacing five broadcasts for even Fourier sizes.

The transform convention, coefficients, cuFFT plans and numerical method are
unchanged. CPU methods are untouched.

## Complete CNRK2 step

`benchmarks/step.jl` measures a complete rotational-form Couette step, with five
warm-up steps and 100 timed samples per size. Each sample restores the same state
outside the timing interval and synchronizes the GPU before and after execution.
Planning, compilation and initial host/device transfers are excluded. Times below
are sample minima, not averages; raw files also contain averages and host allocations.
Both variants ran sequentially on the same allocated GPU (job 1642980).

| Nx × Ny × Nz | Before (ms) | After (ms) | Time reduction | Speedup |
|---|---:|---:|---:|---:|
| 32 × 33 × 32 | 3.917 | 2.831 | 27.7% | 1.38× |
| 64 × 65 × 64 | 6.274 | 5.686 | 9.4% | 1.10× |
| 128 × 129 × 128 | 20.466 | 19.907 | 2.7% | 1.03× |

The improvement is largest for small problems, consistent with reduced launch
and dispatch overhead. These are full-step measurements, not isolated broadcast
kernel timings; they do not establish a speedup for the larger production grid.
The running MKM DNS was not modified or restarted.

Raw data: [before](a100-broadcast-before.csv), [after](a100-broadcast-after.csv).
To measure the current implementation on a GPU allocation:

```sh
CHANNEL_SAMPLES=100 CHANNEL_SIZES=32,64,128 \
CHANNEL_SOURCE_COMMIT=working-tree \
julia --startup-file=no --project=test/cuda benchmarks/step.jl cuda timings.csv
```

## Numerical checks

All 40 additional transform checks and all 118 existing GPU checks passed.
The new checks compare forward and inverse transforms with CPU FFTW for odd/even
wall-normal and Fourier sizes, verify source preservation, and test normalization
with nonzero endpoint and Nyquist coefficients. Existing tests include five-step
CPU/GPU comparisons for all three nonlinear forms, pressure and constrained solves,
viscous decay, diagnostics and a larger-grid comparison.

See [test output](a100-broadcast-tests.txt). Tests ran with scalar CUDA indexing
forbidden. The validation process loaded the candidate methods into the CUDA
extension explicitly, leaving the remote production checkout unchanged.
