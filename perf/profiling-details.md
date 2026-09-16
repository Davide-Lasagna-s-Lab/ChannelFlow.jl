# Detailed DNS CPU profile

Measured 2026-09-16T08:02:26.480 UTC. Julia 1.12.6; CPU apple-m1; Julia threads 1; BLAS/FFTW threads 1.

ChannelFlow: `(commit = "be56ce25852e322b6e2467aa0e1de24841a55d15", dirty = true, sha256 = "4c25ef7b1ddb791b46547b3f9c2459abd8a65d12936fd991088642af9ca74040")`

Chebyshev backend: `(commit = "90fd15668dc78601c3d6e6f820fa1e69e59857a2", dirty = true, sha256 = "b1c045e8bf7a384c471dc843a4f5ff27230c24b8dca5144e0d1678c323236ff3")`

Couette Re=400; Ny=35, Nx=Nz=32; padded 35×48×48; dt=0.025; convective form; no monitor. Initialization/planning/compilation excluded. 21 warmed samples per timing; fixed seeded input restored before each whole step. Allocations refer only to Julia's heap, not native FFTW malloc. Instrumentation is a benchmark-only copy of the constant-pressure-gradient convective step, verified against production before timing. It does not cover forcing, constant flux or other nonlinear forms. Raw CPU profiles sample 100 uninstrumented steps including native frames. Kernel timings use separate caches/intervals and cannot be summed as an exact whole-step model.

## ESTIMATE

Production step: median **47.117 ms**, min 46.052, max 88.445; Julia heap bytes: 0.

Instrumented step: mean 64.75 ms, median 60.519 ms. Agreement with production: velocity max error 0.0, pressure max error 0.0.

| Disjoint phase, summed over three stages | Mean ms/step | Share of instrumented wall time |
|---|---:|---:|
| Copy velocity + base flow | 0.635 | 0.98% |
| Spectral velocity gradient | 1.03 | 1.59% |
| Inverse transforms: velocity (3) | 8.719 | 13.47% |
| Inverse transforms: gradient (9) | 25.859 | 39.94% |
| Physical convection products | 8.851 | 13.67% |
| Forward transforms (3) | 9.096 | 14.05% |
| Store nonlinear RHS | 0.58 | 0.9% |
| RK history | 0.799 | 1.23% |
| CN RHS: Laplacians, pressure gradients, broadcasts | 2.367 | 3.66% |
| Stokes solve | 6.792 | 10.49% |
| Timing/control overhead outside intervals | 0.022 | 0.03% |

Nested kernels (isolated, warmed; **not additive** with phase table):

| Kernel | Median ms/call | Calls/step | Julia bytes/call |
|---|---:|---:|---:|
| Inverse scalar, complete | 0.7966 | 36 | 0 |
| Forward scalar, complete | 0.7454 | 9 | 0 |
| Inverse DCT-I, resolved | 0.7709 | 36 | 16 |
| Inverse Fourier brfft | 0.2641 | 36 | 16 |
| Forward Fourier rfft | 0.3172 | 9 | 16 |
| Forward DCT-I, resolved | 0.4813 | 9 | 16 |
| Scalar Laplacian | 0.0541 | 9 | 0 |
| Streamwise derivative | 0.0306 | 12 | 0 |
| Wall-normal derivative | 0.0352 | 12 | 0 |
| Spanwise derivative | 0.0222 | 12 | 0 |

## MEASURE

Production step: median **50.943 ms**, min 44.824, max 99.65; Julia heap bytes: 0.

Instrumented step: mean 45.61 ms, median 45.336 ms. Agreement with production: velocity max error 0.0, pressure max error 0.0.

| Disjoint phase, summed over three stages | Mean ms/step | Share of instrumented wall time |
|---|---:|---:|
| Copy velocity + base flow | 0.388 | 0.85% |
| Spectral velocity gradient | 0.701 | 1.54% |
| Inverse transforms: velocity (3) | 6.101 | 13.38% |
| Inverse transforms: gradient (9) | 18.19 | 39.88% |
| Physical convection products | 6.92 | 15.17% |
| Forward transforms (3) | 5.698 | 12.49% |
| Store nonlinear RHS | 0.383 | 0.84% |
| RK history | 0.529 | 1.16% |
| CN RHS: Laplacians, pressure gradients, broadcasts | 1.698 | 3.72% |
| Stokes solve | 4.998 | 10.96% |
| Timing/control overhead outside intervals | 0.005 | 0.01% |

Nested kernels (isolated, warmed; **not additive** with phase table):

| Kernel | Median ms/call | Calls/step | Julia bytes/call |
|---|---:|---:|---:|
| Inverse scalar, complete | 0.6486 | 36 | 0 |
| Forward scalar, complete | 0.602 | 9 | 0 |
| Inverse DCT-I, resolved | 0.3679 | 36 | 0 |
| Inverse Fourier brfft | 0.1985 | 36 | 0 |
| Forward Fourier rfft | 0.1701 | 9 | 0 |
| Forward DCT-I, resolved | 0.3597 | 9 | 0 |
| Scalar Laplacian | 0.0366 | 9 | 0 |
| Streamwise derivative | 0.0221 | 12 | 0 |
| Wall-normal derivative | 0.0306 | 12 | 0 |
| Spanwise derivative | 0.0221 | 12 | 0 |
