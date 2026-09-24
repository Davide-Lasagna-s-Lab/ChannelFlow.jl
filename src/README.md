# Source guide

Read [`problem.jl`](problem.jl) for configuration and
[`timesteppers/cnrk2.jl`](timesteppers/cnrk2.jl) for the equations advanced each
stage. The root [README](../README.md) defines the numerical conventions.

| Files | Responsibility |
|---|---|
| `grids.jl` | Domain, Lobatto points, resolved/padded storage sizes |
| `fields/` | Physical/spectral/vector/tensor containers, operators, versioned I/O |
| `state.jl` | Caller-owned perturbation velocity and stage pressure |
| `ffts.jl`, `transforms/chebyshev.jl` | Transform plans, padding, normalisation, FFTW/GEMM backends |
| `nonlinear.jl` | Three nonlinear forms, total-velocity assembly and reusable caches |
| `solvers/batchedinfluence.jl` | Batched pressure/velocity solves, wall influence and tau correction |
| `solvers/meanmode.jl` | Mean pressure gauge and gradient/flux constraints |
| `solvers/stokes.jl` | Full-field zero-copy matrix views and modal assembly |
| `solvers/influence.jl` | Scalar nonzero-mode reference used by analytic comparison tests |
| `timesteppers/` | Three-stage CNRK2 and Flows adapter, including shortened terminal steps |
| `initialization.jl`, `pressure.jl` | Initial states, projection and instantaneous pressure |
| `postprocessing.jl` | Physical-volume inner product, energy, dissipation and power input |
| `adapt.jl` | Transfers of fields and cached numerical data |
| `../ext/` | Optional CUDA plans, kernels, mean constraints and pressure reconstruction |

## Following a timestep

The Flows adapter passes a caller-owned `State` to CNRK2. Each stage computes
nonlinearity on padded physical arrays, assembles the momentum RHS, and solves
Stokes. A spectral field is `(kx,kz,n)`: `spectralmatrix` exposes it as
`(system,coefficient)` without copying. The Helmholtz backend processes those
rows together. Homogeneous influence and tau responses are cached separately
from mutable RHS buffers. The zero mode is overwritten with its own equations.

GPU execution uses the same high-level stage and nonlinear code. CUDA methods
replace storage-specific operations; the dependency supplies its own batched
Helmholtz kernels. Grids remain host metadata. Standard fixed-gradient stages
do not copy field data back to the host.

## Contracts worth preserving

- Components remain `(u,v,w)` even though physical array axes are `(x,z,y)`.
- Velocity in a `State` is a perturbation. Nonlinearity adds the base profile.
- Nonlinearity overwrites its output; forcing adds to that output.
- Rotational form uses modified pressure. Restart pressure is stage history;
  reconstructed instantaneous pressure is an initialization/diagnostic operation.
- A problem owns mutable caches. Do not share it between simultaneous calls.
- Spectral coefficients use one-based arrays and ordinary Chebyshev series.
- Save CPU fields with layout metadata; convert old layouts only when loading.

Tests in `test/` use manufactured functions and scalar references. Independent
physical validations live in `test/physics/`; device tests live in `test/cuda/`.
Benchmarks measure warmed complete steps, with device synchronization included.
