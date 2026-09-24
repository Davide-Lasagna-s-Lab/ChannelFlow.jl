# ChannelFlow.jl

[![CPU validation](https://github.com/Davide-Lasagna-s-Lab/ChannelFlow.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/Davide-Lasagna-s-Lab/ChannelFlow.jl/actions/workflows/ci.yml)
[![Documentation](https://github.com/Davide-Lasagna-s-Lab/ChannelFlow.jl/actions/workflows/docs.yml/badge.svg)](https://Davide-Lasagna-s-Lab.github.io/ChannelFlow.jl/)
[![Julia](https://img.shields.io/badge/Julia-1.11%2B-9558B2)](https://julialang.org/)
[![CUDA](https://img.shields.io/badge/GPU-NVIDIA%20CUDA-76B900)](https://cuda.juliagpu.org/stable/)
[![MIT licence](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**GPU-accelerated Fourier–Chebyshev DNS of plane Couette and Poiseuille flow in Julia.**

Run on an NVIDIA GPU with batched pressure–velocity solves, cuFFT transforms
and reusable device workspaces. CPU execution is also supported through the
same interface. The primitive-variable method follows the Kleiser–Schumann
influence-matrix formulation with Chebyshev tau correction and Gibson's
three-stage, second-order **CNRK2** scheme.

**[Documentation](https://Davide-Lasagna-s-Lab.github.io/ChannelFlow.jl/)** ·
[Numerical method](https://Davide-Lasagna-s-Lab.github.io/ChannelFlow.jl/spatial/) ·
[Benchmarks](https://Davide-Lasagna-s-Lab.github.io/ChannelFlow.jl/benchmarks/) ·
[API](https://Davide-Lasagna-s-Lab.github.io/ChannelFlow.jl/api/)

## Contents

- [Install](#install)
- [GPU example](#gpu-example)
- [Dependencies and references](#dependencies-and-references)

## Install

With Julia 1.11 or later:

```sh
git clone https://github.com/Davide-Lasagna-s-Lab/ChannelFlow.jl.git
cd ChannelFlow.jl
julia --project=test/cuda -e 'using Pkg; Pkg.instantiate()'
julia --project=test/cuda
```

For CPU-only installation and application environments, see the
[installation guide](https://Davide-Lasagna-s-Lab.github.io/ChannelFlow.jl/installation/).

## GPU example

```julia
using ChannelFlow, Flows, FFTW, CUDA, Adapt, Random

CUDA.allowscalar(false)
Random.seed!(42)
grid = Grid(32, 33, 32, 2π/1.14, 2π/2.5) # Nx, Ny, Nz, Lx, Lz
problem = CouetteFlow(grid, 1/400, 0.01; fftwflags=FFTW.ESTIMATE)
state = random_state(problem, 0.1)

problem = adapt(CuArray, problem)
state = adapt(CuArray, state)
integrate = Flows.flow(problem)
integrate(state, (0.0, 1.0))
CUDA.synchronize()
```

Velocity is stored as a perturbation of the base flow. The manual explains
pressure conventions, driving constraints, diagnostics and restart. The
[MKM590 example](examples/mkm590/README.md) includes GPU simulation,
mean-flow/Reynolds-stress collection and comparison with published data.

## Dependencies and references

[ChebyshevHelmoltzSolvers.jl](https://github.com/Davide-Lasagna-s-Lab/ChebyshevHelmoltzSolvers.jl)
provides the batched Helmholtz backend and documents its tau discretisation.
[Flows.jl](https://github.com/Davide-Lasagna-s-Lab/Flows.jl) provides integration
and monitors; [FFTW.jl](https://github.com/JuliaMath/FFTW.jl),
[CUDA.jl](https://cuda.juliagpu.org/stable/) and
[Adapt.jl](https://github.com/JuliaGPU/Adapt.jl) provide transforms and device support.

[Gibson's Channelflow](https://github.com/epfl-ecps/channelflow) is the numerical
reference, not a runtime dependency. See the manual's
[references](https://Davide-Lasagna-s-Lab.github.io/ChannelFlow.jl/references/)
and [validation](https://Davide-Lasagna-s-Lab.github.io/ChannelFlow.jl/validation/).
This package is a separate Julia implementation, not a complete port of the C++ software.

[MIT licence](LICENSE) · Davide Lasagna
