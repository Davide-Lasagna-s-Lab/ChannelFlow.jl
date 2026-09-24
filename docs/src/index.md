# ChannelFlow.jl

**GPU-accelerated Fourier–Chebyshev direct numerical simulation of plane Couette and Poiseuille flow in Julia.**

ChannelFlow evolves three-component incompressible velocity between parallel
no-slip walls, with periodic streamwise and spanwise directions. NVIDIA GPUs
execute the transforms and batched pressure–velocity solves on device. The same
interface also supports CPU execution for small problems, development and verification.

The solver follows the primitive-variable **Kleiser–Schumann formulation**, with
Chebyshev tau discretisation, influence-matrix enforcement of incompressibility,
and the **CNRK2** scheme used by Gibson's `RungeKuttaDNS`. It stores velocity as
a perturbation of a prescribed base flow. See [Numerical formulation](equations.md)
and [References](references.md) for the numerical lineage and terminology.

## Purpose

ChannelFlow.jl aims to lower the barrier to entry to direct numerical simulation
of channel flows at moderate Reynolds numbers. It combines GPU-accelerated DNS
with the interactivity and composability of the Julia ecosystem, making it easy
to explore flows, build diagnostics, and connect simulations to other analysis
and modelling tools.

## Start here

| Task | Guide |
|---|---|
| Install and run your first simulation | [Installation](installation.md), [Quick start](quickstart.md) |
| Run on an NVIDIA GPU | [GPU execution](gpu.md) |
| Choose pressure-gradient or mass-flux driving | [Equations and configuration](equations.md) |
| Understand the numerical method | [Spatial discretisation](spatial.md), [CNRK2](timestepping.md) |
| Understand field layout and transform conventions | [Fields and transforms](fields.md) |
| Initialise velocity and recover pressure | [Initialisation and pressure](pressure.md) |
| Measure, sample and restart a simulation | [Diagnostics](diagnostics.md), [Restart and sampling](restart.md) |
| Run a documented turbulent-channel case | [MKM590 example](mkm590.md) |
| Assess cost and numerical checks | [Benchmarks](benchmarks.md), [Validation](validation.md) |
| Find a function or contribute | [API reference](api.md), [Developer guide](contributing.md) |

## Dependencies

| Package | Role |
|---|---|
| [ChebyshevHelmoltzSolvers.jl](https://github.com/Davide-Lasagna-s-Lab/ChebyshevHelmoltzSolvers.jl) | Batched tau/Helmholtz solves on CPU and CUDA |
| [Flows.jl](https://github.com/Davide-Lasagna-s-Lab/Flows.jl) | Optional integration interface and monitors |
| [FFTW.jl](https://github.com/JuliaMath/FFTW.jl) | CPU Fourier and DCT-I transforms |
| [CUDA.jl](https://cuda.juliagpu.org/stable/) | Optional NVIDIA GPU execution and cuFFT |
| [Adapt.jl](https://github.com/JuliaGPU/Adapt.jl) | Explicit transfer of states and cached solver data |

The source is available on [GitHub](https://github.com/Davide-Lasagna-s-Lab/ChannelFlow.jl)
under the [MIT licence](https://github.com/Davide-Lasagna-s-Lab/ChannelFlow.jl/blob/main/LICENSE).
