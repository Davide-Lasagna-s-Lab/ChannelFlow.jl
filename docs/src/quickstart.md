# Quick start

Choose either the CPU or GPU example below. Each is self-contained and runs
100 fixed time steps of plane Couette flow from a random perturbation.
Both use the manual stepping interface; Flows.jl is optional and is introduced
at the end of this page. See [Installation](installation.md) for package setup.

## CPU quick start

```@example quickstart
using ChannelFlow, FFTW, Random

# Resolution before padding and periodic domain lengths.
Nx, Ny, Nz = 32, 33, 32
Lx, Lz = 2π/1.14, 2π/2.5

# Re is based on wall speed and channel half-height.
Re = 400.0
dt = 0.01
nsteps = 100
amplitude = 0.1

# Construct the solver and a divergence-free initial condition.
Random.seed!(42)
grid = Grid(Nx, Ny, Nz, Lx, Lz)
problem = CouetteFlow(grid, 1/Re, dt; fftwflags=FFTW.ESTIMATE)
state = random_state(problem, amplitude)
initial_energy = kinetic_energy(velocity(state))

# Advance velocity and stage pressure in place.
t = let t = 0.0
    for _ = 1:nsteps
        t = step!(problem.scheme, problem.nlterm,
                  velocity(state), stagepressure(state), t;
                  forcing=problem.forcing, problem.constraint...)
    end
    t
end

# Inspect the final perturbation and reconstruct it in physical space.
final_energy = kinetic_energy(velocity(state))
u = IFFT(velocity(state))
@show t initial_energy final_energy
nothing # hide
```

`Nx`, `Ny` and `Nz` are the numbers of grid points **before padding**, in the
streamwise, wall-normal and spanwise directions. Here the resolved grid is
32×33×32. Nonlinear products use 3/2 padding only in x and z, giving a physical
work grid of 48×33×48; `Ny=33` remains unchanged. These counts are physical grid
sizes, not the dimensions of the stored real-FFT spectrum. Arrays themselves
use `(x,z,y)` storage order; see [Fields and transforms](fields.md).

`random_state` projects random velocity onto the divergence-free, no-slip
space and reconstructs pressure. Its amplitude scales the random samples
**before projection**, not the final kinetic energy. `roll_state(problem, 0.1)`
creates a smooth streamwise-independent roll instead.

`step!` updates velocity and stage pressure in place and returns the new time.
Time is supplied explicitly, while `problem.scheme.dt` sets the step duration.
This loop never shortens a final step and is useful for uniform sampling and
restarting in blocks with a prescribed number of steps. The energy and physical
velocity above refer to the **perturbation**; total velocity also includes the
base profile `Ub(y)=y`.

## GPU quick start

This example requires an NVIDIA GPU supported by CUDA.jl. Install the optional
GPU packages in your application environment once:

```julia
using Pkg
Pkg.add(["CUDA", "Adapt"])
```

The complete script below constructs the problem and initial state on the CPU,
transfers them once, advances on the GPU, and downloads the final state for
analysis. It does not depend on running the CPU example first.

```julia
using ChannelFlow, FFTW, Random, CUDA, Adapt

CUDA.functional() || error("A working CUDA installation and NVIDIA GPU are required")
CUDA.allowscalar(false)

# Resolution before padding; y is not padded.
Nx, Ny, Nz = 32, 33, 32
Lx, Lz = 2π/1.14, 2π/2.5
Re = 400.0
dt = 0.01
nsteps = 100
amplitude = 0.1

# Initialise on the CPU, including the initial pressure reconstruction.
Random.seed!(42)
grid = Grid(Nx, Ny, Nz, Lx, Lz)
problem = CouetteFlow(grid, 1/Re, dt; fftwflags=FFTW.ESTIMATE)
state = random_state(problem, amplitude)
initial_energy = kinetic_energy(velocity(state))

# Transfer the state and solver caches and create the device transform plans.
# Reuse these objects throughout the simulation.
gpu_problem = adapt(CuArray, problem)
gpu_state = adapt(CuArray, state)

# All time steps operate on device-resident fields.
t = let t = 0.0
    for _ = 1:nsteps
        t = step!(gpu_problem.scheme, gpu_problem.nlterm,
                  velocity(gpu_state), stagepressure(gpu_state), t;
                  forcing=gpu_problem.forcing, gpu_problem.constraint...)
    end
    t
end
CUDA.synchronize()

# Download once after the run, then use the usual CPU diagnostics and FFTs.
final_state = adapt(Array, gpu_state)
final_energy = kinetic_energy(velocity(final_state))
u = IFFT(velocity(final_state))
@show t initial_energy final_energy
```

Adaptation preserves double precision and builds the device transform plans.
The Chebyshev transform uses cuFFT on the GPU;
periodic transforms also use cuFFT. The integration loop performs no full-field
host transfers. `CUDA.synchronize()` waits for queued GPU work to finish.

The CPU `state` remains the initial condition: subsequent steps must use
`gpu_state`. To continue this simulation, reuse `gpu_problem`, `gpu_state` and
`t` without reconstructing or transferring them again. This small grid makes
setup easy; GPU speedups depend on problem size, as shown in
[Benchmarks](benchmarks.md). See [GPU execution](gpu.md) for transform details,
memory use and performance considerations.

## Optional integration and monitors with Flows.jl

[Flows.jl](https://github.com/Davide-Lasagna-s-Lab/Flows.jl) is an optional
dependency. Install it in your application environment if you want its
integration and monitoring interface:

```julia
using Pkg
Pkg.add(url="https://github.com/Davide-Lasagna-s-Lab/Flows.jl.git")
```

Loading both packages activates the integration extension automatically.
The following continues the CPU example and samples perturbation energy:

```@example quickstart
import Flows
integrate = Flows.flow(problem)
monitor = Flows.Monitor(state, (t, s) -> kinetic_energy(velocity(s)); oneevery=10)
integrate(state, (t, t + 1.0), monitor)
sample_times = Flows.times(monitor)
energies = Flows.samples(monitor)
t += 1.0
@assert all(isfinite, energies) # hide
nothing # hide
```

For the GPU example, construct `Flows.flow(gpu_problem)` and pass `gpu_state`
to both the monitor constructor and the integration call.

Flows can shorten the last step to reach the requested endpoint. The monitor
records perturbation energy without storing the full trajectory; this quantity
is not total-flow energy. Diagnostics allocate and may synchronize on a GPU.
See [Restart and sampling](restart.md) for longer runs and saved trajectories.
