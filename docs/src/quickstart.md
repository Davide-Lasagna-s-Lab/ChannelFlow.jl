# Quick start

Start with a small CPU problem, then transfer it to the GPU as shown below:

```@example quickstart
using ChannelFlow, Flows, FFTW, Random

Random.seed!(42)
grid = Grid(32, 33, 32, 2π/1.14, 2π/2.5) # Nx, Ny, Nz, Lx, Lz
problem = CouetteFlow(grid, 1/400, 0.01; fftwflags=FFTW.ESTIMATE)
state = random_state(problem, 0.1)

integrate = Flows.flow(problem)
integrate(state, (0.0, 1.0))

U = velocity(state) # spectral PERTURBATION velocity
u = IFFT(U)         # padded physical perturbation velocity
nothing # hide
```

`random_state` projects the random velocity onto the divergence-free, no-slip
space and reconstructs pressure. Its amplitude scales the random samples
**before projection**, not the final kinetic energy. `roll_state(problem, 0.1)`
creates a smooth streamwise-independent roll instead. Neither initializer
guarantees transition to turbulence.

For an explicit number of steps, with no shortened final step:

```@example quickstart
state = random_state(problem, 0.1)
t = let t = 0.0
  for _ = 1:100
    t, gradients = step!(problem.scheme, problem.nlterm,
                         velocity(state), stagepressure(state), t;
                         forcing=problem.forcing, problem.constraint...)
  end
  t
end
```

A monitor can record perturbation energy without storing the full trajectory:

```@example quickstart
monitor = Flows.Monitor(state, (t, s) -> kinetic_energy(velocity(s)); oneevery=10)
integrate(state, (t, t + 1.0), monitor)
sample_times = Flows.times(monitor)
energies = Flows.samples(monitor)
t += 1.0
@assert all(isfinite, energies) # hide
nothing # hide
```

This observable uses perturbation velocity, so it is not total-flow energy.
Diagnostics allocate and, on the GPU, may synchronize; benchmark the solver
without a monitor when measuring integration cost alone.

The returned `gradients` from `step!` are `(dPdx, dPdz)` from the final implicit stage.
Use this loop when comparing devices or appending uniformly sampled records.

## Move the simulation to the GPU

```julia
using CUDA, Adapt
CUDA.allowscalar(false)
gpu_problem = adapt(CuArray, problem)
gpu_state = adapt(CuArray, state)
gpu_integrate = Flows.flow(gpu_problem)
gpu_integrate(gpu_state, (t, t + 1.0))
CUDA.synchronize()
```

The CPU state and GPU state are distinct. Keep evolving `gpu_state`; downloading
it is an explicit operation. Continue with [GPU execution](gpu.md) for memory,
backend and timing details, and [Restart and sampling](restart.md) for long runs.
