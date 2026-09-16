# Source guide

All source files contribute to the `ChannelFlow` module; there are no nested
modules. `ChannelFlow.jl` lists them in dependency order.

| Start here | Responsibility |
|---|---|
| [grids.jl](grids.jl) | Geometry, Lobatto nodes, resolved and padded sizes |
| [fields/](fields/) | Scalar/vector/tensor storage and spatial operators |
| [indexing.jl](indexing.jl) | FFT mode indices and contiguous column iteration |
| [state.jl](state.jl) | Perturbation velocity and pressure, owned by the caller |
| [ffts.jl](ffts.jl) | Public transform plans, allocating transforms, padding and filtering |
| [transforms/chebyshev.jl](transforms/chebyshev.jl) | Dense/FFTW wall-normal transform backends |
| [nonlinear.jl](nonlinear.jl) | Nonlinear forms, their caches, and base-flow addition |
| [helmoltz.jl](helmoltz.jl) | Entry point loading the three Stokes solver implementations |
| [solvers/influence.jl](solvers/influence.jl) | Nonzero-mode pressure/velocity solves, influence matrix and tau correction |
| [solvers/meanmode.jl](solvers/meanmode.jl) | Mean mode, pressure gauge and fixed-gradient/fixed-flux constraints |
| [solvers/fourierstokes.jl](solvers/fourierstokes.jl) | Full Fourier assembly and views of Chebyshev columns |
| [timesteppers/cnrk2.jl](timesteppers/cnrk2.jl) | Three-stage CNRK2 caches and step assembly |
| [problem.jl](problem.jl) | Problem configuration and Couette/Poiseuille constructors |
| [timesteppers/channel.jl](timesteppers/channel.jl) | Flows construction and stepping adapters |
| [initialization.jl](initialization.jl) | Zero/random states and the Stokes projection |
| [postprocessing.jl](postprocessing.jl) | Reusable diagnostic workspace and energy-budget reductions |

## Conventions to read first

- Physical directions are `(x, y, z)`; array storage is `(y, x, z)` in physical
  space and `(n, kx, kz)` in spectral space. The first axis is contiguous.
- `State` stores perturbation velocity. The stationary streamwise base profile
  belongs to `ChannelFlowProblem` and is added when evaluating total velocity.
- `RotatingForm()` is the default; its pressure includes `|u_total|²/2`.
  The multiplier returned by `project!` is an auxiliary projection pressure,
  not a dynamically consistent initial pressure for CNRK2.
- A problem owns factors and workspaces, not a trajectory. Reused caches are
  mutable and must not be shared by concurrent calls.
- Postprocessing reports volume averages. Pass `baseflow` explicitly for total
  quantities. Power includes moving-wall and uniform-pressure work, but not
  additional body-force work. With fixed flux, supply the actual computed
  uniform pressure gradients.

## Following one step

`Flows.flow(problem)` calls the adapter in `timesteppers/channel.jl`, then
`step!` in `timesteppers/cnrk2.jl`. Each stage evaluates `nonlinear.jl`, builds
the Crank--Nicolson right-hand side, and calls `FourierStokesSolver`. The
Fourier solver delegates the mean mode to `MeanModeSolver` and all active
nonzero modes to `InfluenceModeSolver`.

Keep manufactured numerical checks in `test/`, physical validation cases in
`test/physics/`, and reproducible timing/profiling scripts in `perf/`.
