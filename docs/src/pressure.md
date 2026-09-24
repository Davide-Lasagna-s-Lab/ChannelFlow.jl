# Initialisation and pressure

`project!(U, problem)` overwrites velocity with a divergence-free, no-slip field
and returns U. It solves a stationary Stokes problem with source ``-\Delta U``;
this is a gradient-seminorm projection, not an orthogonal L² projection.
Its multiplier is discarded. Construct a complete state with
`State(U, pressure(U, problem))` afterwards.



## Construct a consistent state

```julia
U = velocity(zero_state(grid))
# Fill U with a chosen perturbation, then enforce wall conditions and continuity.
project!(U, problem)
state = State(U, pressure(U, problem, 0.0))
```

`random_state(problem, amplitude)` and `roll_state(problem, amplitude)` provide
complete states. The roll is streamwise independent; by itself it cannot excite
a streamwise-dependent instability. Random amplitude is a pre-projection scale,
not a prescribed perturbation energy. Neither routine guarantees turbulence.

For a fixed-flux problem, initialization must also meet the requested total bulk
velocity. Projection accepts the problem's constraint; pressure reconstruction
assumes velocity is already divergence free. Arbitrary saved velocity arrays
should be checked or projected before use.

## Pressure Poisson problem

For a divergence-free total velocity ``v`` and signed nonlinear/body-force
acceleration ``N+f``, taking the divergence of momentum gives

```math
\Delta q=\nabla\cdot(N+f).
```

The viscous term vanishes in this interior divergence, but normal momentum at
an impermeable stationary wall still gives

```math
\partial_y q\big|_{\pm1}=(N+f)_y\big|_{\pm1}+\nu\partial_y^2 v_y\big|_{\pm1}.
```

Here the derivative is with respect to increasing ``y`` at both walls, not the
outward normal derivative. Omitting the viscous wall term changes the problem.
The pressure routine uses the Helmholtz dependency for each Fourier Neumann
problem. The horizontally uniform mode needs a compatibility condition and a
pressure gauge; additive spatial constants have no effect on velocity.

In rotational form ``q=p+|v|^2/2``. Recover physical pressure by subtracting total
kinetic energy per unit mass in physical space, with a consistent additive gauge.
The nonperiodic uniform driving gradient is stored separately from periodic ``q``.
