# ChannelFlow.jl

**Direct numerical simulation of plane Couette and Poiseuille flow in Julia.**
ChannelFlow.jl solves the incompressible Navier–Stokes equations in primitive
variables using a Fourier–Chebyshev–Fourier discretisation and a semi-implicit
time integrator. Its pressure–velocity coupling follows the influence-matrix
and tau formulation used in John Gibson's
[Channelflow](https://github.com/epfl-ecps/channelflow).

The package provides flow constructors, divergence-free initialisation,
physical/spectral field transforms, integration through Flows.jl, and energy,
dissipation and power-input diagnostics. The current solver runs on the CPU
without MPI.

## Numerical method

### Equations and flow configuration

The domain is periodic in the streamwise and spanwise directions,
$`x\in[0,L_x)`$ and $`z\in[0,L_z)`$, with no-slip walls at $`y=\pm1`$.
For total velocity $`\boldsymbol{v}`$, the equations are

```math
\begin{aligned}
\partial_t\boldsymbol{v}
+ (\boldsymbol{v}\cdot\nabla)\boldsymbol{v}
&= -\nabla p - \boldsymbol{G}
+ \nu\nabla^2\boldsymbol{v} + \boldsymbol{f}, \\
\nabla\cdot\boldsymbol{v} &= 0.
\end{aligned}
```

Here $`p`$ is the periodic pressure, $`\boldsymbol{G}=(dP/dx,0,dP/dz)`$ is the
spatially uniform driving pressure gradient, and $`\boldsymbol{f}`$ is an optional
body force. The code evolves the perturbation
$`\boldsymbol{u}=\boldsymbol{v}-U_b(y)\boldsymbol{e}_x`$, which has homogeneous
wall conditions. Nonlinear terms are evaluated using the **total velocity**, so
advection by the base flow and the effect of its shear are included.

| Constructor | Base profile | Default driving |
|---|---|---|
| `CouetteFlow(grid, nu, dt)` | $`U_b(y)=y`$ | Walls moving at $`\pm1`$; zero pressure gradient |
| `PoiseuilleFlow(grid, nu, dt)` | $`U_b(y)=1-y^2`$ | Stationary walls; $`dP/dx=-2\nu`$ |

With these nondimensionalisations, `nu = 1/Re`, using the channel half-height
and either wall speed (Couette) or laminar centreline speed (Poiseuille).
`ChannelFlowProblem(grid, profile, nu, dt)` accepts a custom stationary
streamwise profile. A prescribed total bulk velocity can replace the prescribed
pressure gradient.

### Spatial discretisation and nonlinear terms

Fields are expanded in Fourier modes in $`x,z`$ and ordinary Chebyshev polynomials
in $`y`$. The wall-normal collocation points are
$`y_j=\cos(\pi j/(N_y-1))`$, ordered from the upper wall to the lower wall.
Fourier derivatives multiply coefficients by their physical wavenumbers;
wall-normal derivatives use Chebyshev coefficient recurrences.

Nonlinear products are formed in physical space and transformed back. The
periodic directions use **3/2 padding**, followed by truncation to the retained
Fourier modes; the resolved Nyquist planes are set to zero. The wall-normal
resolution is unchanged during this operation: the current implementation does
not apply Chebyshev dealiasing.

The default `RotatingForm()` evaluates
$`\boldsymbol{v}\times(\nabla\times\boldsymbol{v})`$ and absorbs the kinetic-energy
gradient into the modified pressure
$`q=p+\tfrac12|\boldsymbol{v}|^2`$.
`ChannelFlow.ConvectiveForm()` and `ChannelFlow.DivergenceForm()` instead
use the ordinary pressure. Fourier transforms use FFTW; Chebyshev transforms
use either FFTW's DCT-I (`chebbackend=:fftw`, the default) or dense matrix
multiplication (`chebbackend=:gemm`).

### Time integration

`CNRK2` is Channelflow's **three-stage, second-order** combination of
Crank–Nicolson and Runge–Kutta. Viscosity is treated implicitly, while nonlinear
terms and additional forcing are evaluated explicitly. Each stage assembles a
momentum right-hand side from the current fields and explicit-stage history,
then solves a coupled velocity–pressure problem.

For fixed grid, viscosity and time step, the three stage-specific Helmholtz
factorisations and influence responses are cached and reused. Integration uses
a fixed nominal `dt`; Flows.jl may shorten the final step to reach the requested
endpoint, in which case factors are constructed for that step.

A `State` retains both perturbation velocity and algebraic stage pressure:
CNRK2 uses the preceding stage's pressure gradient. Pressure has no independent
evolution equation, but must be kept when copying or restarting the discrete
state. Access the fields with `velocity(state)` and `stagepressure(state)`.

### Helmholtz solves and the influence matrix

For a nonzero Fourier pair, let
$`\alpha=2\pi k_x/L_x`$, $`\beta=2\pi k_z/L_z`$,
$`\kappa^2=\alpha^2+\beta^2`$, and $`D=d/dy`$.
Each implicit stage solves a Chebyshev tau discretisation of

```math
\begin{aligned}
\left[\lambda+\nu\kappa^2-\nu D^2\right]\widehat{\boldsymbol{u}}
+\nabla_k\widehat{q} &= \widehat{\boldsymbol{r}}, \\
\nabla_k\cdot\widehat{\boldsymbol{u}} &= 0,
\qquad \nabla_k=(i\alpha,D,i\beta).
\end{aligned}
```

with zero perturbation velocity at both walls. The positive shift $`\lambda`$
depends on the stage and `dt`; $`q`$ denotes the pressure appropriate to the
chosen nonlinear form.

The solve first obtains particular pressure and wall-normal velocity solutions.
Two precomputed homogeneous pressure responses then supply a **2×2 influence
matrix**: their amplitudes are chosen so the wall-normal velocity also satisfies
$`D\widehat{u}_y(\pm1)=0`$, as required by continuity and tangential no slip.
Gibson's tau correction accounts for the highest-order momentum residuals so
that the discrete pressure and velocity equations remain consistent.
The tangential velocity components follow from scalar Helmholtz solves with
the corrected pressure. The present tau correction requires **odd `Ny ≥ 3`**.

The **zero Fourier mode** is solved separately. Its wall-normal velocity is
zero; its two tangential components satisfy one-dimensional Helmholtz equations.
The solver either applies a specified pressure gradient or determines the
gradient needed to impose the total bulk velocities. The mean pressure is
reconstructed from normal momentum, with zero wall-normal average fixing its
arbitrary additive constant.

## Dependencies

| Package | Role in ChannelFlow.jl |
|---|---|
| [ChebyshevHelmoltzSolvers.jl](https://github.com/Davide-Lasagna-s-Lab/ChebyshevHelmoltzSolvers.jl) | Structured Chebyshev tau Helmholtz solves, coefficient differentiation and endpoint evaluation; also supports the Neumann solves used for initial pressure reconstruction |
| [Flows.jl](https://github.com/Davide-Lasagna-s-Lab/Flows.jl) | Time-integration interface, monitors and trajectory storage |
| [FFTW.jl](https://github.com/JuliaMath/FFTW.jl) | Fourier transforms and the FFTW backend for Chebyshev transforms |

Julia's standard libraries provide
[LinearAlgebra](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/),
[Random](https://docs.julialang.org/en/v1/stdlib/Random/), and
[Serialization](https://docs.julialang.org/en/v1/stdlib/Serialization/).
The upstream C++ Channelflow project is the numerical reference, not a runtime
dependency.

## Installation

Requires **Julia 1.11 or later**. Clone the repository and instantiate its
environment; `Project.toml` supplies the GitHub sources for Flows.jl and
ChebyshevHelmoltzSolvers.jl.

```sh
git clone https://github.com/Davide-Lasagna-s-Lab/ChannelFlow.jl.git
cd ChannelFlow.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Example: plane Couette flow

Start Julia with `julia --project=.` from the repository directory:

```julia
using ChannelFlow, Flows, Random

Random.seed!(42)
Nx, Ny, Nz = 32, 33, 32
grid = Grid(Nx, Ny, Nz, 2π / 1.14, 2π / 2.5)
Re, dt = 400.0, 0.025
problem = CouetteFlow(grid, 1 / Re, dt; chebbackend=:fftw)

# Project a random perturbation to satisfy continuity and no slip,
# then reconstruct a consistent initial pressure.
state = random_state(problem, 0.1)
integrate = Flows.flow(problem)
integrate(state, (0.0, 1.0))

U = velocity(state)  # spectral perturbation velocity
u = IFFT(U)          # physical perturbation velocity
```

The amplitude passed to `random_state` scales the samples **before projection**;
it does not prescribe the final RMS amplitude. Constructor dimensions are
ordered `(Nx, Ny, Nz)`, while field arrays store the wall-normal direction first.
The base profile belongs to `problem` and is not included in `velocity(state)`
or in its inverse transform.

For pressure-driven flow, replace the constructor with
`PoiseuilleFlow(grid, 1/Re, dt)`. For fixed total bulk velocity, use
`PoiseuilleFlow(grid, 1/Re, dt; bulkvelocity=(2/3, 0))`.

See the [source guide](src/README.md) for the code layout and the function
docstrings for field operations, initialisation and diagnostics.
Run the interface and manufactured-solution tests with
`julia --project=. -e 'using Pkg; Pkg.test()'`.
Additional physical validation cases are in [`test/physics`](test/physics).
