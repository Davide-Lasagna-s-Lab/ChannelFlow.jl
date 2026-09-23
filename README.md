# ChannelFlow.jl

A Julia solver for direct numerical simulation of incompressible plane Couette
and Poiseuille flow. The periodic directions are streamwise (`x`) and spanwise
(`z`); the walls are at `y = ±1`.

## Numerical method

Velocity and pressure use Fourier expansions in `x,z` and Chebyshev polynomials
in `y`. Nonlinear terms are evaluated pseudospectrally with 3/2 padding in the
periodic directions and zeroed Nyquist modes. The rotational form is the default;
convective and divergence forms are also available. Wall-normal transforms can
use FFTW (default) or dense matrix multiplication (`chebbackend=:gemm`).

Time integration uses Gibson's three-stage, second-order CNRK2 scheme, with
Crank–Nicolson treatment of viscosity. Each stage solves the coupled velocity–
pressure system using Chebyshev Helmholtz solves, an influence matrix and tau
correction to enforce incompressibility and no slip. The mean Fourier mode is
handled separately, with either prescribed pressure gradient or bulk velocity.
The formulation follows [Channelflow](https://github.com/epfl-ecps/channelflow).

The evolved velocity is a perturbation about the selected laminar profile.
`State` also retains the algebraic stage pressure needed by CNRK2; with the
rotational form this is modified pressure. The solver is CPU-based, without MPI.

## Dependencies and installation

Requires Julia 1.11 or later. The main dependencies are
[Flows.jl](https://github.com/Davide-Lasagna-s-Lab/Flows.jl) for integration and
monitoring, [ChebyshevHelmoltzSolvers.jl](https://github.com/Davide-Lasagna-s-Lab/ChebyshevHelmoltzSolvers.jl)
for wall-normal solves, and [FFTW.jl](https://github.com/JuliaMath/FFTW.jl)
for transforms.

Clone this repository and initialise its environment:

```sh
git clone https://github.com/Davide-Lasagna-s-Lab/ChannelFlow.jl.git
cd ChannelFlow.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Example

Run Julia with `--project=.` from the repository directory:

```julia
using ChannelFlow, Flows, Random

Random.seed!(42)
Nx, Ny, Nz = 32, 33, 32  # Ny must be odd for the tau correction.
grid = Grid(Nx, Ny, Nz, 2π / 1.14, 2π / 2.5)
Re, dt = 400.0, 0.025
problem = CouetteFlow(grid, 1 / Re, dt; chebbackend=:fftw)

# Random perturbation, projected to satisfy continuity and wall conditions,
# with a consistent initial pressure. Amplitude is set before projection.
state = random_state(problem, 0.1)
integrate = Flows.flow(problem)
integrate(state, (0.0, 1.0))

U = velocity(state)  # spectral perturbation velocity
u = IFFT(U)          # physical perturbation velocity
```

Use `PoiseuilleFlow(grid, 1/Re, dt)` for pressure-driven flow; its default
pressure gradient sustains the unit-centreline laminar profile. Supply
`bulkvelocity=(2/3, 0)` for fixed bulk velocity instead. Constructor dimensions
are ordered `(Nx, Ny, Nz)`; field arrays store the wall-normal direction first.

See the [source guide](src/README.md) for the code layout. Run the numerical
and interface tests with `julia --project=. -e 'using Pkg; Pkg.test()'`;
additional physical validation cases are in [`test/physics`](test/physics).
