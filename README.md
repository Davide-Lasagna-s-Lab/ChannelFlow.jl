# ChannelFlow.jl

**Fourier–Chebyshev direct numerical simulation of plane Couette and Poiseuille flow, in Julia.**

ChannelFlow evolves incompressible velocity in a periodic channel with no-slip
walls. It uses a Chebyshev tau discretisation, an influence matrix for
pressure–velocity coupling, and Gibson's three-stage, second-order CNRK2
scheme. Independent Fourier problems are solved together by the batched
[ChebyshevHelmoltzSolvers.jl](https://github.com/Davide-Lasagna-s-Lab/ChebyshevHelmoltzSolvers.jl)
backend. CPU arrays and NVIDIA GPU arrays use the same equations and spectral
normalisation.

The numerical reference is [Channelflow](https://github.com/epfl-ecps/channelflow).
This package provides its own Julia DNS interface; it is not a complete port
of Channelflow's MPI, symmetry, continuation or invariant-solution tools.

## Getting started

Requires Julia 1.11 or later. The repository records the source URLs of its
unregistered dependencies. The implementation described here is on the
`batched-cpu-cuda` branch:

```sh
git clone --branch batched-cpu-cuda https://github.com/Davide-Lasagna-s-Lab/ChannelFlow.jl.git
cd ChannelFlow.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

A complete CPU simulation is:

```julia
using ChannelFlow, Flows, FFTW, Random

Random.seed!(42)
grid = Grid(32, 33, 32, 2π/1.14, 2π/2.5) # Nx, Ny, Nz, Lx, Lz
problem = CouetteFlow(grid, 1/400, 0.01; fftwflags=FFTW.ESTIMATE)
state = random_state(problem, 0.1)

integrate = Flows.flow(problem)
integrate(state, (0.0, 1.0))

U = velocity(state) # spectral PERTURBATION velocity
u = IFFT(U)         # padded physical perturbation velocity
```

`random_state` projects the random velocity onto the divergence-free, no-slip
space and reconstructs pressure. Its amplitude scales the random samples
**before projection**, not the final kinetic energy. `roll_state(problem, 0.1)`
creates a smooth streamwise-independent roll instead. Neither initializer
guarantees transition to turbulence.

For an explicit number of steps, with no shortened final step:

```julia
state = random_state(problem, 0.1)
t = 0.0
for _ = 1:100
    t, gradients = step!(problem.scheme, problem.nlterm,
                         velocity(state), stagepressure(state), t;
                         forcing=problem.forcing, problem.constraint...)
end
```

A monitor can record perturbation energy without storing the full trajectory:

```julia
monitor = Flows.Monitor(state, (t, s) -> kinetic_energy(velocity(s)); oneevery=10)
integrate(state, (1.0, 2.0), monitor) # continue the first example
sample_times = Flows.times(monitor)
energies = Flows.samples(monitor)
```

This observable uses perturbation velocity, so it is not total-flow energy.
Diagnostics allocate and, on the GPU, may synchronize; benchmark the solver
without a monitor when measuring integration cost alone.

The returned `gradients` from `step!` are `(dPdx, dPdz)` from the final implicit stage.
Use this loop when comparing devices or appending uniformly sampled records.

## Running on an NVIDIA GPU

Install CUDA.jl in the environment once. Construct the problem and initial
condition on the CPU, then transfer them explicitly:

```julia
using CUDA, Adapt
CUDA.allowscalar(false)

gpu_problem = adapt(CuArray, problem)
gpu_state = adapt(CuArray, state)

for j = 0:99
    step!(gpu_problem.scheme, gpu_problem.nlterm,
          velocity(gpu_state), stagepressure(gpu_state), j*0.01;
          forcing=gpu_problem.forcing, gpu_problem.constraint...)
end
CUDA.synchronize()             # only needed here to wait for completion
state = adapt(Array, gpu_state) # explicit download for storage or CPU analysis
```

Adaptation transfers the cached factors and creates new device transform
plans. It preserves `Float64`/`ComplexF64` precision. A timestep contains no
full-field host transfers. The constant-bulk-velocity interface currently retrieves mean
mode information and its two pressure gradients on the host each stage.
GPU timings must include a final `CUDA.synchronize()`.

The device Chebyshev backend uses cuFFT on an even extension by default.
Construct the CPU problem with `chebbackend=:gemm` before adaptation to select
cuBLAS matrix multiplication for the complete GPU simulation.
Dense matrix multiplication through cuBLAS is also available through
`ChannelFlow.plan_cheb(U, :gemm)` and `ChannelFlow.plan_icheb(U, :gemm)`. CPU Chebyshev transforms default to FFTW's
DCT-I (`:fftw`). CPU periodic transforms always use FFTW; GPU periodic
transforms use cuFFT. CUDA is optional and is loaded through a Julia extension.

## What is stored?

Directions and array axes are different concepts:

| Object | Array order | Meaning |
|---|---|---|
| `Grid(Nx, Ny, Nz, Lx, Lz)` | constructor arguments `(x,y,z)` | resolved physical point counts |
| `PhysicalField` | `(x,z,y)` | periodic samples and descending Lobatto nodes |
| `SpectralField` | `(kx,kz,n)` | Fourier amplitudes and ordinary Chebyshev coefficients |
| `VectorField` components | `(u,v,w)` | streamwise, wall-normal, spanwise velocity |
| `State` | velocity and stage pressure | data needed to continue CNRK2 |

With `Nxh = floor(Nx/2)+1`, a spectral field has shape `(Nxh,Nz,Ny)`.
`reshape(parent(U), Nxh*Nz, Ny)` is a **zero-copy** matrix: each row is one
Fourier system. This is the batched Helmholtz layout. Adjacent CPU SIMD lanes
or GPU threads process adjacent systems without assembling a packed RHS.

The x half-spectrum stores nonnegative wavenumbers. The z axis follows FFT
order: zero, positive, then negative wavenumbers. The integer pair `(kx,kz)`
corresponds to physical wavenumbers `α = 2π*kx/Lx`, `β = 2π*kz/Lz`.
Chebyshev degree `n` is stored at index `n+1`. Code that previously selected
a profile with `U[:,ix,iz]` must now use `U[ix,iz,:]`; the public grid constructor
continues to take `(Nx,Ny,Nz)`. Restart Julia after changing package versions
because field and solver types have changed.

`physicalsize(grid, Padded())` and `spectralsize(grid, NotPadded())` return
array shapes. Physical fields constructed from a function default to padded
storage; the function is always called as `f(x,y,z)`.

## Equations and driving

The channel occupies $x\in[0,L_x)$, $y\in[-1,1]$, $z\in[0,L_z)$.
The total velocity is $\boldsymbol{v}=U_b(y)\boldsymbol{e}_x+\boldsymbol{u}$.
The stored perturbation $\boldsymbol{u}$ is zero at both walls. We solve

```math
\partial_t\boldsymbol{u}
=\mathcal N(\boldsymbol{v})-\nabla q-\boldsymbol G
 +\nu\Delta\boldsymbol{u}+\nu U_b''\boldsymbol e_x+\boldsymbol f,
\qquad \nabla\cdot\boldsymbol{u}=0.
```

Here $\boldsymbol G=(dP/dx,0,dP/dz)$ is the uniform driving gradient.
A negative streamwise gradient drives positive streamwise flow.

| Constructor | Base flow | Default driving |
|---|---|---|
| `CouetteFlow(grid, nu, dt)` | $U_b=y$ | walls at velocities ±1; zero gradient |
| `PoiseuilleFlow(grid, nu, dt)` | $U_b=1-y^2$ | stationary walls; `dPdx=-2nu` |
| `ChannelFlowProblem(grid, profile, nu, dt)` | `profile(y)` | zero gradient unless specified |

For these standard profiles, `nu=1/Re` uses the half-height and wall speed
(Couette) or laminar centreline speed (Poiseuille). The wall-normal interval
is fixed; arbitrary wall locations are not supported.

Choose **one** driving condition:

```julia
PoiseuilleFlow(grid, nu, dt; pressuregradient=(-2nu, 0))
PoiseuilleFlow(grid, nu, dt; bulkvelocity=(2/3, 0))
```

Bulk velocities refer to the **total** flow, including the base profile.
A custom profile is not automatically a steady Navier–Stokes solution; its
viscous term and specified driving must balance if laminar stationarity is
intended.

A body force implements `forcing(t, U, rhs)` and **adds** spectral acceleration
to the existing `rhs`. It must preserve `U` and run on the same device as its
arguments. `NoForcing()` does nothing. GPU forcing should use device broadcasts
or kernels, not host scalar indexing.

## Spatial method

### Representation and nonlinear products

Each Fourier amplitude is a degree-$P$ Chebyshev polynomial, $P=N_y-1$:

```math
\widehat u(y)=\sum_{n=0}^{P}a_n T_n(y),\qquad
T_n(\cos\theta)=\cos(n\theta),\qquad
y_j=\cos(\pi j/P).
```

These are ordinary coefficients: the constant and highest coefficients are
not implicitly halved. Fourier coefficients use the normalised forward
transform, so a constant field has the same constant coefficient at any
resolution. Derivatives in x and z multiply by $i\alpha$ and $i\beta$;
y derivatives use coefficient recurrences.

Nonlinear products are evaluated on a grid padded by 3/2 in both periodic
directions. Transforming back and truncating removes quadratic Fourier
aliases. Resolved Nyquist planes are excluded and explicitly set to zero.
**The y direction is not padded:** there is currently no Chebyshev dealiasing.
Resolution convergence must therefore check wall-normal tails as well as
Fourier tails and timestep convergence.

All nonlinear forms include the base velocity and its shear:

| Form | Explicit acceleration $\mathcal N$ | Pressure $q$ |
|---|---|---|
| `RotatingForm()` (default) | $\boldsymbol v\times\nabla\times\boldsymbol v$ | $p+|\boldsymbol v|^2/2$ |
| `ChannelFlow.ConvectiveForm()` | $-(\boldsymbol v\cdot\nabla)\boldsymbol v$ | $p$ |
| `ChannelFlow.DivergenceForm()` | $-\nabla\cdot(\boldsymbol v\boldsymbol v)$ | $p$ |

These continuous identities need not give identical under-resolved discrete
trajectories. Changing the form also changes the pressure convention.

### Why Helmholtz problems appear

For one implicit stage and one nonzero Fourier pair, write
$\kappa^2=\alpha^2+\beta^2$ and $D=d/dy$. The momentum and continuity equations
are

```math
(\lambda+\nu\kappa^2-\nu D^2)\widehat{\boldsymbol u}
 +(i\alpha\widehat q,D\widehat q,i\beta\widehat q)
 =\widehat{\boldsymbol R},\qquad
 i\alpha\widehat u+D\widehat v+i\beta\widehat w=0.
```

Taking the divergence gives a pressure Poisson problem. Once pressure is
known, each velocity component satisfies a scalar Helmholtz problem. The
backend solves operators of the form $\theta_0D^2-\theta_1$, hence the velocity
source is **pressure gradient minus R**, not the reverse.

### Tau discretisation and influence matrix

A degree-$P$ solution has $P+1$ unknown coefficients. A second-order equation
supplies its lowest $P-1$ residual equations (degrees 0 through $P-2$);
the two remaining equations impose boundary conditions. This is the **tau
method**. The highest two residual coefficients need not vanish: they are
the tau residuals. The Helmholtz package integrates the coefficient equations
and separates even/odd degrees into structured quasi-tridiagonal systems.
Their factors are cached rather than recomputed each timestep.

Pressure has no independently prescribed wall value. The influence method
finds the pressure boundary values that make velocity satisfy continuity:

1. Solve pressure with provisional zero wall values, then solve normal velocity
   with zero wall values.
2. Add two homogeneous responses: one has unit pressure at the upper wall,
   the other at the lower wall. Their normal velocities remain zero at both
   walls.
3. Choose their amplitudes from a 2×2 system so that $D\widehat v$ also vanishes
   at both walls. These derivative conditions follow from continuity and
   tangential no slip.
4. Apply Gibson's auxiliary tau correction, then solve the two tangential
   momentum equations with the corrected pressure.

Step 4 matters: taking the divergence of the **truncated** momentum equations
also differentiates their tau residuals. Correcting wall values alone does not
remove that discrete divergence. The cached auxiliary pressure/velocity pair
accounts for the two highest residuals. Tests check divergence at **every**
coefficient, while checking momentum only at the retained residual degrees.
The parity construction currently requires **odd `Ny ≥ 5`**.

All nonzero systems are processed in batches. The mean slot uses a harmless
pressure placeholder during those operations and is then overwritten by its
own equations; excluded Nyquist outputs are zeroed.

### The zero Fourier mode

At $(k_x,k_z)=(0,0)$, continuity gives $Dv=0$ and impermeability gives $v=0$.
The tangential equations are scalar one-dimensional Helmholtz problems.
For fixed pressure gradient, it is inserted into their forcing. For fixed
bulk velocity, a cached response to a unit gradient determines the additional
gradient required to obtain the requested total flux.

Normal momentum gives $Dq=R_y$. The retained source is integrated and the
constant is chosen so $\frac12\int_{-1}^1q\,dy=0$. This is a physical mean
condition; setting just the zeroth Chebyshev coefficient to zero would not
impose it. No singular pressure matrix is inverted for this mode.

## CNRK2 time integration and pressure history

CNRK2 has three stages and overall order two. For the signed explicit
acceleration $N_j=\mathcal N+f$, its coefficients are

```math
A=(0,-5/9,-153/128),\quad
B=(1/3,15/16,8/15),\quad
C=(1/6,5/24,1/8).
```

At stage times $t+(0,1/3,3/4)\Delta t$, update $Q=A_jQ+N_j$ and solve

```math
(\lambda_j-\nu\Delta)u_{new}+\nabla q_{new}+G_{new}
=\lambda_j u+\nu\Delta u-\nabla q-G_{old}
 +(B_j/C_j)Q+2\nu U_b''e_x,
\qquad \lambda_j=1/(C_j\Delta t).
```

The first stage overwrites Q. Three banks of factors correspond to the three
shifts; no factorisation occurs during a nominal fixed step. The base-flow
curvature appears twice because it enters both Crank–Nicolson halves.

`stagepressure(state)` is the algebraic pressure retained by this discrete
scheme. It is needed in the next stage and must be saved for an exact restart.
It has no independent evolution equation. `pressure(U, problem, t)` instead
reconstructs instantaneous physical/modified pressure from a divergence-free
velocity, using

```math
\Delta q=\nabla\cdot(\mathcal N+f),\qquad
Dq|_{\pm1}=(\mathcal N+f)_y+\nu D^2v|_{\pm1}.
```

Viscosity drops out of the interior divergence but remains in the wall data.
Use reconstructed pressure to initialise a state; do not replace stage pressure
after every timestep.

`Flows.flow(problem)` uses fixed nominal steps and can shorten the terminal
step to hit the requested time. A shortened step builds temporary factors on
the active backend. For evenly spaced sampling, use an integer number of
nominal steps and carry the final time into the next run.

## Initialisation, diagnostics and restart

`project!(U, problem)` overwrites velocity with a divergence-free, no-slip field
and returns U. It solves a stationary Stokes problem with source $-\Delta U$;
this is a gradient-seminorm projection, not an orthogonal L² projection.
Its multiplier is discarded. Construct a complete state with
`State(U, pressure(U, problem))` afterwards.

Energy diagnostics use the supplied velocity as-is. To obtain total-flow
quantities on the CPU:

```julia
V = copy(velocity(state))
@views parent(V[1])[1,1,:] .+= problem.scheme.baseflow
E = kinetic_energy(V)
D = dissipation_rate(V, problem.scheme.nu)
I = power_input(V, problem.scheme.nu; pressuregradient=(0.0,0.0))
```

They return volume averages:

```math
E=\tfrac12\langle|v|^2\rangle,\qquad
D=\nu\langle|\nabla\times v|^2\rangle,\qquad
I=\frac\nu2[\overline v\cdot D\overline v]_{-1}^{1}
 -G_x\langle v_x\rangle-G_z\langle v_z\rangle.
```

The dissipation identity assumes incompressibility, periodic directions and
impermeable uniformly moving no-slip walls. Body-force power is not included
in `power_input`. Under these conditions the unforced total-energy balance is
$dE/dt=I-D$. Base-flow and perturbation cross terms generally matter.
Spectral inner products use Fourier Parseval weights and the exact unweighted
Chebyshev mass matrix, not a Euclidean norm of coefficients.

`laminar_kinetic_energy`, `laminar_dissipation_rate` and `laminar_power_input`
accept a problem. For Couette flow they are $1/6,\nu,\nu$; for consistently
pressure-driven Poiseuille flow they are $4/15,4\nu/3,4\nu/3$.

```julia
savefield("state.bin", state)
restarted = loadfield("state.bin")
```

Save CPU state after downloading a GPU state. The file stores both velocity
and stage pressure, grid and layout metadata. Recreate the problem with the
same parameters and provide the saved simulation time separately. Legacy
coefficient-first spectral files are converted on load. Julia Serialization
requires compatible type definitions and trusted input; it is not a portable
long-term archive format.

## Validation and performance

```sh
julia --project=. test/runtests.jl
julia --project=. test/physics/runtests.jl
julia --project=test/cuda -e 'using Pkg; Pkg.instantiate()'
julia --project=test/cuda test/cuda/runtests.jl
```

Analytic tests cover normalisation, padding, derivatives, nonlinear forms,
Helmholtz solves, influence/tau correction, pressure, projection and mean-flow
constraints. Batched solves are compared with scalar reference solves using
complex forcing at every polynomial degree. Physical cases check exact
viscous decay, unstable Poiseuille Tollmien–Schlichting growth, and the lower/
upper Waleffe R=400 Couette equilibria. CPU/GPU comparisons test full steps
with scalar device indexing disabled.

These checks establish specific numerical properties; they do not establish
resolution independence for an arbitrary turbulent run. There is no adaptive
CFL controller, MPI decomposition, moving mesh or wall-normal dealiasing.
Time-stepping currently uses double precision. Problem instances own mutable
workspaces and cannot be used concurrently; independent trajectories need
independent problem caches.

See [benchmarks](benchmarks/README.md) for reproducible full-step timings,
profiling, hardware details, and the measured source revision. See the
[source guide](src/README.md) to navigate the implementation.

## Dependencies and references

| Dependency | Purpose |
|---|---|
| [ChebyshevHelmoltzSolvers.jl](https://github.com/Davide-Lasagna-s-Lab/ChebyshevHelmoltzSolvers.jl) | CPU/CUDA batched Helmholtz factors, tau solves, coefficient calculus |
| [Flows.jl](https://github.com/Davide-Lasagna-s-Lab/Flows.jl) | Integration interface, monitors and trajectory storage |
| [FFTW.jl](https://github.com/JuliaMath/FFTW.jl) | CPU Fourier transforms and DCT-I |
| [Adapt.jl](https://github.com/JuliaGPU/Adapt.jl) | Explicit transfer of fields and cached numerical storage |
| [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) (optional) | NVIDIA kernels, cuFFT and cuBLAS |

Julia standard libraries supply linear algebra, random numbers and serialization.
Channelflow C++ is a numerical reference, not a runtime dependency.

- Canuto, Hussaini, Quarteroni & Zang (2006),
  [*Spectral Methods: Fundamentals in Single Domains*](https://doi.org/10.1007/978-3-540-30726-6):
  polynomial expansions, spectral differentiation and tau discretisation.
- Gibson and contributors, [Channelflow `TauSolver`](https://github.com/epfl-ecps/channelflow/blob/master/channelflow/tausolver.cpp)
  and [`RungeKuttaDNS`](https://github.com/epfl-ecps/channelflow/blob/master/channelflow/dnsalgo.cpp):
  reference influence/tau construction and CNRK2 terminology.
- Validation data and literature provenance are recorded with the
  [physical test cases](test/physics/data/README.md).

## Licence

[MIT](LICENSE), copyright Davide Lasagna.
