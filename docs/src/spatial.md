# Spatial discretisation and pressure–velocity coupling

This is a primitive-variable Fourier–Chebyshev method in the
Kleiser–Schumann family. The `StokesSolver` performs the role of Gibson's
`TauSolver`: it combines pressure and velocity Helmholtz problems, a wall
influence correction, and a tau correction. It does not advance a separate
pressure evolution equation. [Gibson's implementation](https://github.com/epfl-ecps/channelflow/blob/master/channelflow/tausolver.cpp)
provides the reference algorithm; see also [References](references.md).

## Representation and nonlinear products

With ``\alpha_k=2\pi k/L_x`` and ``\beta_\ell=2\pi\ell/L_z``, each component is represented as

```math
u(x,y,z,t)=\sum_{k,\ell}\sum_{n=0}^{N_y-1}
 a_{k\ell n}(t)T_n(y)\exp\!\left(i\alpha_k x+i\beta_\ell z\right).
```

Real velocity implies conjugate symmetry in the Fourier coefficients. The
stored nonnegative streamwise half-spectrum exploits that symmetry; the
spanwise axis retains its positive and negative Fourier modes.


Each Fourier amplitude is a degree-``P`` Chebyshev polynomial, ``P=N_y-1``:

```math
\widehat u(y)=\sum_{n=0}^{P}a_n T_n(y),\qquad
T_n(\cos\theta)=\cos(n\theta),\qquad
y_j=\cos(\pi j/P).
```

These are ordinary coefficients: the constant and highest coefficients are
not implicitly halved. Fourier coefficients use the normalised forward
transform, so a constant field has the same constant coefficient at any
resolution. Derivatives in x and z multiply by ``i\alpha`` and ``i\beta``;
y derivatives use coefficient recurrences.

Nonlinear products are evaluated on a grid padded by 3/2 in both periodic
directions. Transforming back and truncating removes quadratic Fourier
aliases. Resolved Nyquist planes are excluded and explicitly set to zero.
**The y direction is not padded:** there is currently no Chebyshev dealiasing.
Resolution convergence must therefore check wall-normal tails as well as
Fourier tails and timestep convergence.

All nonlinear forms include the base velocity and its shear:

| Form | Explicit acceleration ``\mathcal N`` | Pressure ``q`` |
|---|---|---|
| `RotatingForm()` (default) | ``\boldsymbol v\times\nabla\times\boldsymbol v`` | ``p+|\boldsymbol v|^2/2`` |
| `ChannelFlow.ConvectiveForm()` | ``-(\boldsymbol v\cdot\nabla)\boldsymbol v`` | ``p`` |
| `ChannelFlow.DivergenceForm()` | ``-\nabla\cdot(\boldsymbol v\boldsymbol v)`` | ``p`` |

These continuous identities need not give identical under-resolved discrete
trajectories. Changing the form also changes the pressure convention.

## Why Helmholtz problems appear

For one implicit stage and one nonzero Fourier pair, write
``\kappa^2=\alpha^2+\beta^2`` and ``D=d/dy``. The momentum and continuity equations
are

```math
(\lambda+\nu\kappa^2-\nu D^2)\widehat{\boldsymbol u}
 +(i\alpha\widehat q,D\widehat q,i\beta\widehat q)
 =\widehat{\boldsymbol R},\qquad
 i\alpha\widehat u+D\widehat v+i\beta\widehat w=0.
```

Taking the divergence gives a pressure Poisson problem. Once pressure is
known, each velocity component satisfies a scalar Helmholtz problem. The
backend solves operators of the form ``\theta_0D^2-\theta_1``, hence the velocity
source is **pressure gradient minus R**, not the reverse.

## Tau discretisation and influence matrix

A degree-``P`` solution has ``P+1`` unknown coefficients. A second-order equation
supplies its lowest ``P-1`` residual equations (degrees 0 through ``P-2``);
the two remaining equations impose boundary conditions. This is the **tau
method**. The highest two residual coefficients need not vanish: they are
the tau residuals. The detailed tau equations, spectral integration, even/odd factorisation and
scalar/batched interfaces are documented in
[ChebyshevHelmoltzSolvers.jl](https://github.com/Davide-Lasagna-s-Lab/ChebyshevHelmoltzSolvers.jl#the-chebyshev-tau-method).
ChannelFlow caches those factors and adds the incompressibility constraint;
it does not refactor them at every nominal timestep.

Pressure has no independently prescribed wall value. The influence method
finds the pressure boundary values that make velocity satisfy continuity:

1. Solve pressure with provisional zero wall values, then solve normal velocity
   with zero wall values.
2. Add two homogeneous responses: one has unit pressure at the upper wall,
   the other at the lower wall. Their normal velocities remain zero at both
   walls.
3. Choose their amplitudes from a 2×2 system so that ``D\widehat v`` also vanishes
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

## The zero Fourier mode

At ``(k_x,k_z)=(0,0)``, continuity gives ``Dv=0`` and impermeability gives ``v=0``.
The tangential equations are scalar one-dimensional Helmholtz problems.
For fixed pressure gradient, it is inserted into their forcing. For fixed
bulk velocity, a cached response to a unit gradient determines the additional
gradient required to obtain the requested total flux.

Normal momentum gives ``Dq=R_y``. The retained source is integrated and the
constant is chosen so ``\frac12\int_{-1}^1q\,dy=0``. This is a physical mean
condition; setting just the zeroth Chebyshev coefficient to zero would not
impose it. No singular pressure matrix is inverted for this mode.

## Connection with Gibson's terminology

| Channelflow terminology | ChannelFlow.jl |
|---|---|
| Fourier–Chebyshev `FlowField` | `SpectralField`, `PhysicalField`, `VectorField` |
| `Ubase` | prescribed base profile and cached coefficients |
| `TauSolver` | `StokesSolver`, with batched influence/tau solves |
| influence correction | homogeneous pressure responses selected by wall continuity |
| tau correction | auxiliary responses removing divergence from truncated momentum residuals |
| `RungeKuttaDNS`, `CNRK2` | `CNRK2`, three substeps, second order |
| constant pressure gradient / bulk velocity | `pressuregradient` / `bulkvelocity` |

These are correspondences of mathematical responsibilities, not identical APIs.
The fourth-order influence correction in the Helmholtz dependency and the
pressure–velocity influence correction here enforce different constraints.
