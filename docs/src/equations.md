# Equations and configuration

## Domain, units and governing equations

The channel is periodic in the streamwise and spanwise directions, with

```math
x\in[0,L_x),\qquad y\in[-1,1],\qquad z\in[0,L_z).
```

Lengths are scaled by the channel **half-height** ``h``, velocities by a
reference speed ``U_{\mathrm{ref}}``, and time by ``h/U_{\mathrm{ref}}``.
The viscosity passed to the constructor is therefore

```math
\nu=\frac{\nu_{\mathrm{dim}}}{U_{\mathrm{ref}}h}
    =\frac{1}{Re_{\mathrm{ref}}}.
```

The solver takes `nu`, not a Reynolds number. Its wall-normal interval is
always ``[-1,1]``; changing physical units does not change this interval.
Pressure is kinematic pressure, scaled by ``U_{\mathrm{ref}}^2``.

Write the total velocity as ``\boldsymbol v=(v_x,v_y,v_z)`` and split pressure
into a uniform driving part and a periodic part:

```math
P(\boldsymbol x,t)=G_x(t)x+G_z(t)z+p(\boldsymbol x,t),
\qquad \boldsymbol G=(G_x,0,G_z).
```

The incompressible Navier–Stokes equations are

```math
\partial_t\boldsymbol v
 +(\boldsymbol v\cdot\nabla)\boldsymbol v
 =-\nabla p-\boldsymbol G+\nu\Delta\boldsymbol v+\boldsymbol f,
\qquad \nabla\cdot\boldsymbol v=0.
```

Thus ``G_x<0`` drives flow in the positive streamwise direction. The uniform
pressure gradient is handled separately from the periodic pressure field.
For the default rotational form, the equivalent equation is

```math
\partial_t\boldsymbol v
 =\boldsymbol v\times\boldsymbol\omega-\nabla q-\boldsymbol G
  +\nu\Delta\boldsymbol v+\boldsymbol f,
\qquad
\boldsymbol\omega=\nabla\times\boldsymbol v,
\qquad q=p+\tfrac12|\boldsymbol v|^2.
```

The stored velocity is a perturbation to a stationary reference profile:

```math
\boldsymbol v=U_b(y)\boldsymbol e_x+\boldsymbol u,
\qquad \boldsymbol u(x,\pm1,z,t)=\boldsymbol0.
```

The nonlinear term uses the **total** velocity. The viscous term contains both
``\nu\Delta\boldsymbol u`` and ``\nu U_b''\boldsymbol e_x``. Consequently,
``U_b`` need not equal the turbulent mean profile, and prescribing it does not
hold the mean flow fixed. See [Initialisation and pressure](pressure.md) for
constructing velocity and stage pressure consistently.

## Plane Couette flow

For plane Couette flow the walls move in opposite directions at speeds
``\pm U_w``. The usual scales are ``h`` and ``U_w``, giving

```math
Re_w=\frac{U_wh}{\nu_{\mathrm{dim}}},\qquad \nu=\frac1{Re_w},
\qquad
\boldsymbol v(x,\pm1,z,t)=(\pm1,0,0).
```

With no imposed pressure gradient or body force, the laminar solution is

```math
U_{\mathrm{lam}}(y)=y,\qquad U_{\mathrm{bulk,lam}}=0,
\qquad G_x=G_z=0.
```

`CouetteFlow` selects this reference profile and zero pressure gradient:

```julia
using ChannelFlow, FFTW

grid = Grid(32, 33, 32, 2π, π) # Nx, Ny, Nz, Lx, Lz; before padding
Re_w = 400.0
dt = 0.01
couette = CouetteFlow(grid, 1/Re_w, dt; fftwflags=FFTW.ESTIMATE)
```

The wall velocities stay fixed throughout the simulation. Zero imposed
pressure gradient leaves the bulk velocity free to evolve. To constrain its
instantaneous streamwise and spanwise values to zero instead, use

```julia
couette_fixed_flux = CouetteFlow(grid, 1/Re_w, dt;
    bulkvelocity=(0.0, 0.0), fftwflags=FFTW.ESTIMATE)
```

This second problem permits the solver to apply a uniform pressure gradient
to enforce the specified fluxes. It is a different driving condition from
unforced Couette flow, even though both have the same laminar solution.
An explicitly imposed nonzero ``G_x`` produces a Couette–Poiseuille flow,
whose laminar profile is

```math
U_{\mathrm{lam}}(y)=y-\frac{G_x}{2\nu}(1-y^2).
```

## Plane Poiseuille flow

Plane Poiseuille flow has stationary walls:

```math
\boldsymbol v(x,\pm1,z,t)=\boldsymbol0.
```

For a constant streamwise pressure gradient and no other forcing, a steady
laminar solution satisfies

```math
0=-G_x+\nu U_{\mathrm{lam}}'',
\qquad
U_{\mathrm{lam}}(y)=-\frac{G_x}{2\nu}(1-y^2),
\qquad
U_{\mathrm{bulk,lam}}=-\frac{G_x}{3\nu}.
```

A common convention uses the laminar centreline velocity ``U_c`` as the
reference speed. Then

```math
Re_c=\frac{U_ch}{\nu_{\mathrm{dim}}},\qquad
\nu=\frac1{Re_c},\qquad
U_{\mathrm{lam}}=1-y^2,\qquad
G_x=-2\nu,\qquad U_{\mathrm{bulk,lam}}=\frac23.
```

These are the defaults of `PoiseuilleFlow`:

```julia
Re_c = 6000.0
poiseuille = PoiseuilleFlow(grid, 1/Re_c, dt;
    fftwflags=FFTW.ESTIMATE)

# The same configuration with the default gradient written explicitly:
poiseuille_fixed_gradient = PoiseuilleFlow(grid, 1/Re_c, dt;
    pressuregradient=(-2/Re_c, 0.0), fftwflags=FFTW.ESTIMATE)
```

Here ``Re_c`` refers to the **laminar reference** centreline velocity, not to
the centreline velocity of the evolving turbulent flow. The turbulent bulk
velocity is an output at fixed pressure gradient; it is not constrained to
``2/3``.

## Fixed pressure gradient or fixed bulk velocity

The two constructor keywords select mutually exclusive conditions:

| Keyword | Prescribed quantity | Quantity determined by the simulation |
|---|---|---|
| `pressuregradient=(Gx, Gz)` | Constant uniform streamwise and spanwise pressure derivatives | Both total bulk velocities |
| `bulkvelocity=(Ubulk, Wbulk)` | Constant total streamwise and spanwise bulk velocities | Both uniform pressure derivatives, at each implicit stage |

Passing both keywords with non-`nothing` values raises an error. The bulk
velocities are volume averages of the **total** velocity:

```math
U_{\mathrm{bulk}}(t)=\frac12\int_{-1}^{1}
 \langle v_x\rangle_{xz}(y,t)\,dy,
\qquad
W_{\mathrm{bulk}}(t)=\frac12\int_{-1}^{1}
 \langle v_z\rangle_{xz}(y,t)\,dy.
```

At fixed flux, the solver modifies the ``(k,l)=(0,0)`` velocity mode using
a cached response to a unit pressure gradient. It chooses the response
amplitude to satisfy each bulk constraint at each implicit stage. The
streamwise perturbation target is obtained by subtracting the mean of
``U_b`` from the requested total bulk velocity. Higher Fourier modes do not
contribute to the bulk mean.

`CouetteFlow` and `ChannelFlowProblem` default to zero pressure gradient.
`PoiseuilleFlow` defaults to ``(-2\nu,0)``, but supplying `bulkvelocity`
automatically disables that default gradient.

Changing either driving keyword **does not rescale the reference profile**:
`PoiseuilleFlow` always stores ``U_b=1-y^2`` and `CouetteFlow` always stores
``U_b=y``. A different mean profile is represented by the perturbation's zero
Fourier mode. For an alternative reference profile, use
`ChannelFlowProblem(grid, profile, nu, dt; ...)` directly.

### Example: prescribe the bulk Reynolds number

Define the half-height bulk Reynolds number by

```math
Re_{b,h}=\frac{U_{\mathrm{bulk,dim}}h}{\nu_{\mathrm{dim}}}
        =\frac{U_{\mathrm{bulk}}}{\nu}.
```

Some publications use the **full height** instead:
``Re_{b,2h}=2Re_{b,h}``. Always check which definition is quoted.
With bulk velocity as the velocity scale, set ``U_{\mathrm{bulk}}=1`` and
``\nu=1/Re_{b,h}``. The unit-bulk laminar profile is ``3(1-y^2)/2``:

```julia
Re_bulk_halfheight = 10000.0
nu = 1/Re_bulk_halfheight

# Choose a reference profile whose bulk mean already matches the target.
poiseuille_unit_bulk = ChannelFlowProblem(grid, y -> 1.5*(1-y^2), nu, dt;
    bulkvelocity=(1.0, 0.0), fftwflags=FFTW.ESTIMATE)

# The convenience constructor describes the same walls, viscosity and flux.
# Its reference profile remains 1-y^2, so the perturbation carries the rest.
poiseuille_unit_bulk_alt = PoiseuilleFlow(grid, nu, dt;
    bulkvelocity=(1.0, 0.0), fftwflags=FFTW.ESTIMATE)
```

For the first reference profile, zero perturbation velocity has the correct
bulk mean. For the second, it does not. `random_state(problem, amplitude)`
projects onto the problem's bulk constraint as well as enforcing
incompressibility and no slip. A consistent periodic pressure must also be
initialised; `zero_state` alone only allocates zero fields.

At fixed unit bulk, the **laminar** pressure gradient is ``G_x=-3\nu``.
In turbulence the required gradient changes with the evolving wall drag;
it must not also be prescribed as ``-3\nu``.

## Friction Reynolds number and wall stress

For pressure-driven flow with positive bulk velocity, define the drag from
the two stationary walls, per unit density, by

```math
d_-(t)=\nu\left.\partial_y\langle v_x\rangle_{xz}\right|_{y=-1},
\qquad
d_+(t)=-\nu\left.\partial_y\langle v_x\rangle_{xz}\right|_{y=+1}.
```

For statistically stationary channel flow, let an overbar denote a time
average. The conventional friction velocity and Reynolds number use the
mean drag:

```math
u_\tau^2=\frac{\overline{d_-}+\overline{d_+}}2,
\qquad Re_\tau=\frac{u_{\tau,\mathrm{dim}}h}{\nu_{\mathrm{dim}}}
             =\frac{u_\tau}{\nu}.
```

Without an additional mean body force, integrating streamwise momentum
across the channel gives

```math
\frac{dU_{\mathrm{bulk}}}{dt}
 =-G_x-\frac{d_-+d_+}{2}.
```

At fixed pressure gradient, **statistical stationarity** therefore implies

```math
u_\tau^2=-G_x,\qquad
Re_\tau=\frac{\sqrt{-G_x}}{\nu},\qquad
G_x=-(\nu Re_\tau)^2.
```

These relations concern equilibrium mean drag. During startup the bulk flow
can accelerate, and the instantaneous wall-stress estimate need not match
``\sqrt{-G_x}/\nu``. At fixed bulk velocity, the acceleration is zero and the
pressure gradient adjusts to the wall drag. Its time average gives
``Re_\tau=\sqrt{-\overline{G_x}}/\nu``. Neither viscosity nor bulk velocity
alone fixes that turbulent friction Reynolds number.

These wall-drag signs apply to Poiseuille flow. For pure Couette flow, use
the magnitude of the shear at each moving wall: the laminar slopes are both
``+1``, so ``u_{\tau,\mathrm{lam}}=\sqrt{\nu}`` and
``Re_{\tau,\mathrm{lam}}=\sqrt{Re_w}``. A zero pressure gradient does **not**
imply zero friction in Couette flow; the walls supply momentum and energy.

### Example: fixed gradient for a target friction Reynolds number

For the default unit-centreline Poiseuille reference, ``G_x=-2\nu``. Hence
``Re_\tau=\sqrt{2/\nu}`` in statistical equilibrium. Choose viscosity as
follows to target ``Re_\tau=180``:

```julia
Re_tau = 180.0
nu = 2/Re_tau^2 # Re_c = Re_tau^2/2 = 16200

poiseuille_retau = PoiseuilleFlow(grid, nu, dt;
    pressuregradient=(-2nu, 0.0), fftwflags=FFTW.ESTIMATE)
```

More generally, choose a velocity scale and viscosity first, then set
`Gx = -(nu*Re_tau)^2`. For example, **friction units** take
``U_{\mathrm{ref}}=u_\tau``, so ``\nu=1/Re_\tau`` and ``G_x=-1``:

```julia
Re_tau = 180.0
poiseuille_friction_units = PoiseuilleFlow(grid, 1/Re_tau, dt;
    pressuregradient=(-1.0, 0.0), fftwflags=FFTW.ESTIMATE)
```

In this last example ``1-y^2`` is only the stored reference profile; the
laminar solution in friction units is ``(Re_\tau/2)(1-y^2)``. These two
examples use different velocity and time units. Their dimensionless time
steps are not physically equivalent, and the step must be chosen for the
actual velocities and grid. The small `grid` above illustrates construction;
it is not a resolution recommendation for turbulent DNS.

For comparison data and a documented resolution choice, see the
[Moser–Kim–Mansour reference database](https://turbulence.oden.utexas.edu/MKM_1999.html)
and the [MKM590 example](mkm590.md). The latter uses bulk-velocity scaling.

## Additional body forcing

A body force implements `forcing(t, U, rhs)` and **adds** spectral acceleration
to the existing `rhs`. It must preserve `U` and run on the same device as its
arguments. `NoForcing()` does nothing. GPU forcing should use device broadcasts
or kernels, not host scalar indexing.

The wall-stress/pressure-gradient relations above assume no additional mean
streamwise forcing. With such forcing, the bulk momentum balance also contains
its volume average. A custom reference profile is a steady laminar solution
only when its viscous term, imposed gradient and body force balance.
