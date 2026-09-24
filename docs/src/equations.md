# Equations and configuration

The channel occupies ``x\in[0,L_x)``, ``y\in[-1,1]``, ``z\in[0,L_z)``.
The total velocity is ``\boldsymbol{v}=U_b(y)\boldsymbol{e}_x+\boldsymbol{u}``.
The stored perturbation ``\boldsymbol{u}`` is zero at both walls. We solve

```math
\partial_t\boldsymbol{u}
=\mathcal N(\boldsymbol{v})-\nabla q-\boldsymbol G
 +\nu\Delta\boldsymbol{u}+\nu U_b''\boldsymbol e_x+\boldsymbol f,
\qquad \nabla\cdot\boldsymbol{u}=0.
```

Here ``\boldsymbol G=(dP/dx,0,dP/dz)`` is the uniform driving gradient.
A negative streamwise gradient drives positive streamwise flow.

| Constructor | Base flow | Default driving |
|---|---|---|
| `CouetteFlow(grid, nu, dt)` | ``U_b=y`` | walls at velocities ±1; zero gradient |
| `PoiseuilleFlow(grid, nu, dt)` | ``U_b=1-y^2`` | stationary walls; `dPdx=-2nu` |
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

## Reynolds-number conventions

The solver takes viscosity, not a friction Reynolds number. With channel
half-height ``h`` and a chosen velocity scale ``U_*``, ``Re=U_*h/\nu``.
Do not interchange a wall-speed Reynolds number, a bulk Reynolds number
``Re_b=U_{\mathrm{bulk}}h/\nu`` and the friction Reynolds number

```math
Re_\tau=\frac{u_\tau h}{\nu},\qquad
u_\tau^2=\frac{\nu}{2}\left[\overline U'(-1)-\overline U'(+1)\right].
```

Here the overbar is a plane average and ``h=1`` in solver coordinates.
For symmetric pressure-driven flow this averages the positive drag from both
walls before taking the square root. At fixed flux, the pressure gradient and
wall stress evolve; specifying viscosity and flux does not prescribe the
instantaneous ``Re_\tau``. For laminar unit-bulk flow,
``U=3(1-y^2)/2`` and ``Re_\tau=\sqrt{3/\nu}``.

The MKM example uses a bulk-velocity scaling, distinct from the default
unit-centreline Poiseuille constructor; see [MKM590](mkm590.md).
