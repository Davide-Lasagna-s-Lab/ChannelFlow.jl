# Diagnostics

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
``dE/dt=I-D``. Base-flow and perturbation cross terms generally matter.
Spectral inner products use Fourier Parseval weights and the exact unweighted
Chebyshev mass matrix, not a Euclidean norm of coefficients.

`laminar_kinetic_energy`, `laminar_dissipation_rate` and `laminar_power_input`
accept a problem. For Couette flow they are ``1/6,\nu,\nu``; for consistently
pressure-driven Poiseuille flow they are ``4/15,4\nu/3,4\nu/3``.



## Mean profiles and Reynolds stresses

A plane mean is the zero Fourier mode evaluated at the Lobatto points. Add the
base profile to obtain total streamwise mean velocity. For Reynolds stresses,
accumulate ``\langle v_i v_j\rangle_{x,z,t}`` and the corresponding space–time
means, then subtract their product:

```math
\overline{v_i'v_j'}(y)=\langle v_i v_j\rangle_{x,z,t}
 -\langle v_i\rangle_{x,z,t}\langle v_j\rangle_{x,z,t}.
```

Subtracting a different instantaneous plane mean from every sample would omit
fluctuations of the plane mean itself. The [MKM example](mkm590.md) accumulates
both contributions with an online covariance update and retains them in its
restart checkpoint. Sampling should start after the transient and continue long
enough to estimate sampling uncertainty.
