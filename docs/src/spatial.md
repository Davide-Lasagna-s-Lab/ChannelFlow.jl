# Spatial discretisation and pressure–velocity coupling

The spatial discretisation uses Fourier series in the periodic streamwise
and spanwise directions and Chebyshev polynomials in the wall-normal
direction. Velocity and pressure are expanded in this tensor-product basis;
spatial derivatives are evaluated spectrally, while nonlinear products are
evaluated in physical space.

## Fourier–Chebyshev representation

Define the fundamental wavenumbers ``\alpha=2\pi/L_x`` and ``\beta=2\pi/L_z``.
The integers ``k`` and ``l`` label Fourier modes, whose wavenumbers are
``\alpha k`` and ``\beta l``. Each velocity component is represented as

```math
u(x,y,z,t)=\sum_{k,l}\sum_{n=0}^{N_y-1}
 a_{k,l,n}(t)T_n(y)\exp\!\left(i\alpha k x+i\beta l z\right).
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
resolution.

## Spectral differentiation

The Fourier basis functions are eigenfunctions of differentiation:

```math
\partial_x e^{i(\alpha k x+\beta l z)}
 =i\alpha k e^{i(\alpha k x+\beta l z)},\qquad
\partial_z e^{i(\alpha k x+\beta l z)}
 =i\beta l e^{i(\alpha k x+\beta l z)}.
```

Consequently, differentiation in either periodic direction acts independently
on each Fourier–Chebyshev coefficient:

```math
(\partial_x u)_{k,l,n}=i\alpha k a_{k,l,n},\qquad
(\partial_z u)_{k,l,n}=i\beta l a_{k,l,n}.
```

Second derivatives multiply these coefficients by ``-(\alpha k)^2`` and
``-(\beta l)^2``, respectively. In particular, the zero Fourier mode has
zero derivative in both periodic directions.

Wall-normal differentiation instead couples Chebyshev degrees within each
Fourier mode. Suppressing the Fourier indices, write

```math
\widehat u(y)=\sum_{n=0}^{P}a_nT_n(y),\qquad
D\widehat u(y)=\sum_{n=0}^{P-1}b_nT_n(y),\qquad D=\frac{d}{dy}.
```

The Chebyshev identity
``2T_n=T_{n+1}'/(n+1)-T_{n-1}'/(n-1)`` for ``n\ge2``, together with
``T_1'=T_0`` and ``T_2'=4T_1``, gives the descending recurrence

```math
\begin{aligned}
b_P&=b_{P+1}=0,\\
b_n&=b_{n+2}+2(n+1)a_{n+1},\qquad n=P-1,P-2,\ldots,1,\\
b_0&=a_1+\tfrac12 b_2.
\end{aligned}
```

The distinct formula for ``b_0`` follows from the ordinary-coefficient
convention used above. For example, differentiating ``T_2`` gives ``4T_1``,
and differentiating ``T_1`` gives ``T_0``. Applying the same recurrence to
``b_n`` gives the second derivative. These operations differentiate the
represented polynomial exactly, apart from floating-point roundoff; their
accuracy for a general function depends on the resolution of its expansion.

Combining the periodic and wall-normal derivatives, the Laplacian of each
Fourier amplitude is

```math
\widehat{\Delta u}_{k,l}(y)
 =\left[D^2-(\alpha k)^2-(\beta l)^2\right]
   \widehat u_{k,l}(y).
```

## Nonlinear forms and computational cost

Let ``\boldsymbol v=U_b\boldsymbol e_x+\boldsymbol u`` denote the total
velocity and ``\boldsymbol\omega=\nabla\times\boldsymbol v`` its vorticity.
For an incompressible field,

```math
(\boldsymbol v\cdot\nabla)\boldsymbol v
 =\nabla\cdot(\boldsymbol v\otimes\boldsymbol v)
 =\nabla\!\left(\tfrac12|\boldsymbol v|^2\right)
  -\boldsymbol v\times\boldsymbol\omega.
```

These identities give three equivalent continuous forms of the nonlinear
acceleration. In the rotational form the kinetic-energy gradient is absorbed
into the modified pressure ``q``. Every form includes the base velocity and
its shear through the total velocity.

| Form | Explicit acceleration ``\mathcal N`` | Pressure ``q`` | Transform cost per evaluation |
|---|---|---|---|
| Rotational (default) | ``\boldsymbol v\times\boldsymbol\omega`` | ``p+\tfrac12\boldsymbol v\cdot\boldsymbol v`` | 6 inverse + 3 forward: velocity and vorticity to physical space, then the three acceleration components to spectral space |
| Convective | ``-(\boldsymbol v\cdot\nabla)\boldsymbol v`` | ``p`` | 12 inverse + 3 forward: velocity and its nine derivatives to physical space, then the three acceleration components to spectral space |
| Divergence | ``-\nabla\cdot(\boldsymbol v\otimes\boldsymbol v)`` | ``p`` | 3 inverse + 9 forward: velocity to physical space, then all nine velocity products to spectral space before differentiation |

One transform here means a complete scalar Fourier–Chebyshev transform
between spectral coefficients and the padded physical grid. The counts are
for one nonlinear evaluation, excluding pressure–velocity solves and time
integration. The divergence count uses the full nine-component tensor;
exploiting its symmetry would reduce the forward count to six, but that
optimisation is not used here.

The rotational form requires fewer transforms than the full-gradient
convective form and avoids storing all nine physical velocity derivatives.
Vorticity is formed spectrally before transformation. The divergence form
instead differentiates the transformed products. These counts explain an
important part of the computational cost, but are not timing ratios:
derivatives, pointwise products, memory traffic and transform direction also
contribute.

The continuous equivalence assumes incompressibility and the corresponding
pressure definition. Truncation and aliasing can make the discrete forms
differ, particularly for under-resolved fields. Changing the form also
requires a consistent change in pressure convention.

## Dealiasing

Nonlinear products are evaluated on a grid padded by 3/2 in both periodic
directions. Transforming back and truncating removes quadratic Fourier
aliases. Resolved Nyquist planes are excluded and explicitly set to zero.
**The y direction is not padded:** there is currently no Chebyshev dealiasing.
Resolution convergence must therefore check wall-normal tails as well as
Fourier tails and timestep convergence.

### Example: a quadratic product on a periodic grid

Consider one periodic direction with ``L_x=2\pi``, so ``\alpha=1``, and
``N_x=12`` points. Excluding the Nyquist mode, the retained Fourier modes are
``k=-5,\ldots,5``. Let

```math
f(x)=\cos(4x).
```

This function lies within the retained spectrum, but its square contains a
higher harmonic:

```math
f(x)^2=\tfrac12+\tfrac12\cos(8x).
```

The exact projection of this product onto the retained modes is simply
``1/2``, because the modes ``k=\pm8`` must be discarded.

**Without padding**, the product is evaluated at ``x_j=2\pi j/12``.
At these points,

```math
e^{i8x_j}=e^{-i4x_j},\qquad
\cos(8x_j)=\cos(4x_j).
```

The sampled product cannot distinguish mode 8 from mode −4. Its discrete
transform therefore reports the incorrect retained field

```math
\tfrac12+\tfrac12\cos(4x).
```

The unwanted contribution has already folded into the resolved spectrum;
truncating the transform afterwards cannot remove it.

**With 3/2 padding**, evaluate the same function on ``18`` points instead.
The product modes ``k=\pm8`` are now distinguishable and lie below the padded
Nyquist frequency. Transforming the product and retaining only
``k=-5,\ldots,5`` removes them, leaving the correct constant ``1/2``:

| Product evaluation | Nonzero Fourier coefficients after truncation |
|---|---|
| 12 points, without padding | ``a_0=1/2``, spurious ``a_{4}=a_{-4}=1/4`` |
| 18 points, with 3/2 padding | ``a_0=1/2`` only |

Padding means embedding the **spectral coefficients** in a larger spectrum
with zeros in the additional modes, then transforming to the finer physical
grid. It does not mean appending zeros to physical samples. After forming
the product, transform back and discard the extra Fourier modes. The same
procedure is applied independently in x and z; it does not remove aliasing
in the unpadded Chebyshev direction.

## From Navier–Stokes to the discrete Helmholtz problems

The Helmholtz equations arise when diffusion is treated implicitly in time
and the periodic directions are expanded in Fourier series. This section
derives the stage equations, their reduction to one-dimensional boundary-value
problems, and their finite-dimensional Chebyshev representation.

The progression follows Sections 4.4–4.6 of
[Gibson's Channelflow manual](https://download.savannah.gnu.org/releases/channelflow/channelflow_manual-0.9.13.pdf).
That older manual describes different time-stepping variants; the stage
coefficients below are those of the current CNRK2 method used here.

### 1. Separate known nonlinear terms from the implicit terms

Use ``\boldsymbol u=(u,v,w)`` for the perturbation velocity and
``\boldsymbol v_{\mathrm{tot}}=U_b(y)\boldsymbol e_x+\boldsymbol u`` for the
total velocity. Define the signed explicit acceleration

```math
\boldsymbol F(\boldsymbol u,t)
 =\mathcal N(\boldsymbol v_{\mathrm{tot}})+\boldsymbol f(t).
```

Here ``\mathcal N`` is one of the nonlinear forms above; a velocity-dependent
body force is also evaluated explicitly. Let ``q`` be the corresponding
periodic pressure (modified pressure for the rotational form), and
``\boldsymbol G=(G_x,0,G_z)`` the uniform pressure gradient. Then

```math
\partial_t\boldsymbol u
 =\boldsymbol F(\boldsymbol u,t)
  +\nu\Delta\boldsymbol u+\nu U_b''\boldsymbol e_x
  -\nabla q-\boldsymbol G,
\qquad \nabla\cdot\boldsymbol u=0,
\qquad \boldsymbol u|_{y=\pm1}=\boldsymbol0.
```

The base profile is stationary, so it has no time derivative. Its curvature
remains in the diffusion term. The nonlinear term includes both interaction
with the base profile and perturbation self-advection.

### 2. Discretise one time step with CNRK2

Let ``t^n=n\Delta t``. Write ``\boldsymbol u^{(0)}=\boldsymbol u^n`` and
``\boldsymbol u^{(3)}=\boldsymbol u^{n+1}``, with three stages ``j=1,2,3``.
Parenthesised superscripts denote stages, not powers. The pressure is updated
alongside the velocity at each stage to impose incompressibility.

First evaluate the explicit term at the known velocity and accumulate its
Runge–Kutta history:

```math
\begin{aligned}
\boldsymbol F^{(j-1)}
 &=\boldsymbol F\!\left(\boldsymbol u^{(j-1)},t^n+c_{j-1}\Delta t\right),\\
\boldsymbol Q^{(j)}
 &=A_j\boldsymbol Q^{(j-1)}+\boldsymbol F^{(j-1)},
\qquad \boldsymbol Q^{(0)}=\boldsymbol0,
\end{aligned}
```

where ``(c_0,c_1,c_2)=(0,1/3,3/4)``. The coefficients are

| Stage ``j`` | ``A_j`` | ``B_j`` | ``C_j`` |
|---|---|---|---|
| 1 | ``0`` | ``1/3`` | ``1/6`` |
| 2 | ``-5/9`` | ``15/16`` | ``5/24`` |
| 3 | ``-153/128`` | ``8/15`` | ``1/8`` |

They match Gibson's [CNRK2 implementation](https://github.com/epfl-ecps/channelflow/blob/master/channelflow/dnsalgo.cpp).
The velocity update is

```math
\begin{aligned}
\boldsymbol u^{(j)}-\boldsymbol u^{(j-1)}
={}&B_j\Delta t\,\boldsymbol Q^{(j)}\\
 &+C_j\Delta t\Big[
 \nu\Delta\boldsymbol u^{(j)}+\nu\Delta\boldsymbol u^{(j-1)}
 +2\nu U_b''\boldsymbol e_x\\
 &\hspace{28mm}
 -\nabla q^{(j)}-\nabla q^{(j-1)}
 -\boldsymbol G^{(j)}-\boldsymbol G^{(j-1)}\Big].
\end{aligned}
```

The old and new viscous terms have equal weights: this is the
Crank–Nicolson part of the scheme. The base curvature appears twice because
it contributes to both halves. The coefficients ``C_j`` are stage weights;
``C_j\Delta t`` must not be interpreted as the elapsed time between stage
labels. The three-stage method has overall order two; see
[CNRK2 time integration](timestepping.md).

Divide by ``C_j\Delta t`` and move the unknown terms to the left. With
``\lambda_j=1/(C_j\Delta t)``, this gives

```math
(\lambda_j-\nu\Delta)\boldsymbol u^{(j)}
 +\nabla q^{(j)}+\boldsymbol G^{(j)}
 =\boldsymbol R^{(j)},
```

where the entire right-hand side is known:

```math
\boldsymbol R^{(j)}
 =\lambda_j\boldsymbol u^{(j-1)}
  +\nu\Delta\boldsymbol u^{(j-1)}
  -\nabla q^{(j-1)}-\boldsymbol G^{(j-1)}
  +\frac{B_j}{C_j}\boldsymbol Q^{(j)}
  +2\nu U_b''\boldsymbol e_x.
```

At fixed pressure gradient, both gradient terms equal the prescribed value.
At fixed bulk velocity, the old gradient is obtained from the known state's
bulk momentum balance and the new gradient is an unknown enforcing the new
stage's flux constraint. The uniform gradient and base curvature contribute
only to the zero Fourier mode.

### 3. Transform the stage equations in the periodic directions

For a nonzero Fourier pair ``(k,l)``, define

```math
\kappa^2=(\alpha k)^2+(\beta l)^2,\qquad
D=\frac{d}{dy},\qquad
\Delta\longrightarrow D^2-\kappa^2.
```

Suppress the stage superscript and mode indices on the amplitudes. Substituting
``\partial_x\to i\alpha k`` and ``\partial_z\to i\beta l`` gives

```math
\begin{aligned}
[\lambda_j+\nu\kappa^2-\nu D^2]\widehat u
   +i\alpha k\widehat q&=\widehat R_x,\\
[\lambda_j+\nu\kappa^2-\nu D^2]\widehat v
   +D\widehat q&=\widehat R_y,\\
[\lambda_j+\nu\kappa^2-\nu D^2]\widehat w
   +i\beta l\widehat q&=\widehat R_z,\\
i\alpha k\widehat u+D\widehat v+i\beta l\widehat w&=0.
\end{aligned}
```

The wall conditions are

```math
\widehat u(\pm1)=\widehat v(\pm1)=\widehat w(\pm1)=0.
```

Every retained Fourier pair now has its own one-dimensional problem in
``y``. Nonlinear interactions between different modes have already been
included in ``\widehat{\boldsymbol R}``; the implicit solve does not couple
different Fourier pairs. The zero mode also contains the uniform driving
terms and is treated separately below.

### 4. Eliminate velocity from the pressure equation

Multiply streamwise momentum by ``i\alpha k``, differentiate normal
momentum with ``D``, and multiply spanwise momentum by ``i\beta l``.
Adding the three equations gives

```math
\begin{aligned}
 &[\lambda_j+\nu\kappa^2-\nu D^2]
   (i\alpha k\widehat u+D\widehat v+i\beta l\widehat w)\\
 &\qquad +(D^2-\kappa^2)\widehat q
 =i\alpha k\widehat R_x+D\widehat R_y+i\beta l\widehat R_z.
\end{aligned}
```

The first term vanishes by continuity. Thus the pressure satisfies

```math
(D^2-\kappa^2)\widehat q
 =i\alpha k\widehat R_x+D\widehat R_y+i\beta l\widehat R_z.
```

This is the Fourier representation of ``\Delta q=\nabla\cdot\boldsymbol R``.
It is a Poisson equation in physical space and a shifted second-order
boundary-value problem for each nonzero Fourier mode.

Pressure wall values are **not** prescribed independently of velocity.
Normal momentum at the wall implies

```math
D\widehat q(\pm1)=\widehat R_y(\pm1)+\nu D^2\widehat v(\pm1),
```

but the normal velocity curvature on the right is not yet known. Tangential
no slip and continuity also imply

```math
D\widehat v(\pm1)=0.
```

The influence-matrix method determines the pressure boundary data so that
these coupled velocity and continuity requirements are satisfied. The wall
identities here describe the differential problem; the finite-degree tau
residuals must also be accounted for, as explained below.

### 5. Recover velocity through three Helmholtz problems

Once the pressure has been determined, each momentum equation has the same
scalar operator. Define the positive shift
``\mu_j=\lambda_j+\nu\kappa^2``. Rearranging gives

```math
\begin{aligned}
(\nu D^2-\mu_j)\widehat u
 &=i\alpha k\widehat q-\widehat R_x,\\
(\nu D^2-\mu_j)\widehat v
 &=D\widehat q-\widehat R_y,\\
(\nu D^2-\mu_j)\widehat w
 &=i\beta l\widehat q-\widehat R_z.
\end{aligned}
```

These are Helmholtz problems with homogeneous Dirichlet wall conditions.
The normal velocity and pressure must be determined together through the
influence correction; the two tangential velocities can then be recovered.
The common scalar form is

```math
(\theta_0D^2-\theta_1)\phi=f.
```

For pressure, ``(\theta_0,\theta_1)=(1,\kappa^2)``. For velocity,
``(\theta_0,\theta_1)=(\nu,\mu_j)``. This rearrangement explains the sign of
the velocity forcing: it is the pressure derivative **minus** the known
momentum right-hand side. The pressure operator is independent of the time
step; the velocity operator depends on both the Fourier mode and stage.

### 6. Form the finite-dimensional Chebyshev equations

The preceding equations are discrete in time and Fourier modes, but still
written as differential equations in ``y``. To obtain algebraic equations,
expand each amplitude through degree ``P=N_y-1``. For a generic scalar
problem, write

```math
\phi(y)=\sum_{m=0}^{P}a_mT_m(y),\qquad
f(y)=\sum_{m=0}^{P}f_mT_m(y).
```

Let ``\boldsymbol D`` be the ``(P+1)\times(P+1)`` **coefficient** differentiation
matrix defined by the recurrence in the differentiation section. It maps
Chebyshev coefficients to derivative coefficients, not nodal values to nodal
derivatives. Then ``\boldsymbol D^2\boldsymbol a`` contains the coefficients
of ``D^2\phi``. Define ``\boldsymbol S`` to select coefficient rows
``0,\ldots,P-2``. The scalar tau equations and Dirichlet conditions are

```math
\begin{bmatrix}
\boldsymbol S(\theta_0\boldsymbol D^2-\theta_1\boldsymbol I)\\
\boldsymbol b_+^{\mathsf T}\\
\boldsymbol b_-^{\mathsf T}
\end{bmatrix}\boldsymbol a
=
\begin{bmatrix}
\boldsymbol S\boldsymbol f\\
\phi(+1)\\
\phi(-1)
\end{bmatrix},
\qquad
\boldsymbol b_+^{\mathsf T}=(1,1,\ldots,1),\quad
\boldsymbol b_-^{\mathsf T}=(1,-1,\ldots,(-1)^P).
```

There are ``P-1`` differential equations and two boundary equations for
``P+1`` coefficients. This matrix displays the mathematical discretisation;
the actual solution uses spectral integration and even/odd factorisations
rather than assembling this dense matrix.

For the coupled momentum equations, let ``\boldsymbol u_c,\boldsymbol v_c,
\boldsymbol w_c,\boldsymbol q_c`` denote coefficient vectors and set
``\boldsymbol H_j=\mu_j\boldsymbol I-\nu\boldsymbol D^2``. The retained
momentum rows are

```math
\begin{aligned}
\boldsymbol S(\boldsymbol H_j\boldsymbol u_c+i\alpha k\boldsymbol q_c)
 &=\boldsymbol S\boldsymbol r_x,\\
\boldsymbol S(\boldsymbol H_j\boldsymbol v_c+\boldsymbol D\boldsymbol q_c)
 &=\boldsymbol S\boldsymbol r_y,\\
\boldsymbol S(\boldsymbol H_j\boldsymbol w_c+i\beta l\boldsymbol q_c)
 &=\boldsymbol S\boldsymbol r_z.
\end{aligned}
```

They are accompanied by the six velocity wall conditions
``\boldsymbol b_\pm^{\mathsf T}\boldsymbol u_c=
\boldsymbol b_\pm^{\mathsf T}\boldsymbol v_c=
\boldsymbol b_\pm^{\mathsf T}\boldsymbol w_c=0`` and coefficient continuity

```math
i\alpha k\boldsymbol u_c+\boldsymbol D\boldsymbol v_c
 +i\beta l\boldsymbol w_c=\boldsymbol0.
```

A subtlety appears when deriving a separate pressure equation from these
**truncated** momentum equations. Their polynomial residual can have the form

```math
(\lambda_j-\nu\Delta)\widehat{\boldsymbol u}
 +\widehat{\nabla q}-\widehat{\boldsymbol R}
 =\boldsymbol\tau_{P-1}T_{P-1}+\boldsymbol\tau_PT_P
 \equiv\boldsymbol\tau(y).
```

Taking its divergence while imposing coefficient continuity gives

```math
(D^2-\kappa^2)\widehat q
 =\widehat{\nabla\cdot\boldsymbol R}
  +i\alpha k\tau_x+D\tau_y+i\beta l\tau_z.
```

In particular, ``D\tau_y`` contains lower-degree terms: differentiation and
discarding the highest residual equations do not commute. Simply solving the
uncorrected pressure Poisson equation and adjusting its wall values is
therefore insufficient to reproduce the coupled discrete equations.
This is why the influence correction is supplemented by a **tau correction**.
The following section describes how both corrections are constructed.

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

## Zero Fourier indices and the mean mode

A zero Fourier index means independence of one periodic coordinate. Only the
pair ``(k,l)=(0,0)`` requires a different pressure–velocity formulation:

| Fourier pair | Spatial dependence | Consequence |
|---|---|---|
| ``k=0``, ``l\ne0`` | Independent of ``x``, varying in ``y,z`` | ``\kappa^2=(\beta l)^2>0``; use the nonzero-mode equations |
| ``k\ne0``, ``l=0`` | Independent of ``z``, varying in ``x,y`` | ``\kappa^2=(\alpha k)^2>0``; use the nonzero-mode equations |
| ``k=l=0`` | Varying only in ``y`` | ``\kappa=0``; use the mean-mode equations derived below |

For example, when ``k=0`` but ``l\ne0``, continuity reads
``D\widehat v+i\beta l\widehat w=0``. It does **not** force
``\widehat v`` to vanish: streamwise-independent rolls are allowed.
Likewise, for ``l=0`` and ``k\ne0``, continuity is
``i\alpha k\widehat u+D\widehat v=0``. The pressure Poisson operator and
wall influence correction remain applicable to both cases.

### Meaning of the ``(0,0)`` coefficient

With the normalised Fourier convention, the zero coefficient is exactly the
instantaneous plane average:

```math
u_{00}(y,t)=\langle u\rangle_{xz}
 =\frac{1}{L_xL_z}\int_0^{L_x}\int_0^{L_z}u(x,y,z,t)\,dz\,dx.
```

It is a function of ``y`` represented by all ``N_y`` Chebyshev coefficients,
not a single constant and not a time-averaged profile. The total mean
streamwise velocity is ``U_b(y)+u_{00}(y,t)``. In turbulent flow this profile
need not equal the laminar base profile.

Real physical fields have real ``(0,0)`` coefficients. The complex Fourier
storage does not give this mode an independent imaginary physical part.

### Continuity eliminates the mean normal velocity

At ``k=l=0``, continuity reduces to

```math
Dv_{00}=0.
```

Thus ``v_{00}`` is constant in ``y``. Impermeability at either wall fixes the
constant to zero:

```math
v_{00}(y)=0\qquad\text{for every }y.
```

This statement concerns the plane average only; the instantaneous
wall-normal velocity and its nonzero Fourier modes can be nonzero.
The two tangential mean velocities remain unknown functions of ``y``.

### Stage equations for the tangential velocities

Retain the uniform pressure derivatives explicitly. At CNRK2 stage ``j``,
with ``\lambda_j=1/(C_j\Delta t)``, the mean equations are

```math
\begin{aligned}
(\lambda_j-\nu D^2)u_{00}+G_x&=R_{x,00},\\
(\lambda_j-\nu D^2)w_{00}+G_z&=R_{z,00},\\
Dq_{00}&=R_{y,00},
\end{aligned}
\qquad
u_{00}(\pm1)=w_{00}(\pm1)=0.
```

Stage superscripts on the unknowns and right-hand sides are suppressed here.
The source ``\boldsymbol R_{00}`` is the zero mode of the known stage source
derived above, including the old pressure gradient and base-flow curvature.
The wall conditions apply to the perturbation; the reference profile supplies
the moving-wall velocities in Couette flow.

The periodic pressure ``q_{00}(y)`` has no streamwise or spanwise derivative,
so it does not enter the two tangential equations. The uniform gradients
``G_x`` and ``G_z`` are separate quantities: they cannot be represented by a
periodic zero-mode pressure. The tangential operator has homogeneous Dirichlet
conditions and is invertible for ``\nu>0`` and ``\lambda_j>0``. There is no
singular velocity solve at ``\kappa=0``.

### Prescribed pressure gradient

When ``G_x`` and ``G_z`` are given, solve

```math
(\nu D^2-\lambda_j)u_{00}=G_x-R_{x,00},\qquad
(\nu D^2-\lambda_j)w_{00}=G_z-R_{z,00},
```

with zero wall values. Both use the same scalar Helmholtz operator.
A spatially uniform gradient enters only the degree-zero Chebyshev forcing
coefficient. The resulting bulk velocities are outputs, not constraints.

An equivalent construction first solves the zero-gradient particular problems

```math
(\nu D^2-\lambda_j)u_*=-R_{x,00},\qquad
(\nu D^2-\lambda_j)w_*=-R_{z,00},
```

and the unit-gradient response

```math
(\nu D^2-\lambda_j)c=1,\qquad c(\pm1)=0.
```

Then

```math
u_{00}=u_*+G_xc,\qquad w_{00}=w_*+G_zc.
```

The response is negative in the interior, so a negative imposed pressure
gradient adds positive velocity. The same response can be reused whenever
the viscosity, time step, stage and wall-normal resolution remain unchanged.

### Prescribed total bulk velocity

Define the wall-normal average functional

```math
\mathcal B[\phi]=\frac12\int_{-1}^{1}\phi(y)\,dy.
```

For requested **total** bulk velocities ``U_{\mathrm{bulk}}`` and
``W_{\mathrm{bulk}}``, the perturbation constraints are

```math
\mathcal B[u_{00}]=U_{\mathrm{bulk}}-\mathcal B[U_b],\qquad
\mathcal B[w_{00}]=W_{\mathrm{bulk}}.
```

Insert the particular solution and unit-gradient response into these
constraints. The two unknown gradients follow directly:

```math
G_x=\frac{U_{\mathrm{bulk}}-\mathcal B[U_b]-\mathcal B[u_*]}
          {\mathcal B[c]},\qquad
G_z=\frac{W_{\mathrm{bulk}}-\mathcal B[w_*]}{\mathcal B[c]}.
```

Substituting these values into ``u_*+G_xc`` and ``w_*+G_zc`` gives the
velocity satisfying both no slip and the required flux. There is no
iterative search for the pressure gradient. Each constraint adds one scalar
unknown gradient and one scalar bulk equation; the precomputed response
reduces their solution to the formulas above.

For Couette's reference ``U_b=y``, ``\mathcal B[U_b]=0``. For the default
Poiseuille reference ``U_b=1-y^2``, ``\mathcal B[U_b]=2/3``. Consequently,
requesting unit total bulk velocity with the latter reference requires
``\mathcal B[u_{00}]=1/3``, not 1. The gradients are determined separately
at every implicit stage. The initial velocity must also satisfy the target
flux; see [Equations and configuration](equations.md).

The integral is evaluated directly from ordinary Chebyshev coefficients.
For ``\phi=\sum_{m=0}^{P}a_mT_m``,

```math
\mathcal B[\phi]
 =a_0+\sum_{\substack{m=2\\m\ \mathrm{even}}}^{P}
       \frac{a_m}{1-m^2}.
```

Odd polynomials integrate to zero. Even coefficients beyond ``a_0`` therefore
contribute to the bulk velocity; the zeroth Chebyshev coefficient alone is
not its value.

### Mean pressure and its arbitrary constant

Since ``v_{00}=0``, normal momentum gives the first-order relation
``Dq_{00}=R_{y,00}``. Integrating it determines the pressure profile up to an
additive constant. Differentiating it would yield ``D^2q_{00}=DR_{y,00}``,
but solving that second-order equation introduces boundary and nullspace
issues that the original first-order equation avoids. The nonzero-mode
pressure influence construction is therefore not used for this mode.

The pressure constant is fixed by

```math
\mathcal B[q_{00}]=0.
```

This fixes the volume average of the periodic pressure, since all other
Fourier modes have zero plane average. It does not set both wall pressures
to zero and does not constrain the imposed uniform pressure gradient.
For the rotational form, this gauge applies to modified pressure:

```math
q_{00}=p_{00}+\tfrac12\langle|\boldsymbol v_{\mathrm{tot}}|^2\rangle_{xz}.
```

The kinetic-energy term includes fluctuations, not just the squared mean
velocity. Pressure and modified pressure therefore need not have the same
wall-normal profile or the same chosen additive constant.

At finite polynomial degree, let ``R_{y,00}=\sum_{m=0}^{P}r_mT_m`` and
``q_{00}=\sum_{m=0}^{P}q_mT_m``. A degree-``P`` pressure derivative can
represent only degrees through ``P-1``. Integrate the retained source with
``\widetilde r_m=r_m`` for ``0\le m\le P-1`` and
``\widetilde r_m=0`` for ``m\ge P``:

```math
q_1=\widetilde r_0-\frac{\widetilde r_2}{2},\qquad
q_m=\frac{\widetilde r_{m-1}-\widetilde r_{m+1}}{2m},
\quad m=2,\ldots,P.
```

The highest source coefficient ``r_P`` remains a normal-momentum tau
residual; integrating it would require a degree-``P+1`` pressure. Finally,
choose

```math
q_0=-\sum_{\substack{m=2\\m\ \mathrm{even}}}^{P}
          \frac{q_m}{1-m^2}
```

to enforce the gauge. Simply setting ``q_0=0`` would generally leave a
nonzero volume-mean pressure.

### The mean mode still interacts with the fluctuations

Decoupling Fourier modes applies only to the **implicit linear solve**.
The nonlinear source at ``(0,0)`` includes products of modes
``(k,l)`` and ``(-k,-l)``. It therefore carries momentum transport by
the fluctuations.

For example, define ``\overline U=\langle v_{\mathrm{tot},x}\rangle_{xz}``
and fluctuations relative to the instantaneous plane average. Averaging
streamwise momentum gives

```math
\partial_t\overline U
 =-\partial_y\langle u'v'\rangle_{xz}
  +\nu\partial_y^2\overline U-G_x+\langle f_x\rangle_{xz}.
```

Here the overbar denotes a plane average, not a time average. Even though
``\langle v'\rangle_{xz}=0``, the Reynolds shear stress
``\langle u'v'\rangle_{xz}`` need not vanish. The zero mode consequently
evolves with the turbulent stresses; it is neither removed nor held equal
to the prescribed base profile. A fixed-bulk constraint fixes its integral,
not its wall-normal shape.
