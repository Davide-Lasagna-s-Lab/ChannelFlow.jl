# Numerical method for Fourier–Fourier–finite-difference channel-flow simulation

**ChannelFlow.jl — Technical reference**

**Revision:** 15 September 2026

**Design update:** development now follows Channelflow's Chebyshev tau
formulation and its three-stage CNRK2 scheme. See
[Correspondence with Channelflow](channelflow-correspondence.md) for the pinned
source, implementation mapping and migration status. The report below retains
the earlier finite-difference/CN–Heun design; its wall-normal discretisation,
pressure closure and time-stepping sections do not specify the new target.

## Abstract

This report specifies a primitive-variable direct numerical simulation method
for incompressible plane channel and plane Couette flow. The streamwise and
spanwise directions are represented by Fourier expansions; the wall-normal
direction is discretised by banded finite-difference differentiation matrices.
The velocity is advanced as a perturbation about a stationary parallel
reference profile. Viscous terms are integrated with Crank–Nicolson and
nonlinear terms with the two-stage Heun method used by `Flows.CNRK2`.

The central computational problem is a constrained Stokes solve at each
temporal stage. Scalar Helmholtz inverses provide the velocity response to a
known pressure. An influence matrix determines the two unknown wall-pressure
amplitudes required to satisfy wall-normal continuity. Particular attention is
given to the finite-difference adaptation: independently assembled first- and
second-derivative matrices do not generally satisfy the identities used in the
continuous pressure-Poisson derivation. An exact discrete pressure elimination
is therefore formulated, and its reduction to a two-by-two boundary influence
system is derived.

The report describes the **complete numerical formulation**, including the
pressure closure and time-integration assembly. It serves as a mathematical
specification and user reference; it does not present numerical validation
results or performance measurements. The scalar Helmholtz API is documented
concretely. Algorithm listings for the complete solver are mathematical
pseudocode.

## Contents

1. [Scope and notation](#1-scope-and-notation)
2. [Governing equations and reference flow](#2-governing-equations-and-reference-flow)
3. [Spatial representation](#3-spatial-representation)
4. [Explicit momentum evaluation](#4-explicit-momentum-evaluation)
5. [The scalar Helmholtz solver](#5-the-scalar-helmholtz-solver)
6. [The continuous pressure influence construction](#6-the-continuous-pressure-influence-construction)
7. [A consistent finite-difference influence method](#7-a-consistent-finite-difference-influence-method)
8. [The mean Fourier mode and flow constraints](#8-the-mean-fourier-mode-and-flow-constraints)
9. [Crank–Nicolson–Heun time integration](#9-cranknicolsonheun-time-integration)
10. [Execution and cache organisation](#10-execution-and-cache-organisation)
11. [Accuracy, stability and verification criteria](#11-accuracy-stability-and-verification-criteria)
12. [Scalar-solver usage](#12-scalar-solver-usage)
13. [Sources and relation to Channelflow](#13-sources-and-relation-to-channelflow)

## 1. Scope and notation

The domain is

$$
\Omega=[0,L_x)\times[y_-,y_+]\times[0,L_z),
\qquad H=y_+-y_-.
\tag{1}
$$

The coordinates $x,y,z$ denote streamwise, wall-normal and spanwise position.
Velocity components always follow the physical order $(u,v,w)$.
Array dimensions follow the storage order $(y,x,z)$. These two orderings
must be distinguished when interpreting derivative operators.

The method assumes constant viscosity, periodicity in $x,z$, impermeable
walls, and stationary prescribed tangential wall velocities. Choosing a
reference profile that satisfies the wall velocities gives homogeneous
Dirichlet boundary conditions for the perturbation.

| Symbol | Meaning |
| :--- | :--- |
| $\boldsymbol U$ | Total velocity |
| $U_b(y)\boldsymbol e_x$ | Stationary reference velocity |
| $\boldsymbol q=(u,v,w)$ | Perturbation velocity |
| $\nu$ | Kinematic viscosity; $1/Re$ in nondimensional variables |
| $p$ | Instantaneous periodic pressure, or modified pressure when explicitly stated |
| $k_x,k_z$ | Integer Fourier mode numbers |
| $a=\alpha k_x,\ b=\beta k_z$ | Physical Fourier wave numbers |
| $\alpha=2\pi/L_x,\ \beta=2\pi/L_z$ | Fundamental wave numbers |
| $\kappa^2=a^2+b^2$ | Horizontal wave-number square |
| $D_1,D_2$ | Wall-normal first- and second-derivative matrices |
| $L_\kappa=D_2-\kappa^2I$ | Scalar discrete Laplacian before boundary replacement |
| $\sigma$ | Positive mass coefficient in a Stokes stage |
| $h=\Delta t,\ c=h/2$ | Time step and implicit stage coefficient |
| $\pi,\Phi$ | Elliptic-stage pressure and integrated pressure multiplier |
| $N_y,N_x,N_z$ | Resolved physical grid dimensions |
| $N_{xh}=\lfloor N_x/2\rfloor+1$ | Stored streamwise half-spectrum length |

A Fourier coefficient depends on $y$ and time. Hats are omitted in
mode-local derivations. The pair $(a,b)$ is fixed throughout each such
derivation.

## 2. Governing equations and reference flow

### 2.1 Total-velocity equations

Write the pressure as a periodic part plus a possible uniform streamwise
gradient. Defining $G(t)=-\partial_x\overline P(t)$, the momentum equation is

$$
\partial_t\boldsymbol U
=-(\boldsymbol U\cdot\nabla)\boldsymbol U
-\nabla p+\nu\Delta\boldsymbol U
+G(t)\boldsymbol e_x+\boldsymbol f,
\qquad
\nabla\cdot\boldsymbol U=0.
\tag{2}
$$

Here $\boldsymbol f$ is any additional body force per unit mass. Positive
$G$ accelerates the fluid in the positive streamwise direction. A spatially
uniform streamwise pressure gradient is represented by $G$, since the
linear function $-Gx$ is not periodic.

For dimensional calculations, $\nu$, lengths, velocities and forcing carry
their physical units. For nondimensional calculations, the length and velocity
used to define $Re$ must be stated with the simulation parameters.

### 2.2 Perturbation equations

Decompose the velocity as

$$
\boldsymbol U=U_b(y)\boldsymbol e_x+\boldsymbol q.
\tag{3}
$$

Since $U_b$ has no $x,z$ dependence, its convective self-advection vanishes.
Substitution gives

$$
\begin{aligned}
\partial_t\boldsymbol q={}&
-U_b\partial_x\boldsymbol q
-vU_b'\boldsymbol e_x
-(\boldsymbol q\cdot\nabla)\boldsymbol q
-\nabla p+\nu\Delta\boldsymbol q\\
&+\bigl(\nu U_b''+G(t)\bigr)\boldsymbol e_x+\boldsymbol f,
\qquad \nabla\cdot\boldsymbol q=0.
\end{aligned}
\tag{4}
$$

If the reference is sustained by a known $G_b$ with
$\nu U_b''+G_b=0$, the last streamwise forcing is
$\delta G(t)=G(t)-G_b$. This subtraction is necessary to avoid driving the
base flow twice.

For a supplied tabulated profile, the discrete residual
$\nu D_2U_b+G_b\boldsymbol1$ is the relevant quantity. It must either vanish
to the intended accuracy or be retained explicitly as a known forcing.
An arbitrary profile cannot be treated as a stationary solution merely
because it has been stored in `Grid`.

With the reference matching the wall velocities,

$$
u(y_\pm)=v(y_\pm)=w(y_\pm)=0.
\tag{5}
$$

### 2.3 Channel and Couette reference states

For stationary channel walls and constant $G_b$,

$$
U_b(y)=\frac{G_b}{2\nu}(y-y_-)(y_+-y).
\tag{6}
$$

For plane Couette flow with streamwise wall velocities $U_-,U_+$,

$$
U_b(y)=U_-+\frac{U_+-U_-}{H}(y-y_-),
\qquad G_b=0.
\tag{7}
$$

Their superposition describes combined pressure-driven and wall-driven flow.
The perturbation formulation and elliptic solver are identical for these
cases; the reference profile and mean-flow constraint distinguish them.

## 3. Spatial representation

### 3.1 Fourier expansion and storage

For a scalar field $q$,

$$
q(x,y,z,t)=
\sum_{k_x,k_z}
\widehat q_{k_x,k_z}(y,t)
\exp\!\left[i(\alpha k_xx+\beta k_zz)\right].
\tag{8}
$$

Real-valued physical fields satisfy

$$
\widehat q_{-k_x,-k_z}
=\overline{\widehat q_{k_x,k_z}}.
\tag{9}
$$

With `FFT_DIMS = (2, 3)`, the real transform is performed along $x$.
Only nonnegative $k_x$ are stored, while $z$ retains its complete complex
spectrum. The resolved physical and spectral shapes are therefore

$$
(N_y,N_x,N_z),\qquad
(N_y,N_{xh},N_z).
\tag{10}
$$

The field column `U[:, ix, iz]` contains an entire wall-normal Fourier
profile contiguously. Julia's first index is contiguous, which benefits
finite-difference multiplication and repeated wall-normal solves.

The code's integer mode convention is

$$
k_x=i_x-1,\qquad
k_z=
\begin{cases}
i_z-1,&i_z\le\lfloor N_z/2\rfloor+1,\\
i_z-1-N_z,&\text{otherwise}.
\end{cases}
\tag{11}
$$

For example, $N_z=7$ gives $0,1,2,3,-3,-2,-1$.
For even $N_z$, the exceptional Nyquist entry is assigned the positive
integer by this indexing convention; its coefficient is zeroed, so its
sign is immaterial to the calculation.

### 3.2 Differential operators

At a fixed mode,

$$
\partial_x\longrightarrow ia,\qquad
\partial_z\longrightarrow ib,\qquad
\partial_y\longrightarrow D_1,\qquad
\Delta\longrightarrow L_\kappa=D_2-\kappa^2I.
\tag{12}
$$

Accordingly, `ddx1!` differentiates in $x$, `ddx2!` in $y$, and
`ddx3!` in $z$, despite the array's $(y,x,z)$ storage order.
The full-grid discrete divergence is

$$
\mathcal D\boldsymbol q=ia\,u+D_1v+ib\,w.
\tag{13}
$$

The nodes $y_j$ and matrices $D_1,D_2$ are supplied when constructing
`Grid`. The Helmholtz backend requires $D_2$ to be an
`FDGrids.DiffMatrix`. Its entries are explicit finite-difference weights
stored in a compact banded representation; “compact” here describes storage,
not an implicit Padé differentiation equation.

For stencil nodes $\mathcal S_j$, the derivative weights satisfy polynomial
moment conditions such as

$$
\sum_{\ell\in\mathcal S_j}
(D_m)_{j\ell}(y_\ell-y_j)^r
=
\begin{cases}
m!,&r=m,\\
0,&r\ne m,
\end{cases}
\qquad r=0,\ldots,W-1,\quad m=1,2.
\tag{14}
$$

The stencil width $W$, node distribution and wall closures determine the
accuracy. On arbitrary nodes, $W$ points give the usual local estimate
$O(\delta y^{W-m})$ for a derivative of order $m$, subject to a regular
refinement family. Symmetry can improve the interior order. Boundary rows
and grid stretching can limit the global order; no single spatial order is
implied by the package name.

### 3.3 Padding, normalisation and Nyquist treatment

The padded periodic dimensions are

$$
N_x^p=\operatorname{oddceil}\!\left(\left\lceil3N_x/2\right\rceil\right),
\qquad
N_z^p=\operatorname{oddceil}\!\left(\left\lceil3N_z/2\right\rceil\right),
\tag{15}
$$

where $\operatorname{oddceil}(n)$ is the smallest odd integer at least $n$.
There is no padding in $y$. The four size queries are:

| Query | Shape |
| :--- | :--- |
| `physicalsize(grid, NotPadded())` | $(N_y,N_x,N_z)$ |
| `spectralsize(grid, NotPadded())` | $(N_y,\lfloor N_x/2\rfloor+1,N_z)$ |
| `physicalsize(grid, Padded())` | $(N_y,N_x^p,N_z^p)$ |
| `spectralsize(grid, Padded())` | $(N_y,\lfloor N_x^p/2\rfloor+1,N_z^p)$ |

The forward transform is divided by $N_x^pN_z^p$. Stored coefficients are
Fourier-series amplitudes; the unnormalised backward real transform then
reconstructs physical values without an additional scale factor.

During padding, the $k_x$ half-spectrum remains a prefix in dimension two.
The nonnegative $k_z$ block remains at the start of dimension three, while
the negative block is moved to its end. Truncation reverses this mapping.
For even resolved sizes, the $x$ and $z$ Nyquist planes are set to zero.

Quadratic products evaluated on the padded grid and truncated to the resolved
band avoid Fourier convolution aliasing in $x,z$. This operation does not
remove wall-normal finite-difference error or impose a discrete product rule.

## 4. Explicit momentum evaluation

Define the signed explicit acceleration

$$
\boldsymbol{\mathcal N}(t,\boldsymbol q)=
-\bigl[(U_b\boldsymbol e_x+\boldsymbol q)\cdot\nabla\bigr]
 (U_b\boldsymbol e_x+\boldsymbol q)
+\boldsymbol f
+\bigl(\nu D_2U_b+G(t)\boldsymbol1\bigr)\boldsymbol e_x,
\tag{16}
$$

or its equivalent reference-balanced form. Viscosity acting on the
perturbation remains in the implicit operator.

The reference is added to the streamwise **zero Fourier mode** before
differentiation and inverse transformation. Consequently both
$U_b\partial_x\boldsymbol q$ and $vU_b'\boldsymbol e_x$ are included.
Adding the reference only to the advecting physical velocity would omit
the second term.

The evaluation consists of spectral differentiation, padded inverse
transforms, physical products, forward transforms and truncation. The
convective and divergence forms represent, respectively,

$$
-\boldsymbol U\cdot\nabla\boldsymbol U,\qquad
-\nabla\cdot(\boldsymbol U\boldsymbol U).
\tag{17}
$$

Their equality uses incompressibility and the product rule. These identities
need not hold exactly for independently constructed finite-difference
operators.

The rotational form uses

$$
\boldsymbol U\times\boldsymbol\omega,\qquad
\boldsymbol\omega=\nabla\times\boldsymbol U,\qquad
P=p+\tfrac12|\boldsymbol U|^2.
\tag{18}
$$

The elliptic pressure is then the modified pressure $P$. The gradient
contained in the rotational contribution of the reference is absorbed by
this pressure. Pressure output must state which convention it uses.

The time-accuracy derivation in Section 9 assumes a fixed discrete explicit
map. Alternating the convective and divergence forms is a separate
algorithmic choice, discussed in Section 11.

## 5. The scalar Helmholtz solver

### 5.1 Operator and sign convention

The low-level scalar problem is

$$
\left[\theta_0(D_2-\kappa^2I)-\theta_1I\right]q=f,
\qquad q(y_-)=g_-,\quad q(y_+)=g_+.
\tag{19}
$$

For each stored mode, `FourierHelmoltzSolver.update!` passes

$$
\Theta_0=\theta_0,\qquad
\Theta_1=\theta_1+\theta_0\kappa^2
\tag{20}
$$

to the one-dimensional backend, whose operator is
$\Theta_0D_2-\Theta_1I$. This convention fixes both the horizontal
Laplacian contribution and the sign of the mass term.

The velocity operator in a Stokes stage is

$$
H_\kappa=\sigma I-\nu L_\kappa
=-\nu D_2+(\sigma+\nu\kappa^2)I.
\tag{21}
$$

It is obtained using `update!(solver, -ν, -σ)`. The pressure Poisson
operator $L_\kappa$ uses `update!(solver, 1, 0)`. A time-stage operator
$I-c\nu L_\kappa$ uses `update!(solver, -c*ν, -1)`.

| Intended operator | $\theta_0$ | $\theta_1$ |
| :--- | ---: | ---: |
| $D_2-\kappa^2I$ | $1$ | $0$ |
| $\sigma I-\nu(D_2-\kappa^2I)$ | $-\nu$ | $-\sigma$ |
| $I-c\nu(D_2-\kappa^2I)$ | $-c\nu$ | $-1$ |

The identity shifts in these expressions apply to interior equations.
The wall rows have their own boundary treatment.

### 5.2 Dirichlet row replacement

Let $A=\theta_0D_2-(\theta_1+\theta_0\kappa^2)I$.
The backend replaces its first and last rows by identity rows:

$$
\widetilde A_{1,:}=\boldsymbol e_1^T,\qquad
\widetilde A_{N_y,:}=\boldsymbol e_{N_y}^T,\qquad
\widetilde f_1=g_-,\quad \widetilde f_{N_y}=g_+.
\tag{22}
$$

All interior rows retain their finite-difference coefficients, including
couplings to the boundary unknowns. The system solves simultaneously for
the interior values and the prescribed wall values. Boundary forcing values
from the differential equation are not used in these two replaced rows.

To express the same solve with interior unknowns, let $R$ extract the
$m=N_y-2$ interior nodes, $E=R^T$ inject them with zero wall values, and
$E_B=[\boldsymbol e_1,\boldsymbol e_{N_y}]$ inject the two wall values.
Then

$$
q=Eq_I+E_Bg,\qquad
(RAE)q_I=Rf-RAE_Bg.
\tag{23}
$$

This reduced form is useful when deriving the pressure Schur complement.
It does not require a different wall-normal solver: applying the full
Dirichlet solver with zero wall values realises the inverse of $RAE$
on interior right-hand sides.

### 5.3 Cached banded factorisation

One `FDHelmoltzSolver.HelmoltzSolver` is stored for each pair
`(ix, iz)`. Each owns working matrix storage and an LU representation;
the reference $D_2$ is retained for rebuilding the operator.
`update!` assembles and factorises the matrix. Repeated `solve!`
calls perform only boundary substitution and triangular solution.

The factorisation uses the backend's no-pivot banded LU. For fixed bandwidth
$W$, its characteristic setup and solve costs are $O(N_yW^2)$ and
$O(N_yW)$, with $O(N_yW)$ storage per mode. These are operation-count
estimates, not measured timings.

Although the scalar operator has real coefficients, the current cache uses
complex working matrices to match spectral right-hand sides. Modes with
opposite $k_z$ have the same $\kappa^2$; the cache nevertheless stores
separate solver objects for each stored mode.

No-pivot LU requires acceptable pivots. High-order one-sided stencils and
strongly stretched grids do not automatically guarantee this property.
A successful factorisation must be judged together with the linear-system
residual; the continuum ellipticity of $H_\kappa$ alone is not a proof of
discrete numerical stability.

### 5.4 Right-hand-side handling and cache validity

`solve!(OUT, solver, F)` copies one contiguous wall-normal column of
`F` into the reusable complex buffer `rhs`, solves it in place, and
copies the result to `OUT`. The input is preserved when input and output
are distinct. Using the same field for both arguments gives an in-place
solve. One solver instance is serial because its work buffer is shared.

The optional `lower` and `upper` arguments are real scalar
**Fourier-coefficient boundary values applied to every visited mode**.
Their default zero values provide homogeneous velocity walls. Specifying
the same nonzero coefficient for every mode does not describe a spatially
uniform physical wall value; a uniform wall datum belongs only to the
zero mode. General complex mode-dependent boundary data require a
mode-local treatment in the surrounding Stokes algorithm.

Rebuild the factors when $\theta_0,\theta_1$, viscosity, wall-normal
matrices or domain lengths change. Changing resolution requires rebuilding
the cache structure. Mutating shared grid data does not automatically
invalidate cached factorizations. A constructor allocates the cache;
`update!` must precede the first solve.

## 6. The continuous pressure influence construction

### 6.1 The mode-local Stokes problem

For a nonzero Fourier mode, $\kappa>0$, every implicit stage has the form

$$
H_\kappa\boldsymbol q+\nabla_\kappa\pi=\boldsymbol r,
\qquad
\nabla_\kappa\cdot\boldsymbol q=0,
\qquad \boldsymbol q(y_\pm)=0,
\tag{24}
$$

where, in this section only,

$$
H_\kappa=\sigma-\nu(\partial_{yy}-\kappa^2),
\qquad
\nabla_\kappa=(ia,\partial_y,ib).
\tag{25}
$$

The pressure $\pi$ is an elliptic-stage multiplier. Its relationship to
physical pressure depends on the normalisation of the time-stage equations
and is established in Section 9.

Taking the divergence of (24), using commutation of the continuous
derivatives and the divergence constraint, gives

$$
(\partial_{yy}-\kappa^2)\pi
=ia\,r_x+\partial_y r_y+ib\,r_z.
\tag{26}
$$

The positive divergence on the right follows directly from the
$+\nabla_\kappa\pi$ convention in (24). The wall-normal momentum equation is

$$
H_\kappa v=r_y-\partial_y\pi.
\tag{27}
$$

At the walls, the tangential velocities are zero, so continuity becomes

$$
\partial_yv(y_-)=\partial_yv(y_+)=0.
\tag{28}
$$

Together, $v=0$ and $\partial_yv=0$ provide four wall conditions on the
coupled pressure–normal-velocity problem. Pressure wall values are auxiliary
unknowns determined by these conditions.

### 6.2 Particular and homogeneous solutions

First solve the particular pair

$$
\begin{aligned}
(\partial_{yy}-\kappa^2)\pi^0
  &=ia\,r_x+\partial_yr_y+ib\,r_z,
&\pi^0(y_-)&=\pi^0(y_+)=0,\\
H_\kappa v^0&=r_y-\partial_y\pi^0,
&v^0(y_-)&=v^0(y_+)=0.
\end{aligned}
\tag{29}
$$

The zero pressure endpoint values are a convenient auxiliary choice. They
are not imposed as physical pressure conditions on the final solution.

Construct two pressure responses $\psi^-,\psi^+$:

$$
\begin{aligned}
(\partial_{yy}-\kappa^2)\psi^-&=0,
&(\psi^-(y_-),\psi^-(y_+))&=(1,0),\\
(\partial_{yy}-\kappa^2)\psi^+&=0,
&(\psi^+(y_-),\psi^+(y_+))&=(0,1).
\end{aligned}
\tag{30}
$$

Their wall-normal velocity responses satisfy

$$
H_\kappa v^\pm=-\partial_y\psi^\pm,
\qquad v^\pm(y_-)=v^\pm(y_+)=0.
\tag{31}
$$

By linearity,

$$
\pi=\pi^0+c_-\psi^-+c_+\psi^+,\qquad
v=v^0+c_-v^-+c_+v^+.
\tag{32}
$$

The amplitudes $c_-,c_+$ are exactly the final pressure endpoint values in
this normalisation.

### 6.3 The two-by-two influence system

Define

$$
M_v=
\begin{bmatrix}
(v^-)'(y_-) & (v^+)'(y_-)\\
(v^-)'(y_+) & (v^+)'(y_+)
\end{bmatrix},
\qquad
d_B^0=
\begin{bmatrix}
(v^0)'(y_-)\\
(v^0)'(y_+)
\end{bmatrix}.
\tag{33}
$$

The wall continuity conditions give

$$
M_v
\begin{bmatrix}c_-\\c_+\end{bmatrix}
=-d_B^0.
\tag{34}
$$

Rows refer to the lower and upper wall; columns refer to the two unit
pressure responses. Every entry in (33) is a wall-normal **derivative**,
not a wall velocity, since the latter is already zero.

After recovering $\pi,v$, solve

$$
H_\kappa u=r_x-ia\pi,\qquad
H_\kappa w=r_z-ib\pi,
\qquad u(y_\pm)=w(y_\pm)=0.
\tag{35}
$$

The particular solution depends on the stage right-hand side. The homogeneous
responses and $M_v$ depend only on the grid, mode, viscosity and $\sigma$;
they can be cached.

### 6.4 Why the construction enforces continuity

Set $d=ia\,u+\partial_yv+ib\,w$.
Taking the divergence of the momentum equations and using (26) gives

$$
H_\kappa d=0,\qquad d(y_-)=d(y_+)=0.
\tag{36}
$$

For $\sigma>0,\nu>0$, multiply by $\overline d$, integrate, and use the
wall conditions:

$$
\sigma\int|d|^2\,dy
+\nu\int|\partial_yd|^2\,dy
+\nu\kappa^2\int|d|^2\,dy=0.
\tag{37}
$$

Thus $d=0$ throughout the channel. The argument establishes why only two
remaining wall constraints suffice **after the interior differential
identities are satisfied**.

The construction is the influence-method structure described in
[Gibson's manual, Section 4.5](https://download-mirror.savannah.gnu.org/releases/channelflow/channelflow_manual-0.9.13.pdf).
Equations (24)–(37) are derived here with a consistent sign convention and
lower-wall-first ordering. The finite-difference realisation requires the
additional discrete analysis below.

## 7. A consistent finite-difference influence method

### 7.1 Why replacing derivatives is insufficient

For independently assembled finite-difference matrices, generally

$$
D_1D_1\ne D_2,\qquad D_1D_2\ne D_2D_1.
\tag{38}
$$

Moreover, Dirichlet row replacement means that the momentum differential
equation is imposed at interior nodes, while boundary rows impose velocity
values. Differentiating the assembled system as though every row were a
momentum equation is therefore invalid.

The resulting issue can be written explicitly. Let $H_f=\sigma I-\nu L_\kappa$
on the full nodal space, $G_f\pi=(ia\pi,D_1\pi,ib\pi)$, and

$$
e=H_f\boldsymbol q+G_f\pi-\boldsymbol r,\qquad
d=ia\,u+D_1v+ib\,w.
$$

The vector $e$ includes any residual in the replaced wall momentum rows.
Direct algebra gives

$$
H_fd=
\mathcal D\boldsymbol r-L_\kappa\pi
+\nu(D_1D_2-D_2D_1)v
-(D_1^2-D_2)\pi+\mathcal D e.
\tag{39}
$$

A scalar pressure-Poisson solve removes only the first two terms where that
equation is enforced. Two wall amplitudes cannot generally eliminate all
remaining interior divergence sources. Small residuals in the scalar
Helmholtz solves alone do not establish a correct incompressible stage.

The algebraic reference construction below enforces the velocity–pressure
system directly.
It retains the particular-solution/two-boundary-response organisation, while
including the finite-difference consistency terms in the interior pressure
elimination. This is a finite-difference construction derived here; it is not
a transcription of a Chebyshev tau correction.

### 7.2 Interior velocity and full pressure spaces

Use the restriction and injection matrices $R,E$ introduced in (23).
Store the unknown velocity as

$$
\boldsymbol q=
\begin{bmatrix}E&0&0\\0&E&0\\0&0&E\end{bmatrix}\boldsymbol q_I,
\qquad \boldsymbol q_I\in\mathbb C^{3m},
\tag{40}
$$

so all velocity wall values are identically zero.

Keep the pressure at all $N_y$ nodes. Define

$$
\begin{aligned}
H_I&=R[\sigma I-\nu(D_2-\kappa^2I)]E,\\
A&=\operatorname{diag}(H_I,H_I,H_I),\\
G&=\begin{bmatrix}iaR\\RD_1\\ibR\end{bmatrix},
& C&=\begin{bmatrix}iaE&D_1E&ibE\end{bmatrix}.
\end{aligned}
\tag{41}
$$

Here $G$ maps nodal pressure to the three interior momentum equations.
$C$ evaluates divergence at **all** wall-normal nodes, including the walls.

For $\kappa>0$, the discrete stage is

$$
\begin{bmatrix}
A&G\\ C&0
\end{bmatrix}
\begin{bmatrix}
\boldsymbol q_I\\ \pi
\end{bmatrix}
=
\begin{bmatrix}
\boldsymbol r_I\\0
\end{bmatrix}.
\tag{42}
$$

This defines the reference discrete problem: interior momentum, all-node
continuity, and zero wall velocities. In particular, its wall continuity
rows are the chosen wall derivative rows of $D_1$ acting on $v$.
The same $D_1$ must be used for the pressure gradient, divergence,
influence matrix and diagnostic residual.

A nonzero Fourier pressure mode has no additive spatially constant gauge:
adding a constant wall-normal profile at nonzero $(a,b)$ changes the
tangential pressure gradient.

### 7.3 Pressure Schur complement

Eliminate velocity from the first row of (42):

$$
\boldsymbol q_I=A^{-1}(\boldsymbol r_I-G\pi).
\tag{43}
$$

Continuity becomes

$$
S\pi=d,\qquad
S=CA^{-1}G,\qquad
d=CA^{-1}\boldsymbol r_I.
\tag{44}
$$

The matrix $S$ is the pressure Schur complement: it measures the divergence
generated by the Helmholtz response to a pressure gradient. Its definition
includes both the finite-difference matrices and the actual velocity wall
treatment.

Applications of $A^{-1}$ are three scalar Dirichlet Helmholtz solves using
the cached factors. No explicit inverse of $A$ is formed.

### 7.4 Elimination of interior pressure

Reorder pressure columns and divergence rows into interior and boundary
groups. Let $B$ denote the two walls, ordered lower then upper:

$$
\begin{bmatrix}
S_{II}&S_{IB}\\
S_{BI}&S_{BB}
\end{bmatrix}
\begin{bmatrix}\pi_I\\\pi_B\end{bmatrix}
=
\begin{bmatrix}d_I\\d_B\end{bmatrix}.
\tag{45}
$$

Assuming $S_{II}$ is nonsingular, compute

$$
\pi_I^0=S_{II}^{-1}d_I,\qquad
X=-S_{II}^{-1}S_{IB},
\tag{46}
$$

and write

$$
\pi_I=\pi_I^0+X\pi_B.
\tag{47}
$$

Substituting into the boundary equations gives the exact two-by-two
influence system

$$
\boxed{
M\pi_B=d_B-S_{BI}\pi_I^0,\qquad
M=S_{BB}-S_{BI}S_{II}^{-1}S_{IB}.
}
\tag{48}
$$

The matrix $M$ is the Schur complement of the **interior pressure block**.
Its size is two because only two pressure endpoint values remain after
all interior continuity equations have been satisfied.

### 7.5 Particular and homogeneous discrete responses

The connection to the continuous influence construction is explicit.
Using $E$ and $E_B$ to express pressure in its original nodal order, define

$$
\pi^0=E\pi_I^0,
\qquad
\Psi=EX+E_B,
\tag{49}
$$

and

$$
\boldsymbol q_I^0=A^{-1}(\boldsymbol r_I-G\pi^0),
\qquad
V=-A^{-1}G\Psi.
\tag{50}
$$

The two columns of $\Psi$ have unit lower-wall and upper-wall pressure
traces. Their velocity responses are the columns of $V$. They satisfy

$$
C_I\boldsymbol q_I^0=0,\qquad C_IV=0.
\tag{51}
$$

Only the two wall divergences remain. Therefore

$$
(C_BV)\pi_B=-C_B\boldsymbol q_I^0,
\qquad C_BV=-M,
\tag{52}
$$

and reconstruction is

$$
\pi=\pi^0+\Psi\pi_B,\qquad
\boldsymbol q_I=\boldsymbol q_I^0+V\pi_B.
\tag{53}
$$

This is the discrete influence method in response form. Unlike a direct
substitution into (30), its homogeneous profiles solve the homogeneous
**discrete Stokes and interior-continuity equations**. They need not satisfy
the scalar equation $L_\kappa\psi=0$.

### 7.6 Admissibility and conditioning

The construction requires a nonsingular $H_I$, a valid interior elimination
$S_{II}$, and a nonsingular final influence matrix $M$. These are
properties of the selected derivative matrices and boundary closures.
Arbitrary configurable stencils do not guarantee them.

A singular $S_{II}$ may invalidate this particular pressure partition
even when another formulation is possible. A singular full pressure Schur
operator at a nonzero mode indicates an unresolved pressure/velocity
compatibility issue. Such a mode is not repaired by silently setting its
pressure to zero or introducing a numerical pressure gauge.

The lower and upper pressure responses can be scaled to improve the
conditioning of the two-by-two solve. Any scaling must be applied
consistently to the response columns and recovered amplitudes.
The influence equation is solved by a small factorisation rather than an
explicit matrix inverse.

As $\kappa$ approaches zero, the problem approaches the degeneracy of the
mean-pressure problem, including its gauge and separate boundary closure.
The exact zero mode is handled separately, as described in Section 8.

### 7.7 Cost of the discrete consistency closure

A two-by-two boundary solve does not imply that the entire pressure
calculation has constant cost. If $S_{II}$ is assembled explicitly, it is
generally dense: per-mode storage is $O(m^2)$, factorisation is $O(m^3)$,
and each subsequent interior pressure solve is $O(m^2)$.

The dense construction is a direct algebraic reference realisation.
A matrix-free pressure action evaluates $S p$ using pressure gradients,
three cached Helmholtz inverses and divergence; solving the corresponding
interior system then requires a suitable iterative method and tolerance.
A scalar pressure-Poisson inverse may be used as a preconditioner, but does
not replace the defining operator (44).

An equivalent sparse coupled realisation is also possible. These choices
change cost and storage, not equations (42)–(53). They must not be described
as having the cost of the classical scalar-Poisson construction unless the
additional interior work has actually been removed by a compatible
discretisation.

## 8. The mean Fourier mode and flow constraints

### 8.1 Velocity equations at $(k_x,k_z)=(0,0)$

At the mean mode, $\kappa=0$, and incompressibility reduces to
$\partial_yv_0=0$. With impermeable walls,

$$
v_0(y)=0.
\tag{54}
$$

The corresponding discrete requirement is that the wall-constrained
derivative has no nonzero null vector: $D_1Ev_I=0$ implies $v_I=0$.

The tangential equations are independent scalar Helmholtz problems:

$$
(\sigma-\nu\partial_{yy})u_0=r_{x,0}+\Gamma,\qquad
(\sigma-\nu\partial_{yy})w_0=r_{z,0}.
\tag{55}
$$

Here $\Gamma$ denotes a spatially uniform addition to the **normalised
stage right-hand side**. Its conversion to a physical pressure-gradient
forcing depends on the stage scaling below. The zero mode is not passed
through the nonzero-mode two-by-two influence solve.

### 8.2 Prescribed pressure gradient

When $G(t)$ is specified, its contribution is included in the explicit
stage data with the same time quadrature as the other known forcing.
Only the zero Fourier mode receives it. If a sustaining reference gradient
has already been removed, the perturbation receives $G-G_b$.

No bulk-flow equation is imposed in this case. The mean streamwise velocity
and bulk flow evolve according to the momentum equation.

### 8.3 Prescribed bulk flow

Let $w_j$ be wall-normal quadrature weights satisfying
$\sum_jw_j=H$, and define

$$
\mathcal Q(u_0)=\frac1H\sum_jw_j u_0(y_j),\qquad
U_{\mathrm{bulk}}=\mathcal Q(U_b)+\mathcal Q(u_0).
\tag{56}
$$

The quadrature is part of the flow-constraint definition and must match the
chosen wall-normal grid. Fixing the zero Fourier coefficient at a single
$y$ node would not fix bulk flow.

Solve one particular profile and cache one unit-forcing response:

$$
H_0u_0^0=r_{x,0},\qquad H_0h_G=\boldsymbol1,
\qquad u_0^0(y_\pm)=h_G(y_\pm)=0.
\tag{57}
$$

Then

$$
u_0=u_0^0+\Gamma h_G,\qquad
\Gamma=
\frac{U_{\mathrm{bulk}}^{\mathrm{target}}
-\mathcal Q(U_b)-\mathcal Q(u_0^0)}
{\mathcal Q(h_G)}.
\tag{58}
$$

This scalar response calculation imposes the bulk constraint to the accuracy
of the linear solves and quadrature. It requires
$\mathcal Q(h_G)\ne0$. The predictor and corrected velocity must both
satisfy the prescribed constraint before they are used to evaluate nonlinear
terms.

For the CNRK2 normalisation in Section 9, a forcing amplitude $\Gamma$
added to $\boldsymbol r=\boldsymbol b/c$ corresponds to $c\Gamma$
in the unnormalised stage right-hand side. Its effective acceleration over
the full step is $c\Gamma/h=\Gamma/2$. This value is a stage forcing
multiplier; it must not be reported as an instantaneous endpoint pressure
gradient without an appropriate reconstruction.

### 8.4 Mean pressure and its gauge

The mean normal momentum equation determines a pressure gradient,

$$
\partial_y\pi_0=r_{y,0},
\tag{59}
$$

while the pressure level remains arbitrary.

In the full nodal pressure representation, interior momentum supplies only

$$
RD_1\pi_0=Rr_{y,0}.
\tag{60}
$$

There are $N_y-2$ equations for $N_y$ pressure values. A single gauge
therefore leaves one additional discrete degree of freedom. This is a
boundary-closure issue, not a physical second pressure gauge.

A complete mean-pressure reconstruction retains all interior derivative
equations, one independent wall derivative equation, and a gauge. For a
lower-wall choice,

$$
\begin{bmatrix}
RD_1\\
\boldsymbol e_1^TD_1\\
w^T
\end{bmatrix}\pi_0
=
\begin{bmatrix}
Rr_{y,0}\\
r_{y,0}(y_-)\\
0
\end{bmatrix}.
\tag{61}
$$

The selected square matrix must be nonsingular. The final row fixes the
quadrature mean pressure to zero. The unused upper-wall gradient residual
provides an additional consistency diagnostic under refinement.
If this choice is rank deficient, the wall closure or pressure space must
be redesigned; an arbitrary pressure value is not a valid substitute.

Mean-pressure reconstruction does not alter the tangential velocity solves
or the bulk-flow correction.

## 9. Crank–Nicolson–Heun time integration

### 9.1 The evolution equation on the constrained velocity space

For a fixed spatial discretisation, write the divergence-free velocity
equation abstractly as

$$
\dot q=\mathcal A q+\mathcal F(t,q),
\tag{62}
$$

where $\mathcal A$ is the viscous operator restricted to admissible
incompressible velocities, and $\mathcal F$ is the explicit acceleration
after enforcing the pressure constraint.

Pressure is an algebraic multiplier. It is not an independently time-marched
state component, and should not participate in Runge–Kutta vector arithmetic
as if it had its own evolution equation.

The method used here is the **Crank–Nicolson/Heun predictor–corrector**
implemented by [`Flows.CNRK2`](https://github.com/gasagna/IMEXRK.jl/blob/b3c71e6cd683183b5ac18261c89082d4ad253f4a/src/steps/CNRK2.jl).
The name CNRK2 alone does not identify all coefficients; the equations below
define the scheme unambiguously.

### 9.2 Predictor and corrector

Let $h=\Delta t$, $c=h/2$, and

$$
B=I-c\mathcal A,\qquad C_t=I+c\mathcal A.
\tag{63}
$$

The explicit evaluations occur at $t_n$ and $t_n+h$:

$$
\begin{aligned}
F_n&=\mathcal F(t_n,q_n),\\
Bq_*&=C_tq_n+hF_n,\\
F_*&=\mathcal F(t_n+h,q_*),\\
Bq_{n+1}&=C_tq_n+\frac h2(F_n+F_*).
\end{aligned}
\tag{64}
$$

The predictor $q_*$ approximates the state at the **end** of the step.
It is not a midpoint stage. The corrector again starts from $C_tq_n$,
rather than advancing from $q_*$.

Both implicit equations have the same coefficient $c$, so they share
all Helmholtz and influence factors for a fixed step size.

| Operation | State/time used | Explicit weight | Implicit coefficient |
| :--- | :--- | :--- | :--- |
| First evaluation | $q_n,t_n$ | — | — |
| Predictor solve | $q_n,F_n$ | $hF_n$ | $c=h/2$ |
| Second evaluation | $q_*,t_n+h$ | — | — |
| Corrector solve | $q_n,F_n,F_*$ | $h(F_n+F_*)/2$ | $c=h/2$ |

There is no multistep start-up history: one admissible initial velocity is
sufficient.

### 9.3 Primitive-variable implementation of each stage

The projected fields in (64) need not be formed explicitly. Using the
unprojected viscous Laplacian and explicit momentum acceleration, form

$$
\begin{aligned}
\boldsymbol b_* &=
(I+c\nu L)\boldsymbol q_n
+h\boldsymbol{\mathcal N}(t_n,\boldsymbol q_n),\\
\boldsymbol b_{n+1} &=
(I+c\nu L)\boldsymbol q_n
+\frac h2\left[
\boldsymbol{\mathcal N}(t_n,\boldsymbol q_n)
+\boldsymbol{\mathcal N}(t_n+h,\boldsymbol q_*)\right].
\end{aligned}
\tag{65}
$$

For $s=*$ and $s=n+1$, solve

$$
(I-c\nu L)\boldsymbol q_s+G_f\Phi_s=\boldsymbol b_s,
\qquad \mathcal D\boldsymbol q_s=0,\qquad
\boldsymbol q_s(y_\pm)=0.
\tag{66}
$$

The nonzero-mode realisation of (66) is the discrete system (42),
with

$$
\sigma=\frac1c=\frac2h,\qquad
\boldsymbol r_s=\frac{\boldsymbol b_s}{c},\qquad
\pi_s=\frac{\Phi_s}{c}.
\tag{67}
$$

Thus the normalised Helmholtz cache uses
`update!(solver, -ν, -2/h)`. Alternatively one may factor the unnormalised
operator using `update!(solver, -h*ν/2, -1)`, provided every pressure
response and right-hand side uses the same scaling.

Changing only the Helmholtz coefficient while retaining an influence cache
built with a different pressure scaling produces incorrect amplitudes.
A cache must have one documented normalisation.

### 9.4 Why the constrained solve is the correct implicit operation

On the compatible interior spaces, and with mean/gauge issues treated
separately, define

$$
P=I-G(CG)^{-1}C.
\tag{68}
$$

This expression requires $CG$ to be nonsingular on the selected nonzero-mode
spaces. That is an additional regularity condition: invertibility of
$CA^{-1}G$ at a particular finite $\sigma$ does not establish it by itself.
Then $Pq=q$ for divergence-free $q$, and $PG=0$.
Applying $P$ to the stage equation shows that (66) implements (64) with

$$
\mathcal A=P\nu L,\qquad
\mathcal F=P\boldsymbol{\mathcal N}.
\tag{69}
$$

Consequently an explicit gradient added to the raw right-hand side changes
the pressure multiplier while leaving the stage velocity unchanged.

Equations (68)–(69) describe homogeneous incompressibility constraints.
For prescribed bulk flow, a nonzero target perturbation flux defines an
affine admissible set. To recover a strictly linear projected-operator
interpretation, choose a fixed, wall-compatible, divergence-free lifting
$q_{\mathrm{ref}}$ with the target perturbation flux, evolve
$z=q-q_{\mathrm{ref}}$ with zero flux, and include
$\nu Lq_{\mathrm{ref}}$ in the known forcing. The projection must then
enforce both divergence and zero flux, using the uniform streamwise forcing
as the additional constraint multiplier. If the base profile already has
the target bulk flow, the extra lifting is zero, but the zero-flux constraint
is still required. A dedicated stage solver can instead enforce the affine
target directly through (58).

The exact constrained inverse is not generally obtained by independently
inverting scalar Helmholtz equations and then applying the ordinary
projection $P$. Wall conditions make these operations noncommuting.
The influence/Schur solve belongs **inside** each implicit stage.

### 9.5 Meaning of stage pressure

Dividing (66) by $h$ after subtracting $\boldsymbol q_n$ gives

$$
\frac{\boldsymbol q_s-\boldsymbol q_n}{h}
=
\frac\nu2 L(\boldsymbol q_s+\boldsymbol q_n)
+\boldsymbol{\mathcal N}_s
-G_f\frac{\Phi_s}{h},
\tag{70}
$$

where $\boldsymbol{\mathcal N}_*$ uses the first explicit evaluation and
the corrected stage uses their average.

Hence $\Phi_s/h$ is the effective pressure in that time-discrete momentum
balance, while

$$
\pi_s=\frac{\Phi_s}{c}=2\frac{\Phi_s}{h}.
\tag{71}
$$

The elliptic multiplier $\pi_s$ is not, by definition, pressure at
$t_n+h$. A scheme that explicitly trapezoidally averages endpoint
pressures would involve their sum in this normalisation, but (64) does
not separately identify those two endpoint values.

For instantaneous pressure output, use the converged velocity and the
instantaneous explicit acceleration. Differentiating the discrete
continuity constraint gives

$$
CG\,p=C\bigl(\nu Lq+\boldsymbol{\mathcal N}(t,q)\bigr),
\tag{72}
$$

with the same velocity/pressure spaces, the nonsingularity condition on $CG$,
and the separate mean-mode closure. This is a
separate pressure reconstruction. Its numerical accuracy must be assessed
separately from the velocity order. Rotational-form calculations additionally
use the modified-pressure convention in (18).

### 9.6 Optional Flows.jl interface

The implicit interface is defined by

$$
\operatorname{ImcA!}(\mathcal A,c,Y,OUT):
\quad (I-c\mathcal A)\,OUT=Y.
\tag{73}
$$

The companion multiplication evaluates $(I-c\mathcal A)Y$.
The predictor's multiplication call uses $-h/2$, producing
$I+h\mathcal A/2$; both inverse calls use $+h/2$.

A primitive-variable adapter can provide raw viscous multiplication and
raw explicit momentum, with `ImcA!` implementing the full constrained
Stokes solve. Equation (69) establishes its equivalence to the projected
velocity method for admissible input states.

The state passed to `Flows.CNRK2` contains the perturbation velocity.
Pressure, mean-flow multipliers, elliptic factors and temporary fields are
owned by the flow operator or its caches. The five state-sized work arrays
allocated by the CNRK2 implementation support the predictor and corrector;
optional stage storage additionally retains copies of the two states used
for nonlinear evaluations.

### 9.7 Fixed step size and final-step handling

At a fixed $h$, both temporal stages and all time steps reuse the same
elliptic factors and homogeneous responses. A change in the actual step
requires updating the caches that depend on $\sigma=2/h$.

The `Flows` endpoint iterator can shorten the final step to reach the
requested final time. Therefore a nominally constant-step configuration
does not by itself guarantee that every step has the same length.
Either choose an interval containing an integer number of steps or compare
the actual coefficient supplied to `ImcA!` with the cached coefficient.

## 10. Execution and cache organisation

### 10.1 Setup phase

For a chosen grid, viscosity and time step:

1. Construct $D_1,D_2$, the reference profile, and wall-normal quadrature.
2. Allocate resolved spectral fields and padded physical workspaces.
3. Create forward and backward FFT plans with dimensions $(2,3)$.
4. Factor the velocity Helmholtz operator for every stored Fourier mode.
5. For each nonzero retained mode, construct the interior pressure
   elimination, the two homogeneous pressure/velocity responses and the
   two-by-two influence factor.
6. Prepare the separate mean-mode solver, pressure reconstruction and, when
   needed, the unit bulk-forcing response.
7. Initialise a real-compatible, divergence-free perturbation satisfying
   homogeneous wall values, with zero Nyquist planes.

The initial incompressibility constraint is part of the initial condition.
A pressure solve at the first time step does not replace a well-defined
initialisation procedure.

### 10.2 One constrained Stokes solve

The following pseudocode uses the exact finite-difference response
construction. It describes the mathematics rather than additional exported
Julia functions.

~~~text
stokes_stage(r, cache):
    for each retained nonzero Fourier mode, in FFT storage order:
        q_free = A^{-1} r_I
        d = C q_free
        p_I_particular = solve(S_II_factor, d_I)

        p_particular = interior pressure plus zero pressure endpoints
        q_particular = A^{-1}(r_I - G p_particular)

        wall_residual = C_B q_particular
        wall_amplitudes = solve(velocity_influence_factor, -wall_residual)

        p = p_particular + pressure_responses * wall_amplitudes
        q_I = q_particular + velocity_responses * wall_amplitudes
        insert q_I into full profiles with zero wall values

    solve the mean tangential Helmholtz problems
    impose the selected pressure-gradient or bulk-flow condition
    set the mean wall-normal velocity to zero
    reconstruct mean pressure with its derivative closure and gauge
    enforce the excluded Nyquist planes
    return velocity and the stage pressure multiplier
~~~

The two equivalent influence sign conventions must not be mixed:
`velocity_influence_factor` represents $C_BV=-M$, and its right-hand
side is $-C_Bq_I^0$. Using the pressure Schur matrix $M$ instead requires
the right-hand side in (48).

### 10.3 One complete time step

A dedicated `step!` routine can express the entire CN/Heun algorithm
directly. The numerical method does not require a general time-integration
framework.

~~~text
step(q_n, t_n, h):
    refresh coefficient-dependent caches if the actual h has changed

    N_n = explicit_momentum(t_n, q_n)
    base = (I + h*nu*L/2) q_n

    b_predictor = base + h*N_n
    q_predictor, Phi_predictor = constrained_stage(b_predictor, h/2)

    N_predictor = explicit_momentum(t_n + h, q_predictor)
    b_corrector = base + h*(N_n + N_predictor)/2
    q_next, Phi_corrector = constrained_stage(b_corrector, h/2)

    return q_next
~~~

Both `constrained_stage` calls include pressure, wall continuity and the
mean-flow constraint. A `Flows.jl` adapter executes the same mathematical
operations through its callback interface; a dedicated stepper can make
these operations explicit and choose its own workspace layout.

### 10.4 Cache contents and dependencies

| Cached quantity | Depends on | Reused for |
| :--- | :--- | :--- |
| FFT plans and padding buffers | Grid dimensions, scalar type, planning configuration | All nonlinear evaluations |
| Velocity Helmholtz LU factors | $D_2,\kappa^2,\nu,\sigma$ | Particular and homogeneous velocity solves |
| Scalar Poisson factors, if used | $D_2,\kappa^2$, pressure boundary convention | Continuous-form reference or pressure preconditioning |
| Interior pressure factor | $D_1,D_2$, velocity walls, $\kappa^2,\nu,\sigma$ | All right-hand sides at that stage coefficient |
| Homogeneous responses and influence factor | Same discrete Stokes operator and pressure scaling | Both CN/Heun stages and subsequent steps |
| Mean unit-forcing response | $D_2,\nu,\sigma$ | Bulk-flow correction |
| Mean-pressure reconstruction factor | $D_1$, derivative-row selection, gauge | Mean pressure output |
| Nonlinear and stage workspaces | Field shapes, scalar type, selected nonlinear form | All stages |

The scalar Helmholtz right-hand-side buffer can be reused serially. Cached
response profiles are read-only during a stage solve. A future parallel
implementation would need independent mutable workspaces for simultaneous
mode solves.

For $N_m=N_{xh}N_z$, scalar Helmholtz storage scales as
$O(N_mN_yW)$. FFT transforms scale approximately as
$O(N_yN_x^pN_z^p\log(N_x^pN_z^p))$.
The cost of the interior pressure closure in Section 7 must be added;
the two-by-two boundary solve is negligible by comparison.

## 11. Accuracy, stability and verification criteria

### 11.1 Temporal order

For a fixed spatial system with a smooth, fixed explicit map, the predictor
satisfies $q_*=q_n+h(\mathcal A q_n+\mathcal F_n)+O(h^2)$.
Substituting this expansion into the corrector yields the second-order
Taylor expansion of (62). Consequently the local truncation error is
$O(h^3)$, and the global velocity error is $O(h^2)$ over a fixed time
interval under the usual stability and regularity assumptions.

This statement assumes that both constrained stages are solved consistently,
that forcing is evaluated at the specified stage times, and that the initial
condition satisfies the constraints. It is an order statement for velocity;
it does not establish endpoint-pressure order.

### 11.2 Linear stability of the split method

For the scalar split equation

$$
\dot q=(\lambda_I+\lambda_E)q,\qquad
z_I=h\lambda_I,\quad z_E=h\lambda_E,
$$

eliminating the predictor in (64) gives

$$
R(z_I,z_E)=
\frac{1-z_I^2/4+z_E+z_E^2/2}{(1-z_I/2)^2}.
\tag{74}
$$

With $z_E=0$, this reduces to the Crank–Nicolson factor

$$
R(z_I,0)=\frac{1+z_I/2}{1-z_I/2}.
\tag{75}
$$

It is A-stable but not L-stable: strong diffusive modes tend to a sign-changing
amplification factor of $-1$, rather than being damped to zero in one step.

With $z_I=0$, the method is explicit Heun,

$$
R(0,z_E)=1+z_E+\tfrac12z_E^2.
\tag{76}
$$

For a purely imaginary explicit eigenvalue, $z_E=i\eta$,

$$
|R(0,i\eta)|^2=1+\frac{\eta^4}{4}>1
\quad\text{for }\eta\ne0.
\tag{77}
$$

Thus this explicit two-stage method has no nonzero stability interval on
the imaginary axis. Implicit viscosity can stabilise a combined
advection–diffusion discretisation, but does not justify a universal
advective CFL constant or unconditional Navier–Stokes stability.

A practical step-size assessment considers total advection, including
$U_b$, the local wall-normal spacing, spectral wave numbers and the actual
viscous damping. Strongly nonnormal dynamics and nonlinear interactions
require more than the scalar analysis above.

### 11.3 Alternating nonlinear forms

`AlternatingForm` switches the discrete nonlinear form on every
evaluation. With two CNRK2 evaluations per step, the usual phase is divergence
form at $t_n$, then convective form at $t_n+h$.

Let these two maps be $N_D,N_C$. At fixed spatial resolution they need not
be identical. The leading explicit contribution in the corrector is then
$\tfrac12(N_D+N_C)$, while the predictor uses only $N_D$.
The standard Heun proof for a single map does not establish second-order
accuracy for this composition.

Temporal convergence of the alternating variant must therefore be assessed
separately. Diagnostic calls to the mutable alternating operator also change
its phase. A fixed convective or rotational map provides an unambiguous
baseline for verifying the time integrator.

### 11.4 Discrete residuals

For a computed nonzero-mode stage, evaluate at least

$$
\begin{aligned}
r_{\mathrm{mom}}&=Aq_I+G\pi-r_I,\\
r_{\mathrm{div}}&=Cq_I,\\
r_{\mathrm{wall}}&=
(u(y_-),v(y_-),w(y_-),u(y_+),v(y_+),w(y_+)).
\end{aligned}
\tag{78}
$$

A useful scaled momentum residual is

$$
\epsilon_{\mathrm{mom}}=
\frac{\|r_{\mathrm{mom}}\|}
{\|A\|\|q_I\|+\|G\|\|\pi\|+\|r_I\|+\varepsilon_{\mathrm{scale}}},
\tag{79}
$$

where $\varepsilon_{\mathrm{scale}}$ prevents division by zero at a
vanishing state. Divergence should be scaled by a representative inverse
length times a velocity norm. The scalar Helmholtz residual, wall divergence,
interior divergence and bulk-flow residual answer different questions and
should not be collapsed into a single convergence number.

The pressure equation residual alone does not validate incompressibility,
as (39) demonstrates.

### 11.5 Verification problems

The following checks define a verification programme; no measured errors or
passed test results are claimed here.

1. **Fourier reconstruction and derivatives.** Use single modes and mixed
   modes with positive and negative $k_z$, including odd and even periodic
   grid sizes. Check normalisation, conjugacy, and Nyquist exclusion.
2. **Scalar Helmholtz inversion.** Choose a smooth profile with known
   derivatives and wall values. Form the forcing using the desired operator,
   then compare the solve with both the discrete manufactured vector and
   the continuous profile under refinement.
3. **Constrained Stokes inversion.** Construct a divergence-free velocity
   satisfying all wall conditions and an analytic pressure, then manufacture
   the momentum right-hand side. Examine all residuals in (78).
4. **Discrete compatibility.** Compare a direct solve of (42) with the
   influence elimination (45)–(53) at a small resolution. This checks the
   algorithmic reduction without assuming a continuum derivative identity.
5. **Reference equilibrium.** Zero perturbation must remain zero for a
   correctly balanced channel or Couette reference, up to numerical error.
   The rotational pressure gradient must be absorbed consistently.
6. **Temporal convergence.** At fixed fine spatial resolution, compare
   $h,h/2,h/4$ using a fixed nonlinear form and a smooth transient. A
   second-order velocity error gives an asymptotic ratio of about four.
7. **Mean-flow constraints.** Verify both prescribed pressure gradient and
   prescribed bulk flow, including predictor constraints and the
   normalisation of the recovered forcing multiplier.
8. **Pressure output.** Compare independently reconstructed instantaneous
   pressure with a manufactured solution, after imposing the same gauge.
9. **Long-time flow statistics.** Only after the preceding checks, compare
   appropriate mean velocity, stresses, wall shear and balances for
   established channel or Couette benchmarks.

Energy and dissipation diagnostics must use a consistent wall-normal
quadrature and the Fourier conjugacy weights of the stored half-spectrum.
A centered finite-difference stencil or an influence correction alone does
not guarantee a discrete kinetic-energy identity.

## 12. Scalar-solver usage

The following example illustrates the existing scalar Helmholtz interface.
It assumes that the package dependencies are available in the Julia
environment.

~~~julia
using ChannelFlow
import FDGrids

Ny, Nx, Nz = 65, 32, 32
y = collect(range(-1.0, 1.0; length=Ny))
D1 = FDGrids.DiffMatrix(y, 7, 1)
D2 = FDGrids.DiffMatrix(y, 7, 2)
Ub = 1 .- y.^2

grid = Grid(D1, D2, y, Nx, Nz, (2π, π), Ub)
solver = FourierHelmoltzSolver(grid)

nu = 1 / 1000
dt = 0.01
sigma = 2 / dt

# H = sigma*I - nu*(D2 - kappa^2*I), with Dirichlet walls.
ChannelFlow.update!(solver, -nu, -sigma)

F = SpectralField(zeros(ComplexF64, spectralsize(grid, NotPadded())), grid)
OUT = similar(F)

# Populate the interior Fourier coefficients of F with the desired RHS.
ChannelFlow.solve!(OUT, solver, F)
~~~

Calling `solve!` for each of three velocity components with arbitrary
right-hand sides does not, by itself, impose incompressibility.
The pressure-gradient correction and influence construction provide that
coupling. The example is a scalar API illustration, not a complete DNS run
or a claim of validation.

The source spelling `Helmoltz` is retained in type and package names.
The mathematical operator is referred to as Helmholtz throughout the text.

## 13. Sources and relation to Channelflow

The numerical method combines three distinct ingredients: Fourier
mode-by-mode Stokes reduction in the style of Channelflow, a banded
finite-difference wall-normal inverse, and the CN/Heun temporal scheme.
Equations are derived in this report's notation; the full discrete
pressure elimination in Section 7 specifies the finite-difference
adaptation.

### 13.1 External methodological references

1. **John F. Gibson, _Channelflow Users' Manual_, 25 May 2005.**
   [Original PDF](https://download-mirror.savannah.gnu.org/releases/channelflow/channelflow_manual-0.9.13.pdf).
   The cover identifies release 0.9.8, despite the download filename.
   Sections 4.4–4.6 give time discretisation, influence matrices and scalar
   Helmholtz problems. The wall-normal discretisation is Chebyshev.
   Its time schemes are CNAB2 and a three-stage Runge–Kutta scheme;
   their coefficients are not the CN/Heun coefficients in (64).

2. **Channelflow project, `tausolver.cpp`.**
   [Official source](https://github.com/epfl-ecps/channelflow/blob/master/channelflow/tausolver.cpp).
   This source provides the pressure/normal-velocity influence implementation
   and Chebyshev tau corrections. The link follows the upstream branch.
   The finite-difference closure in Section 7 is derived independently.

The old manual contains typographical inconsistencies: under its momentum
sign convention, the pressure right-hand side follows from positive
divergence after rearrangement; the particular pressure wall values in the
influence construction are Dirichlet values; and both rows of the
influence equation involve wall-normal velocity derivatives.
For this report, equations (24)–(34) fix these conventions explicitly.
The manual's tau-correction subsection is unfinished, so it should not be
read as a complete discrete algorithm by itself.

### 13.2 Code references

- [`grids.jl`](../src/grids.jl): physical/spectral sizes, padding,
  reference profile and wall-normal matrices.
- [`ffts.jl`](../src/ffts.jl) and
  [`indexing.jl`](../src/indexing.jl): Fourier storage, normalisation,
  signed modes and Nyquist treatment.
- [`operators.jl`](../src/operators.jl) and
  [`explicit.jl`](../src/explicit.jl): derivatives and nonlinear forms.
- [`helmoltz.jl`](../src/helmoltz.jl): the scalar Fourier Helmholtz cache,
  coefficient update and serial solve API.
- **FDHelmoltzSolver.jl**, source snapshot
  `c706256daa92f63f5b5be2e76b50c4bfbd09dacb`:
  [scalar Helmholtz implementation](https://github.com/Davide-Lasagna-s-Lab/FDHelmoltzSolver.jl/blob/c706256daa92f63f5b5be2e76b50c4bfbd09dacb/src/helmoltz.jl).
- **FDGrids.jl**, source snapshot
  `e1ee1949db788e9ab9f447064d2f7c63f452e6af`:
  [differentiation matrices](https://github.com/Davide-Lasagna-s-Lab/FDGrids.jl/blob/e1ee1949db788e9ab9f447064d2f7c63f452e6af/src/diffmatrix.jl)
  and [banded linear algebra](https://github.com/Davide-Lasagna-s-Lab/FDGrids.jl/blob/e1ee1949db788e9ab9f447064d2f7c63f452e6af/src/linalg.jl).
- **Local Flows.jl checkout**, source snapshot
  `b3c71e6cd683183b5ac18261c89082d4ad253f4a`, whose configured remote
  repository is `gasagna/IMEXRK.jl`:
  [CNRK2 stages](https://github.com/gasagna/IMEXRK.jl/blob/b3c71e6cd683183b5ac18261c89082d4ad253f4a/src/steps/CNRK2.jl),
  [implicit interface](https://github.com/gasagna/IMEXRK.jl/blob/b3c71e6cd683183b5ac18261c89082d4ad253f4a/src/imca.jl),
  and [endpoint step handling](https://github.com/gasagna/IMEXRK.jl/blob/b3c71e6cd683183b5ac18261c89082d4ad253f4a/src/stepper.jl).

The ChannelFlow scalar API and storage conventions were checked against
source revision `2b0feb513489a816a4223fb549f0ff13cd6e62b8`.
The complete stage equations in this report define the numerical
specification, independently of whether their orchestration uses a
dedicated stepper or a general integration library.
