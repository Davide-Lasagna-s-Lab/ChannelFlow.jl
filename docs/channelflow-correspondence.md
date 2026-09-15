# Correspondence with Channelflow

The Chebyshev implementation follows the numerical decomposition in
[epfl-ecps/channelflow, commit `ad37ef3022351d4e4a7a6c274c59a88605ad19e8`](https://github.com/epfl-ecps/channelflow/tree/ad37ef3022351d4e4a7a6c274c59a88605ad19e8).
This pinned revision is the source reference for the migration from finite
differences. The table distinguishes existing building blocks from planned
integration; it is not a claim that the complete DNS solver is operational.

| Channelflow source | Julia counterpart | Status |
| --- | --- | --- |
| [`ChebyCoeff`, `chebyshev.cpp`](https://github.com/epfl-ecps/channelflow/blob/ad37ef3022351d4e4a7a6c274c59a88605ad19e8/channelflow/chebyshev.cpp) | Backend `ChebCoeffs`; coefficient derivatives and transforms | Coefficient container exists; full derivatives and transforms remain to be connected. |
| [`BandedTridiag`, `bandedtridiag.cpp`](https://github.com/epfl-ecps/channelflow/blob/ad37ef3022351d4e4a7a6c274c59a88605ad19e8/channelflow/bandedtridiag.cpp) | Backend `QuasiTridiagonal`, `ul!`, `ldiv!` | Same matrix structure and UL approach, with a different internal factor normalization. |
| [`HelmholtzSolver`, `helmholtz.cpp`](https://github.com/epfl-ecps/channelflow/blob/ad37ef3022351d4e4a7a6c274c59a88605ad19e8/channelflow/helmholtz.cpp) | Backend `HelmoltzSolver`, `update!`, `solve!` | Scalar tau equations already present; both coefficient-count parities are now supported. |
| [`FlowField`, `flowfield.cpp`](https://github.com/epfl-ecps/channelflow/blob/ad37ef3022351d4e4a7a6c274c59a88605ad19e8/channelflow/flowfield.cpp) | `PhysicalField`, `SpectralField`, transform caches | Fourier transforms exist; the Chebyshev representation is still pending. |
| [`TauSolver`, `tausolver.cpp`](https://github.com/epfl-ecps/channelflow/blob/ad37ef3022351d4e4a7a6c274c59a88605ad19e8/channelflow/tausolver.cpp) | A primitive-variable solver for each Fourier mode | Pending: pressure/velocity solves, influence correction, tau correction and mean mode. |
| [`NSE`, `nse.cpp`](https://github.com/epfl-ecps/channelflow/blob/ad37ef3022351d4e4a7a6c274c59a88605ad19e8/channelflow/nse.cpp) | Nonlinear operator and channel-flow assembly | Nonlinear forms exist; constrained linear evaluation and stage solves remain to be assembled. |
| [`RungeKuttaDNS`, `dnsalgo.cpp`](https://github.com/epfl-ecps/channelflow/blob/ad37ef3022351d4e4a7a6c274c59a88605ad19e8/channelflow/dnsalgo.cpp#L351-L456) | Fixed-step CNRK2 integrator | Pending; use Gibson's three-stage coefficients below. |

## Representation and scalar solver

The target spectral field stores ordinary expansion coefficients

$$
\widehat u(y;k_x,k_z)=\sum_{n=0}^{P}u_n(k_x,k_z)T_n(\eta),
\qquad \eta=\frac{2y-a-b}{b-a},\qquad N_y=P+1.
$$

Its array order is `(n, kx, kz)`, with coefficient `n` at Julia index `n+1`.
Physical arrays retain `(y, x, z)`. Channelflow's Lobatto nodes run from
`b` to `a`, with `η_j = cos(π*j/P)`. Forward DCT-I normalization is division
by `P`, followed by halving coefficients `0` and `P`; the inverse doubles
those endpoints and applies half the DCT-I. These are the conventions of
`ChebyCoeff::chebyfft` and `ichebyfft` in the linked source.

Wall-normal differentiation will act on coefficients through Chebyshev
recurrences. The base profile added to the zero Fourier mode must use the
same coefficient convention. This requires a coordinated change to the
grid, operators and transforms: the current `SpectralField` still holds
Fourier coefficients sampled at wall-normal nodes. The current
`FourierHelmoltzSolver` consequently remains the FD wrapper until that change.

For the scalar equation `ν*u_yy - λ*u = f`, the existing backend API maps to
Gibson as follows:

| Quantity | Backend | Channelflow |
| --- | --- | --- |
| Polynomial degree | `P` | `N_` |
| Coefficient count | `P + 1` | `numberModes` |
| Second-derivative coefficient on `[-1,1]` | `θ₀ = ν*(2/(b-a))^2` | `nuscaled` |
| Zeroth-order coefficient | `θ₁ = λ` | `lambda` |
| Even/odd factored matrices | `Be`, `Bo` | `Ae_`, `Ao_` |
| Integrated RHS operator | Stored coefficients `l`, `-d`, `u` | Matrices `Be_`, `Bo_` |
| Boundary argument order | `solve!(h, f, u₊, u₋)` | `solve(u, f, ua, ub)`; lower then upper |

Gibson requires odd `numberModes`, giving one more even coefficient than odd
coefficients. The Julia backend now admits this case while retaining its
previous support for even coefficient counts. The highest two residual
coefficients are tau terms: the scalar differential equation is enforced
through degree `P-2`, with two wall equations completing the system.

For Fourier modes, the planned cache will reuse real scalar factors for the
real and imaginary parts, as `TauSolver` does. The existing
`CoupledHelmoltzSolver` is a factored fourth-order scalar solver; its second
equation is driven by the first solution itself. Primitive pressure/velocity
coupling instead uses the **derivative of pressure** and needs its own solver.

## Pressure and influence correction

Follow `TauSolver` in the linked source in this order:

1. Construct the pressure and velocity Helmholtz solvers for the mode.
2. Cache the two homogeneous pressure responses and their wall-normal
   velocity responses, driven by the pressure derivatives.
3. Form the two-by-two influence matrix from wall derivatives of those
   velocity responses.
4. Solve the particular pressure and wall-normal velocity problems, then
   apply `influenceCorrection` to enforce both wall derivatives of velocity.
5. Apply the tau correction from `solve_P_and_v`, including the cached
   auxiliary pair `P_0_`, `v_0_`. This is an additional discrete correction,
   not another name for the wall influence correction.
6. Recover the tangential velocities and treat the mean Fourier mode and
   its pressure-gradient or bulk-flow constraint separately.

## Time stepping

Gibson's `CNRK2` is a **three-stage RK3/CN scheme of overall order two**.
It is different from the two-stage CN–Heun method in `Flows.CNRK2` used in
the earlier finite-difference design report. The source coefficients are:

| Stage `j` | `A_j` | `B_j` | `C_j` | `λ_t,j = 1/(C_j*h)` |
| --- | --- | --- | --- | --- |
| 0 | 0 | 1/3 | 1/6 | 6/h |
| 1 | −5/9 | 15/16 | 5/24 | 24/(5h) |
| 2 | −153/128 | 8/15 | 1/8 | 8/h |

Writing `N` for the signed nonlinear acceleration, and omitting base-flow
and imposed-forcing terms for this correspondence, each stage is

$$
Q\leftarrow A_jQ+N(\boldsymbol u_j),\qquad
\boldsymbol r_j=\lambda_{t,j}\boldsymbol u_j
+\nu\Delta\boldsymbol u_j-\nabla p_j+\frac{B_j}{C_j}Q,
$$

$$
(\lambda_{t,j}-\nu\Delta)\boldsymbol u_{j+1}
+\nabla p_{j+1}=\boldsymbol r_j,\qquad
\nabla\cdot\boldsymbol u_{j+1}=0.
$$

The old pressure gradient is included in `NSE::linear`; the new pressure is
returned by `NSE::solve` and retained for the next stage. For each Fourier
mode the scalar velocity Helmholtz parameter is
`λ = λ_t,j + ν*κ²`. The three different stage parameters require three cached
elliptic/influence configurations at fixed timestep, as in `NSE::reset_lambda`.
Any future `Flows.jl` adapter must preserve this recurrence to claim numerical
correspondence with Gibson's CNRK2.

## Deliberate implementation differences

The Julia solver remains serial, with contiguous wall-normal columns and an
`x` half-spectrum. Channelflow's field storage uses a `z` half-spectrum.
This storage difference does not change the mode equations; comparisons must
match signed physical wave numbers and coefficient normalization. Julia
types and in-place methods will remain small, with source references at the
corresponding numerical operations.

The scalar changes have received source-level review only. No numerical tests
or performance measurements have been run for this migration.
