# Numerical validation

This page reports **measured numerical checks**, with reference equations,
reproduction scripts and downloadable data. The physical cases live in
[`validation/`](https://github.com/Davide-Lasagna-s-Lab/ChannelFlow.jl/tree/main/validation)
and are also run by the physical regression suite. CPU/GPU agreement and
agreement with another implementation complement analytical checks; neither
can rule out a mathematical error shared by both implementations.

## Reproduce the figures

From the repository root, with the Julia project instantiated:

```sh
julia --project=. validation/run.jl
python validation/plot.py  # Python, NumPy and Matplotlib
```

Run one case by appending `viscous_decay`, `tollmien_schlichting`, or
`waleffe_equilibrium` to the Julia command. The runner writes CSVs and
[execution provenance](assets/validation/environment.txt) to `validation/results/`.
Plotting only reads saved data; it does not rerun DNS. The C++ comparison has
its own build/run instructions below. No reference data are downloaded by the
physical test runner.

Throughout, the walls are ``y=\pm1``, ``\boldsymbol{V}=U_b(y)\boldsymbol{e}_x+
\boldsymbol{u}`` is total velocity and ``\boldsymbol{u}`` is the evolved
perturbation. Define the volume average and perturbation kinetic energy by

```math
\langle f\rangle_V=\frac{1}{2L_xL_z}\int_0^{L_x}\int_{-1}^1\int_0^{L_z}
f\,dz\,dy\,dx,\qquad E'=\tfrac12\langle|\boldsymbol{u}|^2\rangle_V.
```

## Comparison with C++ Channelflow

We compare against [upstream Channelflow at revision
ad37ef3](https://github.com/epfl-ecps/channelflow/tree/ad37ef3022351d4e4a7a6c274c59a88605ad19e8),
built in Release mode without MPI. Both solvers use Couette flow, ``U_b=y``,
``\nu=1/400``, ``L_x=L_z=2\pi``, ``\Delta t=0.002``, rotational nonlinearity,
three-stage CNRK2, tau correction and zero imposed pressure gradient.

The grid conventions differ: Julia's resolved ``(N,N+1,N)`` corresponds to
C++'s **padded physical** ``(3N/2,N+1,3N/2)`` with `DealiasXZ`. Both retain
``|k_x|,|k_z|\le N/2-1``; no wall-normal dealiasing is applied. Using equal
constructor arguments would compare different Fourier resolutions.

The same projected velocity and modified pressure are exported from Julia on
the common padded physical grid, then transformed by C++. Each code advances
one step from that state. The input to projection is

```math
\widetilde{\boldsymbol{u}}=
\begin{pmatrix}
0.1(1-y^2)\cos x\sin z\\
0.05(1-y^2)^2\cos z\\
0.2y(1-y^2)\sin z
\end{pmatrix}.
```

We report maximum physical-space differences. For pressure, subtract the
constant difference at one grid point before taking the maximum:

```math
\epsilon_u=\max_{i,x,y,z}|u_i^{\mathrm{Julia}}-u_i^{\mathrm{C++}}|,
\qquad
\epsilon_q=\max_{x,y,z}|q^{\mathrm{Julia}}-q^{\mathrm{C++}}-c|.
```

This gauge correction is necessary because pressure is determined only up to
a spatial constant. In the rotational formulation ``q=p+|\boldsymbol{V}|^2/2``
(up to the implementation's spatially uniform gauge).

![One-step comparison with C++](assets/validation/cpp.svg)

At the better-resolved grid, the velocity agrees near roundoff and pressure
to approximately ``10^{-11}``. Low-resolution discrepancies need not have the
same roundoff floor. This is a **one-step consistency check for this state and
configuration**, not a long-time trajectory or turbulent-statistics comparison.
Timings from the parity driver are not the performance benchmark; see
[complete-step benchmarks](benchmarks.md).

```sh
bash validation/channelflow/build.sh /scratch/channelflow-cpp /path/to/fftw
bash validation/channelflow/run.sh /scratch/channelflow-cpp/step-release
```

The [comparison instructions](https://github.com/Davide-Lasagna-s-Lab/ChannelFlow.jl/blob/main/validation/channelflow/README.md)
record compiler settings and the binary exchange format.
[Download the measured errors](assets/validation/cpp.csv).

## Exact viscous decay

For either Couette ``U_b=y`` or Poiseuille ``U_b=1-y^2``, choose

```math
\boldsymbol{u}(y,z,t)=
A\sin\!\left[\frac{n\pi}{2}(y+1)\right]\cos(\beta z)
\exp(-\mu t)\,\boldsymbol{e}_x,
\qquad
\mu=\nu\left[\left(\frac{n\pi}{2}\right)^2+\beta^2\right].
```

The field is divergence free, satisfies homogeneous wall conditions, and is
independent of x with zero wall-normal velocity. Therefore all perturbation
advection and base-flow interaction terms vanish in the convective formulation.
The full DNS reduces exactly to diffusion, with zero perturbation pressure.
The Poiseuille base is sustained by ``\partial_x P=-2\nu``.

The independent predictions are

```math
E'(0)=A^2/8,\qquad E'(t)/E'(0)=e^{-2\mu t},\qquad
\tau'_{xy}=\nu A\frac{n\pi}{2}
\cos\!\left[\frac{n\pi}{2}(y+1)\right]\cos(\beta z)e^{-\mu t}.
```

The last expression is the signed xy stress component, not outward wall
traction. The tests check both walls, transverse velocity, pressure, zero
perturbation flux, incompressibility and energy. Energy uses an independently
constructed unweighted Chebyshev quadrature in y and periodic averaging.

We use ``A=0.1``, ``\nu=0.01``, ``\beta=1``, ``n=1,3``, grid ``(5,33,8)``,
and integrate to ``T=4`` with ``\Delta t=0.5,0.25,0.125``. The plotted final
relative maximum field error is
``\|u_h-u_{\mathrm{exact}}\|_\infty/\|u_{\mathrm{exact}}\|_\infty``.
Halving the time step must reduce this error by a factor between 3.5 and 4.5.

![Exact decay and second-order convergence](assets/validation/decay.svg)

Lines show the exact decay and open circles the DNS at the finest time step.
Couette and Poiseuille convergence curves overlap, as expected for this field.

[Download decay measurements](assets/validation/decay.csv).

## Tollmien–Schlichting instability of Poiseuille flow

This case checks growth **and phase propagation**, rather than just damping.
The reference is the unstable ``Re=8000``, ``\alpha=1``, ``\beta=0``
Poiseuille mode in [Mortensen (2017), section 5](https://arxiv.org/abs/1701.03787).
For disturbances proportional to ``e^{i\alpha x+st}``, the Orr–Sommerfeld
problem is

```math
s\mathcal{L}\hat v
=\nu\mathcal{L}^2\hat v-i\alpha U_b\mathcal{L}\hat v
+i\alpha U_b''\hat v,
\qquad \mathcal{L}=D^2-\alpha^2,\qquad
\hat v(\pm1)=D\hat v(\pm1)=0.
```

Continuity gives ``\hat u=iD\hat v/\alpha``. We build a separate dense
Chebyshev–Galerkin eigenproblem using basis functions

```math
\phi_k=T_k-\frac{2(k+2)}{k+3}T_{k+2}+\frac{k+1}{k+3}T_{k+4},
```

which satisfy the four wall conditions identically. This reference uses neither
DNS derivative routines nor the Stokes/influence solver. Its leading target
exponent is

```math
s=0.002664410371-0.2470750602i.
```

Thus ``s=-i\omega`` for the convention ``e^{i(\alpha x-\omega t)}``:
positive ``\Re(s)`` means instability, and
``E'(t)/E'(0)=\exp(2\Re(s)t)``. We compare eigenvalues at 65 and 81 wall-normal
points and check the generalized-eigenpair residual.

The nonlinear DNS uses grid ``(8,81,1)``, amplitude ``10^{-7}``, ``T=50`` and
``\Delta t=0.4,0.2,0.1``. Its positive Fourier coefficient is half the complex
wave amplitude because the negative harmonic is implicit in the real FFT.
Pressure is initialized from the eigenmode's streamwise momentum balance.
Projecting the evolving wall-normal profile onto its initial coefficient vector
gives a complex amplitude ``a(t)``. Then ``\log|a|/t`` estimates growth and
``-\arg(a)/t`` estimates frequency, with incremental phase unwrapping.

![TS energy growth and time refinement](assets/validation/ts.svg)

Energy is evaluated with independent Gauss–Legendre quadrature and Fourier
Parseval multiplicities. The final convergence error is the relative Euclidean
norm of the complete velocity coefficient vector against ``e^{sT}`` times
the initial field; it is **not** a volume-energy norm. Nonlinear contamination
is small but nonzero at finite amplitude. Tests require more than 3.5-fold
error reduction under time-step halving and final error below ``10^{-5}``;
this particular case may converge faster through cancellation.

| Time step | Final relative coefficient error |
|---:|---:|
| 0.4 | 2.915e-04 |
| 0.2 | 2.806e-05 |
| 0.1 | 2.608e-06 |

[Energy/phase history](assets/validation/ts_history.csv) ·
[Convergence and eigenpair data](assets/validation/ts_convergence.csv).

## Waleffe's lower- and upper-branch Couette equilibria

The two finite-amplitude steady states are taken from
[Waleffe's original archive](https://people.math.wisc.edu/~fwaleffe/ECS/RRC-data.html):
``Re=400``, ``\alpha=1.14``, ``\gamma=2.5``,
``L_x=2\pi/\alpha``, ``L_z=2\pi/\gamma``. Cite Waleffe (2003) and
Wang, Gibson & Waleffe (2007), listed in [References](references.md).

An exact equilibrium satisfies

```math
(\boldsymbol{V}\cdot\nabla)\boldsymbol{V}
=-\nabla p+\nu\Delta\boldsymbol{V},\qquad
\nabla\cdot\boldsymbol{V}=0,\qquad
\boldsymbol{V}(\pm1)=\pm\boldsymbol{e}_x.
```

For a statistically steady or exactly steady Couette flow with no volume
forcing, total dissipation ``D`` and wall input ``I`` balance:

```math
D=\nu\langle\nabla\boldsymbol{V}:\nabla\boldsymbol{V}\rangle_V,
\qquad
I=\frac{\nu}{2}\left[
\langle\partial_y V_x\rangle_{xz,y=1}
+\langle\partial_y V_x\rangle_{xz,y=-1}\right].
```

The base gradient must be included in both quantities. For laminar Couette,
``D=I=\nu``. We also monitor the perturbation energy and relative drift

```math
\delta(t)=\frac{\|\boldsymbol{u}(t)-\boldsymbol{u}(0)\|_{L^2(V)}}
{\|\boldsymbol{u}(0)\|_{L^2(V)}}.
```

The archive stores 34 Chebyshev coefficients on a 32×32 physical periodic
mesh. The importer first checks published mean/RMS profiles, then subtracts
``y=T_1`` from the streamwise zero Fourier mode to obtain the perturbation.
It retains the archived velocity unchanged: there is no Newton refinement.
Initial pressure is recovered from a stationary Stokes solve. The relative
difference between that solve's velocity and the archived velocity measures
a stationary defect; it does not replace the initial condition.

We use grids ``(32,35,32)`` and ``(48,49,48)``, steps ``0.025`` and ``0.0125``,
and compare at ``t=0.25,0.5``. Branch-specific defect/drift bounds are
``10^{-4}`` (LB) and ``2\times10^{-3}`` (UB); these are regression tolerances,
not estimates of continuum error. Initial ``|I/D-1|`` must be below ``10^{-3}``.

![Archived equilibrium drift and power balance](assets/validation/waleffe.svg)

Refining the DNS cannot recover absent archive coefficients. A nonzero defect
may reflect finite archive resolution and differences in discretization; these
checks do not uniquely separate their contributions. Time-refined trajectories
start from the same archive and must differ by less than ``10^{-5}`` relatively
at ``T=0.5``. No long-time stability claim is made for these unstable states.

| Branch | Grid | Stationary defect | Drift at t=0.5 (dt=0.0125) |
|---|---|---:|---:|
| LB | 32×35×32 | 1.141e-05 | 8.257e-06 |
| LB | 48×49×48 | 1.151e-05 | 8.284e-06 |
| UB | 32×35×32 | 5.361e-04 | 3.071e-04 |
| UB | 48×49×48 | 5.383e-04 | 3.066e-04 |

The figure uses the finer time step; the CSV also includes the coarser run.

[Measurements](assets/validation/waleffe.csv) ·
[Archive provenance and checksums](https://github.com/Davide-Lasagna-s-Lab/ChannelFlow.jl/blob/main/validation/data/README.md).

## Scope of the evidence

The interface/analytic suite additionally checks transforms, derivatives,
Helmholtz/Stokes residuals, projection, pressure and mean-flow constraints.
The CUDA suite checks CPU/GPU agreement with scalar device indexing disabled.
Run them separately with `test/runtests.jl` and `test/cuda/runtests.jl`.

The [MKM590 example](mkm590.md) provides a restartable turbulent comparison
workflow. It is **not a completed statistical validation**: a convincing
comparison additionally requires spatial/time convergence, resolved wall layers,
adequate averaging and uncertainty estimates. The laminar, linear-instability
and equilibrium checks above do not establish those properties.
