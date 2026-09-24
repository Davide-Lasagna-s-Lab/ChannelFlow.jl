# Validation and limitations

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



## What each test establishes

| Test family | Independent quantity or reference |
|---|---|
| Fields, transforms, derivatives | manufactured periodic fields and polynomial derivatives |
| Helmholtz and Stokes | scalar solves, boundary residuals, momentum residuals, coefficientwise divergence |
| Projection and pressure | idempotence, no slip, incompressibility, manufactured Poisson data |
| CNRK2 | exact viscous decay and time-refinement behaviour |
| Poiseuille instability | Tollmien–Schlichting growth of a prescribed eigenmode |
| Couette equilibria | Waleffe lower/upper branch archives and short-time drift |
| CUDA | CPU/GPU state agreement, all nonlinear forms, odd/even Fourier sizes, both transform backends |

The recorded CPU suite contains 3,078 interface/analytic checks and 833 physical
checks. The current CUDA suite has 158 checks, including 40 transform checks.
Test counts identify the recorded suite, not a measure of physical accuracy.
The [benchmark data](benchmarks.md) and repository test files provide evidence
and exact configurations.

For the Waleffe archive, published resolution and import conventions limit the
comparison: refining the DNS does not restore missing archive coefficients.
Consult the [data provenance](https://github.com/Davide-Lasagna-s-Lab/ChannelFlow.jl/blob/main/test/physics/data/README.md).

A turbulent run additionally needs spatial/time convergence, adequate averaging,
a resolved wall layer, and statistical uncertainty estimates. Agreement between
CPU and GPU alone cannot detect a mathematical error shared by both. The MKM
example is a reproducible comparison workflow; successful setup is not evidence
that its turbulent statistics have converged.
