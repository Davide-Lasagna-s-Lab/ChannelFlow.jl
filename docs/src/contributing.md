# Developer guide

## Source map

| Location | Responsibility |
|---|---|
| `src/problem.jl` | flow configuration and standard base profiles |
| `src/grids.jl`, `src/fields/` | grid, fields, operators and I/O |
| `src/ffts.jl`, `src/transforms/` | normalization, padding and transform plans |
| `src/nonlinear.jl` | rotational, convective and divergence forms |
| `src/solvers/` | modal Stokes, influence, tau and mean-mode constraints |
| `src/timesteppers/` | CNRK2 and Flows adapter |
| `src/pressure.jl`, `src/initialization.jl` | projection, pressure recovery and initial conditions |
| `src/postprocessing.jl` | volume-averaged diagnostics |
| `src/adapt.jl`, `ext/` | storage transfers and CUDA methods |

Keep numerical conventions identical across devices. Velocity components are
`(u,v,w)` even though array axes are `(x,z,y)` or `(kx,kz,n)`. Nonlinearity
includes the base profile and overwrites its output; forcing adds to it. Stage
pressure is retained for exact continuation. Problem caches are mutable and
must not be shared between simultaneous integrations.

## Build the documentation

```sh
julia --project=docs -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

Open `docs/build/index.html`. Set `CI=true` only when serving the generated site
with directory-style URLs. Documenter checks exported API coverage and internal
links; the quick-start CPU example is executed during the build. GPU instructions
are validated by the CUDA suite on a GPU host, not by a CPU-only documentation build.

The documentation workflow builds pull requests and publishes `main` through
GitHub Pages. Publication uses the workflow token and Pages artifact deployment;
no personal access token or deployment key is stored in the repository.

## Validation and benchmarking

Run the relevant checks from [Validation](validation.md). For device changes,
use CPU/GPU comparisons with scalar indexing disabled. Run timings only after
warmup, synchronize CUDA, and preserve raw data and environment provenance.
See [Benchmarks](benchmarks.md) for the complete-step protocol.
