# Restart and sampling

A reproducible restart needs **velocity, stage pressure, simulation time and
problem parameters**. The field file contains the state and grid metadata;
time, viscosity, timestep, base profile, forcing and driving settings belong to
the simulation and must be retained alongside it.

```julia
using Adapt
savefield("state.bin", adapt(Array, gpu_state))
state = loadfield("state.bin")
gpu_state = adapt(CuArray, state)
# Recreate/adapt the same problem and resume at the saved simulation time.
```

Download a GPU state before saving. `savefield` also accepts individual fields
and vector fields. Serialization is intended for compatible Julia/package
versions and trusted files; it is not a portable long-term interchange format.
Reconstructing instantaneous pressure from velocity is useful for new initial
conditions but does not reproduce the stored CNRK2 stage history exactly.

## Uniform time histories across runs

Use integer step counts, a fixed nominal `dt`, and a sampling stride `m`.
The target sampling interval is `m*dt`. Carry the global step counter and time
into the next run, and append only new samples. Do not insert a shortened final
step between batches when a uniformly sampled record is required for spectra.

```julia
for step in first_step+1:last_step
    t = step!(problem.scheme, problem.nlterm,
        velocity(state), stagepressure(state), t;
        forcing=problem.forcing, problem.constraint...)
    if step % save_every == 0
        # Record the observable and t here.
    end
end
```

Checkpoint state and statistics together. If output was appended after the last
checkpoint, truncate those records on restart rather than counting them twice.
The [MKM example](mkm590.md) implements this policy with atomic checkpoint writes,
integer sampling cadence and recovery of accumulated profile statistics.
