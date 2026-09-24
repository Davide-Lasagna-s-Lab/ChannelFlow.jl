# Reproducible numerical validation

The [validation manual](https://davide-lasagna-s-lab.github.io/ChannelFlow.jl/validation/)
explains the reference equations, parameters, error norms and limits of each case.

From the repository root:

```sh
julia --project=test -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
julia --project=test validation/run.jl
python validation/plot.py
```

Plotting requires NumPy and Matplotlib. The Julia runner accepts individual case
names, e.g. `validation/run.jl viscous_decay`. Output defaults to `validation/results`;
set `CHANNEL_VALIDATION_OUTPUT` to select another directory. Case CSVs are replaced
on each run; do not mix results from different solver revisions. `environment.txt`
records the source revision and whether there were local edits. The committed
CSVs and figures are measured outputs, not synthetic illustrations.

- `viscous_decay.jl`: exact diffusion for Couette and Poiseuille, energy, wall shear,
  and time-step convergence.
- `tollmien_schlichting.jl`: independent Orr–Sommerfeld eigenproblem, small-amplitude
  Poiseuille instability, growth, phase and time-step convergence.
- `waleffe_equilibrium.jl`: lower/upper branch archived Couette equilibria,
  stationary defect, short-time drift and energy balance.
- `data/`: original Waleffe archives and source/checksum information.
- `channelflow/`: pinned upstream serial build, identical-state exchange and parity
  driver. Follow its README to regenerate `results/cpp.csv` before plotting.

The physical regression entry point `test/physics/runtests.jl` includes these same
cases, with temporary outputs; no second implementation is maintained. The runner
uses independent quadrature/manufactured-field helpers from `test/helpers.jl`.
The wall-shear comparison indexes y in the final array dimension.

The MKM590 turbulent workflow remains in `examples/mkm590/`. It is not labelled a
passed statistical validation while convergence and sampling uncertainty remain
unestablished. CPU/GPU numerical checks remain in `test/cuda/`.
