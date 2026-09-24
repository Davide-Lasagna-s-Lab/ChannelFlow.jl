# Smaller-box Poiseuille DNS: MKM590

This GPU example targets the **Moser, Kim & Mansour (1999)** channel at
nominal Reτ = 590 (measured 587.19), starting from a small smooth perturbation
of the laminar state. It stores streamwise-velocity y–z slices, restartable
mean-flow and Reynolds-stress statistics, and a complete restart state.
No turbulent result or convergence is implied by merely running the example.

## Reference and grid

The reference uses Lx = 2πh, Ly = 2h, Lz = πh, and
(Nx,Ny,Nz) = (384,257,384), Fourier–Chebyshev discretisation and constant
mass flux. We use the same domain and resolved counts, with 3/2 periodic
padding and CNRK2. At Reτ = 587.19, Δx+ = 9.61, Δz+ = 4.80,
maximum Chebyshev Δy+ ≈ 7.21 and first-node distance y+ ≈ 0.0442.

- [Paper, Table I](https://doi.org/10.1063/1.869966)
- [Authors' database](https://turbulence.oden.utexas.edu/MKM_1999.html)
- [Mean profile](https://turbulence.oden.utexas.edu/data/MKM/chan590/profiles/chan590.means)
- [Reynolds stresses](https://turbulence.oden.utexas.edu/data/MKM/chan590/profiles/chan590.reystress)

The two reference files are saved unchanged in `reference/`, including their
attribution and normalization headers. Their y coordinate is distance from
the lower wall; ours ranges from −1 to +1. The reference velocities and
stresses are in friction units. Chebyshev quadrature of the published mean
gives Ub+ ≈ 18.6544 and Reb ≈ 10953.68. We set h = Ub = 1 and
ν = 1/(587.19 × 18.6544), so constant flux matches the reference bulk Reynolds
number to the precision of its tabulated profile. Reτ is measured, not forced.

The matched bulk Reynolds number does not give Reτ = 590 in the initial
laminar state: it gives about **181.3**. The increased wall stress of the
subsequent turbulent state should raise it toward the reference value. This
transient is expected and must not be included in the final comparison.

The bulk-velocity conversion can be reproduced directly from the reference:

```julia
using DelimitedFiles, ChebyshevHelmoltzSolvers
m = readdlm("examples/mkm590/reference/chan590.means"; comments=true, comment_char='#')
# Reflect the half-channel mean onto the full 257-point Lobatto grid.
Uplus = vcat(m[:, 3], m[end-1:-1:1, 3])
a = chebcoeffs(Uplus)
Ubplus = sum(a[j] / (1-(j-1)^2) for j in 1:2:length(a))
nu = 1 / (587.19 * Ubplus)  # units: Ub = h = 1
```

## Run and restart

From the repository root, with the CUDA environment already instantiated:

```sh
julia --project=test/cuda examples/mkm590/run.jl
```

The settings are at the top of `run.jl` as ordinary assignments (`NX = 384`, `DT = 0.0025`, etc.).
Edit these values directly before running. Defaults: dt = 0.0025, 200000 additional steps,
small perturbation parameter 0.01, snapshots every 100 steps and checkpoints
every 1000. `MAX_WALL_TIME` stops the job cleanly after 11 hours, before
the example Slurm allocation expires. Resume with `RESTART = true`. The initial velocity is analytically divergence-free and no-slip;
pressure is reconstructed before time marching. No artificial forcing or
reference mean profile is used to sustain turbulence. The bulk flux is fixed.

Use `run.slurm` for IRIDIS X (submit from the repository root). This is a large
job: check host/device memory and run a short startup check before production.
For an inexpensive execution check only, edit NX/NY/NZ and the number of
steps; such a run is not a DNS validation at this Reynolds number.

Set `RESTART = true`, keep `OUTPUT` pointing to the same directory, and
run the same command again. `N_STEPS` specifies additional steps.

Checkpoints atomically store velocity, stage pressure, integer step, physical
configuration and accumulated statistics. Restart checks the configuration,
removes slices/history newer than the checkpoint, and restores the matching
statistics. It then samples on the same integer-step cadence. Create a `STOP`
file in the output directory to request a checkpoint and clean stop; remove
it before restarting. Do not run two writers in the same directory.

To discard transient statistics while continuing the saved velocity, restart
with `RESET_STATS = true` and a new `STATS_START`. This deliberately
resets only the profile accumulator; it preserves the state and time history.

## Mean flow and Reynolds stresses

Only samples at t ≥ `STATS_START` (default 200 bulk-time units) contribute
to the average. This cutoff is a starting choice: inspect wall stress, bulk
velocity and fluctuation energy, and move it later if transition is incomplete.
The profile collector stores all three means and all six stresses:

$$\overline{u_i}(y)=\langle u_i\rangle_{x,z,t},\qquad
R_{ij}(y)=\langle(u_i-\overline{u_i})(u_j-\overline{u_j})\rangle_{x,z,t}.$$

The laminar base profile is included in the total velocity. Spatial reductions
run on the GPU; only profiles move to the host. The online variance merge
includes fluctuations of the instantaneous plane mean. It therefore computes
covariance relative to the full plane-and-time mean, not a separate detrending
at every instant. Normal stresses are variances, not RMS values.

- `profiles.csv`: y,U,V,W,uu,vv,ww,uv,uw,vw in bulk units.
- `statistics.toml`: sample count and first/last averaging times.
- `history.csv`: time, bulk velocity, instantaneous wall-stress Reτ, CFL and
  perturbation energy (relative to the laminar profile).
- `slice-*.bin`: total u at x=0, on the nonuniform (z,y) grid.
- `checkpoint.bin`: Julia restart record; use compatible package/Julia versions.

## Compare and animate

Use Python 3.11+ with NumPy and Matplotlib; animation also needs ffmpeg
(on PATH, or supplied through the `FFMPEG` environment variable).

```sh
python examples/mkm590/compare.py /path/to/run
python examples/mkm590/animate.py /path/to/run --stride 2 --fps 20
```

`compare.py` overlays our mean and Reynolds stresses with the reference and
saves `comparison.png` and `.pdf`. It folds both walls, reverses the sign of
uv at the upper wall, and normalizes using the mean wall shear of **our**
averaged profile. It reports the resulting Reτ. Comparing time blocks and
extending the averaging interval are still necessary to judge convergence.
`animate.py` uses a fixed color scale and respects Chebyshev node spacing;
it saves an MP4 and the final frame as PNG/PDF. Select a developed-flow frame
for the package logo only after inspecting the diagnostic history.

## Monitor an IRIDIS run locally

```sh
bash examples/mkm590/sync.sh JOB_ID
bash examples/mkm590/sync.sh JOB_ID --watch
```

The second command refreshes every 60 seconds until the job ends; Ctrl-C
stops only the local watcher, not the DNS. Edit the host, key and remote path
at the top of `sync.sh` if necessary. VPN/SSH access must be available.

Files arrive in `examples/mkm590/remote-output/`, excluded from Git:
`run.log`, `job-status.txt`, `last-sync.txt`, `history.csv`, configuration,
averaged profiles (once averaging starts), and completed y–z slices. The
large binary checkpoint and incomplete temporary files are not transferred.
Synchronization is one-way: it never modifies the remote run. Stale local
slices are removed if the simulation rolls back to an earlier checkpoint;
locally generated figures are preserved.

Run `compare.py` or `animate.py` with `examples/mkm590/remote-output` as
the input directory. Before `STATS_START`, no averaged profiles are expected.
