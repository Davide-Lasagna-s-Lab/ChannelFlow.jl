# Waleffe Couette equilibria

Original, unmodified archives downloaded on 2026-09-15 from
[Fabian Waleffe’s ECS database](https://people.math.wisc.edu/~fwaleffe/ECS/RRC-data.html).
Both branches use Re=400, alpha=1.14, gamma=2.5, with 34 Chebyshev
coefficients and a 32-by-32 periodic mesh. These files contain total velocity.

The importer follows the author's
[reconstruction script](https://people.math.wisc.edu/~fwaleffe/ECS/rebuild_fields.m).
Tests read archive members using `tar` with gzip support; no network access
is needed. Published mean/rms profiles are checked before DNS conversion.

Please cite the publications requested by the data provider, particularly
Wang, Gibson and Waleffe, “Lower branch coherent states in shear flows:
transition and control” (2007), and Waleffe, “Homotopy of exact coherent
structures in plane shear flows” (2003).

## Integrity

SHA-256 hashes of the original archives:

- `RRC.a1.14.g2.5.R400.LB.tar.gz`: `8a89d0104c934e3d67fc7b73f0425491d3007bd16e5faf69c8416b8c68439129`
- `RRC.a1.14.g2.5.R400.UB.tar.gz`: `4aa7521785b07818fb7e60af50f57f40387f92d4b92032c2674155805d005b20`

## Interpretation of the benchmark

The test retains the imported velocity and recovers only its initial pressure
from stationary momentum balance. It does not use a Newton solve or substitute
a numerically refined equilibrium. Two DNS resolutions and two time steps
measure short-time drift. Increasing DNS resolution cannot recover missing
archive coefficients. The branch-specific drift bounds accommodate that
observed residual in this discretization and must not be interpreted as
continuum error bounds. These checks alone cannot attribute the residual
uniquely to the archive or to differences between numerical formulations.
The test does not establish long-time stability of either equilibrium.
