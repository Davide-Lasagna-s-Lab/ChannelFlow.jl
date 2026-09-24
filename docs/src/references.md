# References and numerical lineage

## Formulation and time integration

- **Gibson, J. F. and Channelflow contributors.** The maintained
  [Channelflow source](https://github.com/epfl-ecps/channelflow) is the operational
  reference for terminology and coefficients: [`RungeKuttaDNS`](https://github.com/epfl-ecps/channelflow/blob/master/channelflow/dnsalgo.cpp)
  implements `CNRK2`, and [`TauSolver`](https://github.com/epfl-ecps/channelflow/blob/master/channelflow/tausolver.cpp)
  implements the influence and tau corrections. These links identify the
  algorithms being followed; ChannelFlow.jl is a separate implementation.
- **Kleiser, L. & Schumann, U. (1980).** *Treatment of incompressibility and
  boundary conditions in 3-D numerical spectral simulations of plane channel
  flows*, in the Third GAMM Conference proceedings, pp. 165–173.
  [Institutional bibliographic record](https://publikationen.bibliothek.kit.edu/240013603).
  This is the pressure–velocity influence-matrix lineage.
- **Canuto, C., Hussaini, M. Y., Quarteroni, A. & Zang, T. A. (2006).**
  *Spectral Methods: Fundamentals in Single Domains.*
  [Springer](https://doi.org/10.1007/978-3-540-30726-6).
  Background for Fourier/Chebyshev expansions, tau discretisation and spectral accuracy.
- **Viswanath, D. (2014).** *Spectral integration of linear boundary value problems*,
  Journal of Computational and Applied Mathematics, 268, 159–169.
  [Author's preprint](https://arxiv.org/abs/1205.2717).
  Background for integration-based treatment of wall-normal boundary-value problems.

The detailed scalar and batched Helmholtz method belongs to
[ChebyshevHelmoltzSolvers.jl](https://github.com/Davide-Lasagna-s-Lab/ChebyshevHelmoltzSolvers.jl).
Its README develops the tau equations, quasi-tridiagonal factorisation and
boundary-response construction; the present manual focuses on their role in DNS.

## Physical reference cases

- **Moser, R. D., Kim, J. & Mansour, N. N. (1999).** *Direct numerical simulation
  of turbulent channel flow up to Reτ = 590*, Physics of Fluids, 11, 943–945.
  [Paper](https://doi.org/10.1063/1.869966),
  [authors' reference database](https://turbulence.oden.utexas.edu/MKM_1999.html).
- **Waleffe, F. (2003).** *Homotopy of exact coherent structures in plane shear flows*,
  Physics of Fluids, 15, 1517–1534. [Paper](https://doi.org/10.1063/1.1566753).
- **Wang, J., Gibson, J. & Waleffe, F. (2007).** *Lower branch coherent states in
  shear flows: transition and control*, Physical Review Letters, 98, 204501.
  [Paper](https://doi.org/10.1103/PhysRevLett.98.204501),
  [author's data](https://people.math.wisc.edu/~fwaleffe/ECS/RRC-data.html).

Cite the reference-data papers when using those cases. Benchmark plots in this
manual compare CPU and GPU execution of this Julia solver; they are not a
performance comparison with the C++ Channelflow software.
