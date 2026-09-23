export StokesSolver

#//////////////////////////////////////////////////////////////////////////////#
#///                      GLOBAL FOURIER STOKES SOLVER                      ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    StokesSolver(grid, nu, lambda)

Cache the batched primitive-variable Stokes solve for all resolved Fourier
modes on `grid`. `nu` is viscosity and `lambda` the temporal shift, as in
[`InfluenceModeSolver`](@ref) and [`MeanModeSolver`](@ref).

Store a separate mean-mode solver and one bank of batched influence systems.
Rows follow flattened `(kx,kz)` FFT storage. The mean slot is overwritten
by its dedicated solve and excluded Nyquist planes are cleared afterwards.
Physical wavenumbers include `2π/Lx` and `2π/Lz`.

Require odd `Ny ≥ 3`, positive periodic sizes and finite positive `Lx, Lz`.
Rebuild when the grid, viscosity or temporal shift changes. Factors and
workspaces are reused, so one instance must not be used concurrently.
"""
struct StokesSolver{G, S, M}
     grid::G
    modes::S
     mean::M

    function StokesSolver(  grid::Grid,
                                     nu::Real,
                                 lambda::Real)
        Nx, Nz, Ny = physicalsize(grid, NotPadded())
        Lx, _, Lz = domainsize(grid)
        isodd(Ny) || throw(ArgumentError("Gibson's tau correction requires odd Ny"))
        Nx > 0 && Nz > 0 || throw(ArgumentError("Nx and Nz must be positive"))
        isfinite(Lx) && isfinite(Lz) && Lx > 0 && Lz > 0 ||
            throw(ArgumentError("Lx and Lz must be finite and positive"))

        mean = MeanModeSolver(Ny, nu, lambda)
        Nxh, _, _ = spectralsize(grid, NotPadded())
        kx = vec([(2π/Lx)*(ix-1) for ix=1:Nxh, iz=1:Nz])
        kz = vec([(2π/Lz)*(iz <= (Nz >> 1)+1 ? iz-1 : iz-1-Nz)
                  for ix=1:Nxh, iz=1:Nz])
        # The zero slot is overwritten by the separate mean-mode solve.
        # A nonsingular placeholder keeps every field a direct matrix view.
        modes = BatchedInfluenceSolver(Ny, kx, kz, nu, lambda)
        return new{typeof(grid), typeof(modes), typeof(mean)}(grid, modes, mean)
    end
end

"""Wrap Fourier slot `(ix, iz)` as Chebyshev coefficients without copying data."""
_chebcolumn(U::SpectralField, ix::Int, iz::Int, ::AbstractVector) =
    view(parent(U), ix, iz, :)

"""
    solve!(solver::StokesSolver, U, P, R;
           pressuregradient=nothing, bulkvelocity=nothing, baseflow=nothing)

Overwrite the perturbation velocity `U::VectorField` and pressure
`P::SpectralField` from the momentum source `R::VectorField`. All component
fields must store `ComplexF64` coefficients on the solver's grid, with the
resolved size `spectralsize(grid, NotPadded())`.

Apply the modal Stokes systems through zero-copy `(system, coefficient)` matrix views.
Treat the mean separately and set all output Nyquist planes to zero. Sources
are preserved, including excluded modes. Outputs must have distinct storage
and must not overlap inputs or solver workspaces. Input spectra must satisfy
the conjugate symmetry required for real physical fields.

Return `(dPdx, dPdz)`. The mutually exclusive keywords have the same meaning
as in [`solve!(::MeanModeSolver, u, v, w, p, Rx, Ry, Rz)`](@ref), except that
`bulkvelocity=(Ubulk, Wbulk)` specifies TOTAL bulk velocities here. If
`baseflow` is supplied as ordinary Chebyshev coefficients, its streamwise mean
is subtracted to obtain the perturbation target; otherwise the target is
interpreted as a perturbation target. The grid itself has no base flow.

The stage assembly remains responsible for placing base-flow viscous terms
and any explicit forcing in `R`. This solve does not add them or advance time.
"""
function solve!(          solver::StokesSolver,
                               U::VectorField{F},
                               P::F,
                               R::VectorField{F};
                pressuregradient::Union{Nothing, NTuple{2, Real}}=nothing,
                    bulkvelocity::Union{Nothing, NTuple{2, Real}}=nothing,
                         baseflow=nothing) where {F<:SpectralField{Float64}}
    fields = (U.components..., P, R.components...)
    expected = spectralsize(solver.grid, NotPadded())
    for field in fields
        grid(field) == solver.grid || throw(ArgumentError("fields must use the solver's grid"))
        size(field) == expected || throw(DimensionMismatch("fields must have the resolved spectral size"))
    end

    # Only the mean mode sees the uniform pressure gradient or bulk target.
    target = isnothing(bulkvelocity) ? nothing :
             (bulkvelocity[1] - (isnothing(baseflow) ? 0.0 :
                _bulkmean(baseflow)),
              bulkvelocity[2])
    solve!(solver.modes, map(spectralmatrix, fields)...)
    gradients = solve!(solver.mean, map(field -> _chebcolumn(field, 1, 1, solver.mean.work), fields)...;
                       pressuregradient=pressuregradient, bulkvelocity=target)

    for field in (U.components..., P)
        zero_nyquist!(field)
    end
    return gradients
end
