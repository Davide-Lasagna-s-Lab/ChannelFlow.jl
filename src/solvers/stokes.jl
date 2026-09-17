export StokesSolver

#//////////////////////////////////////////////////////////////////////////////#
#///                      GLOBAL FOURIER STOKES SOLVER                      ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    StokesSolver(grid, nu, lambda)

Cache the serial primitive-variable Stokes solve for all resolved Fourier
modes on `grid`. `nu` is viscosity and `lambda` the temporal shift, as in
[`InfluenceModeSolver`](@ref) and [`MeanModeSolver`](@ref).

Store a separate mean-mode solver and an influence solver for each active
nonzero mode. `modes[ix, iz]` follows the FFT storage order: nonnegative `kx`,
then positive and negative `kz` in their FFT slots. The mean and excluded
Nyquist slots contain `nothing`; they allocate no influence systems.
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
        Ny, Nx, Nz = physicalsize(grid, NotPadded())
        Lx, _, Lz = domainsize(grid)
        isodd(Ny) || throw(ArgumentError("Gibson's tau correction requires odd Ny"))
        Nx > 0 && Nz > 0 || throw(ArgumentError("Nx and Nz must be positive"))
        isfinite(Lx) && isfinite(Lz) && Lx > 0 && Lz > 0 ||
            throw(ArgumentError("Lx and Lz must be finite and positive"))

        mean = MeanModeSolver(Ny, nu, lambda)
        _, Nxh, _ = spectralsize(grid, NotPadded())
        modes = [if (ix == 1 && iz == 1) ||
                    (iseven(Nx) && ix == Nxh) ||
                    (iseven(Nz) && iz == (Nz >> 1)+1)
                     nothing
                 else
                     kx = (2π/Lx)*(ix-1)
                     kz = (2π/Lz)*(iz <= (Nz >> 1)+1 ? iz-1 : iz-1-Nz)
                     InfluenceModeSolver(Ny, kx, kz, nu, lambda)
                 end for ix = 1:Nxh, iz = 1:Nz]
        return new{typeof(grid), typeof(modes), typeof(mean)}(grid, modes, mean)
    end
end

"""Wrap Fourier slot `(ix, iz)` as Chebyshev coefficients without copying data."""
_chebcolumn(U::SpectralField{T}, ix::Int, iz::Int,
            ::ChebCoeffs{S, N}) where {T, S, N} =
    ChebCoeffs{Complex{T}, N}(view(parent(U), :, ix, iz))

"""
    solve!(solver::StokesSolver, U, P, R;
           pressuregradient=nothing, bulkvelocity=nothing, baseflow=nothing)

Overwrite the perturbation velocity `U::VectorField` and pressure
`P::SpectralField` from the momentum source `R::VectorField`. All component
fields must store `ComplexF64` coefficients on the solver's grid, with the
resolved size `spectralsize(grid, NotPadded())`.

Apply the modal Stokes systems through views of contiguous Chebyshev columns.
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
                _bulkmean(ChebCoeffs(baseflow))),
              bulkvelocity[2])
    gradients = solve!(solver.mean, map(field -> _chebcolumn(field, 1, 1, solver.mean.work), fields)...;
                       pressuregradient=pressuregradient, bulkvelocity=target)

    _, Nxh, Nz = expected
    for iz = 1:Nz, ix = 1:Nxh
        mode = solver.modes[ix, iz]
        isnothing(mode) && continue
        solve!(mode, map(field -> _chebcolumn(field, ix, iz, solver.mean.work), fields)...)
    end
    for field in (U.components..., P)
        zero_nyquist!(field)
    end
    return gradients
end
