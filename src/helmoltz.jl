import FDGrids
import FDHelmoltzSolver

export FourierHelmoltzSolver, update!, solve!

"""
    FourierHelmoltzSolver(grid, T=Float64)

Cache one wall-normal Dirichlet Helmholtz factorisation per resolved Fourier
mode. `grid.D2` must be an `FDGrids.DiffMatrix`, as required by
`FDHelmoltzSolver`. Call `update!` before the first solve.
"""
struct FourierHelmoltzSolver{H, T<:AbstractFloat, G<:Grid}
    factors :: Matrix{H}           # one LU-backed FD solver per (kx, kz) mode
        rhs :: Vector{Complex{T}}  # wall-normal work vector, reused for every mode
       grid :: G                   # supplies D2, domain lengths and resolved sizes
end

function FourierHelmoltzSolver(grid::G, ::Type{T}=Float64) where {G<:Grid, T<:AbstractFloat}
    # FDHelmoltzSolver factors the compact banded storage of DiffMatrix;
    # a general dense D2 cannot be passed to its constructor.
    grid.D2 isa FDGrids.DiffMatrix ||
        throw(ArgumentError("grid.D2 must be an FDGrids.DiffMatrix"))

    # The rfft in x stores only nonnegative kx, whereas the z dimension
    # contains the full signed Fourier spectrum. Each stored mode has its
    # own factorisation because its horizontal wave-number square differs.
    Ny, Nxh, Nz = spectralsize(grid, NotPadded())
    first_factor = FDHelmoltzSolver.HelmoltzSolver(grid.D2, Complex{T})
    factors = Matrix{typeof(first_factor)}(undef, Nxh, Nz)
    factors[1, 1] = first_factor

    # Construct the remaining solvers with complex coefficients, matching
    # the element type of the spectral right-hand sides.
    for iz = 1:Nz, ix = 1:Nxh
        (ix, iz) == (1, 1) && continue
        factors[ix, iz] = FDHelmoltzSolver.HelmoltzSolver(grid.D2, Complex{T})
    end

    # A single wall-normal vector suffices because the serial solve visits
    # Fourier modes one at a time.
    return FourierHelmoltzSolver(factors, Vector{Complex{T}}(undef, Ny), grid)
end

"""
    update!(solver, θ0, θ1)

Factorise `θ0*(D2 - α²*kx² - β²*kz²) - θ1*I` for every Fourier mode,
where `α = 2π/Lx` and `β = 2π/Lz`.
Call again when either coefficient changes; solves reuse the cached factors.
"""
function update!(solver::FourierHelmoltzSolver, θ0::Real, θ1::Real)
    _, Nxh, Nz = spectralsize(solver.grid, NotPadded())
    # Convert integer mode numbers to physical wave numbers using the two
    # periodic domain lengths.
    α = 2π / domainsize(solver.grid, 1)
    β = 2π / domainsize(solver.grid, 2)

    # Follow FFTW storage order: nonnegative kx are the rfft prefix; full z
    # storage contains nonnegative kz followed by negative kz. The signed
    # conversion matches the spectral indexing loop used by the operators.
    for iz = 1:Nz, ix = 1:Nxh
        kx = ix - 1
        kz = iz <= (Nz >> 1) + 1 ? iz - 1 : iz - 1 - Nz
        k² = (kx * α)^2 + (kz * β)^2
        # θ0*(D2 - k²) - θ1*I = θ0*D2 - (θ1 + θ0*k²)*I.
        # update! assembles the banded operator and refreshes its LU factors.
        FDHelmoltzSolver.update!(solver.factors[ix, iz], θ0, θ1 + θ0*k²)
    end
    return solver
end

"""
    solve!(OUT, solver, F; lower=0, upper=0)

Solve every Fourier mode with Dirichlet wall values `lower` and `upper`.
`OUT` and `F` must have the resolved spectral size; `F` is preserved.
"""
function solve!(OUT::SpectralField{T}, solver::FourierHelmoltzSolver{H, T},
                F::SpectralField{T}; lower::Real=0, upper::Real=0) where {H, T}
    # The factor cache covers precisely the resolved spectral layout.
    expected = spectralsize(solver.grid, NotPadded())
    size(OUT) == size(F) == expected ||
        throw(DimensionMismatch("fields must have the resolved spectral size"))

    _, Nxh, Nz = expected
    for iz = 1:Nz, ix = 1:Nxh
        # FDHelmoltzSolver.solve! overwrites its right-hand side. Copy the
        # contiguous wall-normal column to preserve F, then place the solved
        # column in OUT; this also permits OUT and F to be the same field.
        @views solver.rhs .= parent(F)[:, ix, iz]

        # The Dirichlet values replace the first and last right-hand-side
        # entries before the cached triangular solve.
        FDHelmoltzSolver.solve!(solver.factors[ix, iz], solver.rhs, lower, upper)

        @views parent(OUT)[:, ix, iz] .= solver.rhs
    end
    return OUT
end
