export Grid

"""
    Grid(D1, D2, Nx, Nz, domainsize, baseflow)

Grid for a Fourier--Fourier--finite-difference discretisation. `D1` and `D2`
differentiate in the wall-normal direction. `Nx` and `Nz` are the numbers of
points in the periodic directions, `domainsize` contains `(Lx, Lz)`, and
`baseflow` is the streamwise reference profile. The number of wall-normal
points is inferred from `D1` and `D2`.
"""
struct Grid{M1<:AbstractMatrix, M2<:AbstractMatrix, B<:AbstractVector}
            D1::M1                 # first derivative in the wall-normal direction
            D2::M2                 # second derivative in the wall-normal direction
      gridsize::NTuple{3, Int}     # physical grid points in x, y and z
    domainsize::NTuple{2, Float64} # lengths of the two periodic directions
      baseflow::B                  # streamwise base velocity at the wall-normal nodes

    function Grid(D1::M1, D2::M2,
                  Nx::Int, Nz::Int,
                  domainsize::NTuple{2, Real},
                  baseflow::B) where {M1<:AbstractMatrix,
                                      M2<:AbstractMatrix,
                                      B<:AbstractVector}
        # The matrices act on the same wall-normal grid.
        n = size(D1, 1)
        size(D1) == size(D2) == (n, n) ||
            throw(DimensionMismatch("D1 and D2 must be square matrices of equal size"))

        # The base profile provides one value at every wall-normal node.
        length(baseflow) == n ||
            throw(DimensionMismatch("baseflow and differentiation matrices must have the same wall-normal size"))

        # Store periodic lengths with a uniform floating-point representation.
        return new{M1, M2, B}(
            D1, D2, (Nx, n, Nz), Float64.(domainsize), baseflow)
    end
end

"""Return the number of physical grid points in `(x, y, z)`."""
gridsize(grid::Grid) = grid.gridsize

"""Return the lengths of the two periodic directions."""
domainsize(grid::Grid) = grid.domainsize

"""Return the length of periodic direction `i`."""
domainsize(grid::Grid, i::Integer) = grid.domainsize[i]

"""Return the streamwise base-flow profile."""
baseflow(grid::Grid) = grid.baseflow
