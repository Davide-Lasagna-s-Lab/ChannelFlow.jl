import FFTW

export Grid, points, Padded, NotPadded, physicalsize, spectralsize

struct Padded end
struct NotPadded end

"""
    Grid(Ny, Nx, Nz, Lx, Lz, baseflow)

Grid for a Fourier--Chebyshev--Fourier discretisation.
`Ny` is the number of Chebyshev--Lobatto nodes, ordered from the upper wall to
the lower wall, as in Channelflow. `Nx` and `Nz` are the resolved periodic
point counts, and `Lx` and `Lz` are the corresponding domain lengths. The
wall-normal domain is fixed to `[-1,1]`; `domainsize` stores `(Lx, 2, Lz)`.

Pass the streamwise reference profile as a function of `y`. The grid stores
its ordinary Chebyshev coefficients, obtained once with a DCT-I, for addition
to the zero Fourier mode.
"""
struct Grid{Y<:AbstractVector, B<:AbstractVector}
             y::Y                  # Lobatto nodes, upper wall first
            Nx::Int                # resolved streamwise extent
            Nz::Int                # resolved spanwise extent
    domainsize::NTuple{3, Float64} # lengths (x, y, z)
      baseflow::B                  # Chebyshev coefficients of the base velocity

    function Grid(      Ny::Int,
                        Nx::Int,
                        Nz::Int,
                        Lx::Real,
                        Lz::Real,
                  baseflow::Function)
        Ny ≥ 3 || throw(ArgumentError("at least three Lobatto nodes are required"))

        # Gibson's convention uses Lobatto points from +1 to -1.
        y = [cospi(n/(Ny-1)) for n = 0:Ny-1]
        values = baseflow.(y)

        # Channelflow's ChebyCoeff::chebyfft convention: unweighted series
        # coefficients, including half the raw DCT weights at degrees 0 and P.
        coefficients = FFTW.r2r(float.(values), FFTW.REDFT00)
        coefficients ./= Ny-1
        coefficients[1] /= 2
        coefficients[end] /= 2

        return new{typeof(y), typeof(coefficients)}(
            y, Nx, Nz, (Float64(Lx), 2.0, Float64(Lz)), coefficients)
    end

end

"""Return the resolved physical array dimensions in storage order `(y, x, z)`."""
physicalsize(grid::Grid, ::NotPadded) = (length(grid.y), grid.Nx, grid.Nz)

"""Return the 3/2-padded physical array dimensions in storage order `(y, x, z)`."""
function physicalsize(grid::Grid, ::Padded)
    Ny, Nx, Nz = physicalsize(grid, NotPadded())
    return (Ny, _paddedsize(Nx), _paddedsize(Nz))
end

_paddedsize(n::Integer) = cld(3n, 2) | 1

"""
    spectralsize(grid, Padded() or NotPadded())

Return the spectral storage size for the selected physical grid. The real
transform in `x` stores only its nonnegative half-spectrum.
"""
function spectralsize(grid::Grid, tag::Union{Padded, NotPadded})
    Ny, Nx, Nz = physicalsize(grid, tag)
    return (Ny, (Nx >> 1) + 1, Nz)
end

"""Return the domain lengths `(Lx, 2, Lz)`."""
domainsize(grid::Grid) = grid.domainsize

"""Return ordinary Chebyshev coefficients of the streamwise base profile."""
baseflow(grid::Grid) = grid.baseflow

"""
    points(grid::Grid)

Return broadcast-compatible coordinates in storage order `(y, x, z)`. The
periodic grids cover `[0,Lx)` and `[0,Lz)` without repeated endpoints.
"""
function points(grid::Grid)
    Ny, Nx, Nz = physicalsize(grid, NotPadded())
    Lx, _, Lz = domainsize(grid)
    y = reshape(grid.y, Ny, 1, 1)
    x = reshape(range(0, Lx; length=Nx + 1)[1:Nx], 1, Nx, 1)
    z = reshape(range(0, Lz; length=Nz + 1)[1:Nz], 1, 1, Nz)
    return y, x, z
end
