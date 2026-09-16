import FFTW

export Grid, points, Padded, NotPadded, physicalsize, spectralsize, chebyshev_coefficients

#//////////////////////////////////////////////////////////////////////////////#
#///                     GRID GEOMETRY AND PADDING TAGS                     ///#
#//////////////////////////////////////////////////////////////////////////////#

struct Padded end
struct NotPadded end

"""
    Grid(Ny, Nx, Nz, Lx, Lz)

Grid for a Fourier--Chebyshev--Fourier discretisation.
`Ny` is the number of Chebyshev--Lobatto nodes, ordered from the upper wall to
the lower wall, as in Channelflow. `Nx` and `Nz` are the resolved periodic
point counts, and `Lx` and `Lz` are the corresponding domain lengths. The
wall-normal domain is fixed to `[-1,1]`; `domainsize` stores `(Lx, 2, Lz)`.

The base flow is a property of a physical problem and belongs to
`ChannelFlowProblem`, not to the grid.
"""
struct Grid{Y<:AbstractVector}
             y::Y                  # Lobatto nodes, upper wall first
             #TODO: it's cleaner to store Nx, Ny, Nz as a tuple (in storage order)
            Nx::Int                # resolved streamwise extent
            Nz::Int                # resolved spanwise extent
    domainsize::NTuple{3, Float64} # lengths (x, y, z)

    function Grid(Ny::Int, Nx::Int, Nz::Int, Lx::Real, Lz::Real)
        # Gibson's convention uses Lobatto points from +1 to -1.
        y = [cospi(n/(Ny-1)) for n = 0:Ny-1]
        return new{typeof(y)}(y, Nx, Nz, (Float64(Lx), 2.0, Float64(Lz)))
    end

end

#//////////////////////////////////////////////////////////////////////////////#
#///                  PHYSICAL AND SPECTRAL STORAGE SIZES                   ///#
#//////////////////////////////////////////////////////////////////////////////#

"""Return the resolved physical array dimensions in storage order `(y, x, z)`."""
#TODO: so that we can return that tuple directly here
physicalsize(grid::Grid, ::NotPadded) = (length(grid.y), grid.Nx, grid.Nz)

"""Return the 3/2-padded physical array dimensions in storage order `(y, x, z)`."""
function physicalsize(grid::Grid, ::Padded)
    Ny, Nx, Nz = physicalsize(grid, NotPadded())
    return (Ny, _paddedsize(Nx), _paddedsize(Nz))
end

#TODO: improve documentation of this function, explaining rationale, mechanisms to avoid aliasing with examples
# Retained Nyquist planes are zero, so an even padded length is valid.
# Do not force odd sizes: e.g. 32 -> 48 is both sufficient and FFT-friendly.
_paddedsize(n::Integer) = cld(3n, 2)

"""
    spectralsize(grid, Padded() or NotPadded())

Return the spectral storage size for the selected physical grid. The real
transform in `x` stores only its nonnegative half-spectrum.
"""
function spectralsize(grid::Grid, tag::Union{Padded, NotPadded})
    Ny, Nx, Nz = physicalsize(grid, tag)
    return (Ny, (Nx >> 1) + 1, Nz)
end

#//////////////////////////////////////////////////////////////////////////////#
#///                     DOMAIN LENGTHS AND COORDINATES                     ///#
#//////////////////////////////////////////////////////////////////////////////#

"""Return the domain lengths `(Lx, 2, Lz)`."""
domainsize(grid::Grid) = grid.domainsize

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

#//////////////////////////////////////////////////////////////////////////////#
#///                     CHEBYSHEV PROFILE COEFFICIENTS                     ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    chebyshev_coefficients(grid, profile)

Return ordinary Chebyshev coefficients of `profile(y)` on the grid's
wall-normal Lobatto points. The profile itself is not stored by `Grid`.
"""
function chebyshev_coefficients(grid::Grid, profile::Function)
    values = profile.(grid.y)
    coefficients = FFTW.r2r(float.(values), FFTW.REDFT00)
    coefficients ./= length(grid.y) - 1
    coefficients[1] /= 2
    coefficients[end] /= 2
    return coefficients
end
