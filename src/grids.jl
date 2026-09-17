export Grid, points, Padded, NotPadded, physicalsize, spectralsize

#//////////////////////////////////////////////////////////////////////////////#
#///                     GRID GEOMETRY AND PADDING TAGS                     ///#
#//////////////////////////////////////////////////////////////////////////////#

struct Padded end
struct NotPadded end

"""
    Grid(Nx, Ny, Nz, Lx, Lz)

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
    physicalsize::NTuple{3, Int}     # number of grid points in physical space in (x, y, z) order
      domainsize::NTuple{3, Float64} # domain lengths in (x, y, z) order

    function Grid(Nx::Int, Ny::Int, Nz::Int, Lx::Real, Lz::Real)
        y = [cospi(n/(Ny-1)) for n = 0:Ny-1]
        return new{typeof(y)}(y, (Nx, Ny, Nz), (Float64(Lx), 2.0, Float64(Lz)))
    end
end

"""Compare grid resolution, domain lengths and wall-normal nodes by value."""
Base.:(==)(a::Grid, b::Grid) =
    a === b || (a.physicalsize == b.physicalsize &&
                a.domainsize == b.domainsize && a.y == b.y)

#//////////////////////////////////////////////////////////////////////////////#
#///                  PHYSICAL AND SPECTRAL STORAGE SIZES                   ///#
#//////////////////////////////////////////////////////////////////////////////#

"""Return the resolved physical array dimensions in storage order `(y, x, z)`."""
function physicalsize(grid::Grid, ::NotPadded)
    Nx, Ny, Nz = grid.physicalsize
    return (Ny, Nx, Nz)
end

"""Return the 3/2-padded physical array dimensions in storage order `(y, x, z)`."""
function physicalsize(grid::Grid, ::Padded)
    Ny, Nx, Nz = physicalsize(grid, NotPadded())
    return (Ny, _paddedsize(Nx), _paddedsize(Nz))
end

"""
    _paddedsize(n)

Return `ceil(3n/2)` periodic samples for evaluating quadratic products.
Resolved Nyquist modes are filtered: padding separates retained modes from
aliases of their products before Fourier truncation. Even padded sizes are
valid and FFT-friendly; for example, 32 resolved samples use 48 padded points.
Only x and z are padded; the Chebyshev direction is unchanged.
"""
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

"""Return the domain lengths in physical order `(Lx, 2, Lz)`."""
domainsize(grid::Grid) = grid.domainsize

"""
    points(grid::Grid, tag=NotPadded())

Return broadcast-compatible coordinates in storage order `(y, x, z)`. The
periodic grids cover `[0,Lx)` and `[0,Lz)` without repeated endpoints.
"""
function points(grid::Grid, tag::Union{Padded, NotPadded}=NotPadded())
    Ny, Nx, Nz = physicalsize(grid, tag)
    Lx, _, Lz = domainsize(grid)
    y = reshape(grid.y, Ny, 1, 1)
    x = reshape(range(0, Lx; length=Nx + 1)[1:Nx], 1, Nx, 1)
    z = reshape(range(0, Lz; length=Nz + 1)[1:Nz], 1, 1, Nz)
    return y, x, z
end
