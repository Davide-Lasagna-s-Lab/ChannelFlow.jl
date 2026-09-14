using CanonicalFlows
using BenchmarkTools
using InteractiveUtils
using Grids
using Profile

# physical grid size
gridsize = (512, 256, 64)

# define the grid in the vertical direction
grid = Grid(gridsize[3], Float64, 3, -1, 1, 0.5)

# make field
U  = SpectralField(gridsize, grid)
Ux = similar(U)
Uz = similar(U)

# @btime ddx1!($Ux, $U)
# @btime ddx2!($Uz, $U)
@btime ddx3!($Uz, $U)
# ddx3!(Uz, U)
# @profile ddx3!(Uz, U)
# Profile.print()
