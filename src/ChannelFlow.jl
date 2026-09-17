"""
    ChannelFlow

Serial Fourier--Chebyshev DNS for plane Couette and Poiseuille flow.

Start with `Grid`, `CouetteFlow` or `PoiseuilleFlow`, and a `State` created by
`zero_state` or `random_state`. Integrate with `Flows.flow(problem)` and
inspect the result with `Postprocessor` and `flow_diagnostics`.

Velocity is stored as a perturbation to `problem.baseflow`. The accompanying
stage pressure is algebraic; the default `RotatingForm` uses pressure augmented
by total kinetic energy per unit mass.
Physical arrays have order `(y, x, z)`; spectral arrays have `(n, kx, kz)`.
"""
module ChannelFlow

using ChebyshevHelmoltzSolvers
import ChebyshevHelmoltzSolvers: solve!
import FFTW
import Flows
import LinearAlgebra
import Random

#//////////////////////////////////////////////////////////////////////////////#
#///                   GRID, FIELDS AND SPATIAL OPERATORS                   ///#
#//////////////////////////////////////////////////////////////////////////////#

include("grids.jl")
include("fields/physicalfield.jl")
include("fields/spectralfield.jl")
include("fields/abstractfield.jl")
include("fields/vectorfield.jl")
include("fields/gradientfield.jl")
include("state.jl")
include("indexing.jl")
include("fields/operators.jl")

#//////////////////////////////////////////////////////////////////////////////#
#///                    TRANSFORMS AND MOMENTUM SOLVERS                     ///#
#//////////////////////////////////////////////////////////////////////////////#

include("transforms/chebyshev.jl")
include("ffts.jl")
include("nonlinear.jl")
# Modal solvers are separated by their mathematical role. All definitions
# remain in ChannelFlow; these files do not introduce nested modules.
include("solvers/influence.jl")
include("solvers/meanmode.jl")
include("solvers/stokes.jl")

#//////////////////////////////////////////////////////////////////////////////#
#///               TIME INTEGRATION AND PROBLEM CONSTRUCTION                ///#
#//////////////////////////////////////////////////////////////////////////////#

include("forcing.jl")
include("timesteppers/cnrk2.jl")
include("problem.jl")
include("timesteppers/channel.jl")

#//////////////////////////////////////////////////////////////////////////////#
#///                   INITIALIZATION AND POSTPROCESSING                    ///#
#//////////////////////////////////////////////////////////////////////////////#

include("pressure.jl")
include("initialization.jl")
include("postprocessing.jl")

end
