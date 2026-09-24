"""
    ChannelFlow

CPU and NVIDIA GPU Fourier--Chebyshev DNS for plane Couette and Poiseuille flow.

Start with `Grid`, `CouetteFlow` or `PoiseuilleFlow`, and a `State` created by
`zero_state` or `random_state`. Advance with `step!` (or optionally
`Flows.flow(problem)` after loading Flows) and
inspect the velocity with `kinetic_energy` and `dissipation_rate`.

Velocity is stored as a perturbation to `problem.baseflow`. The accompanying
stage pressure is algebraic; the default `RotatingForm` uses pressure augmented
by total kinetic energy per unit mass.
Physical arrays have order `(x, z, y)`; spectral arrays have `(k, l, n)`.
"""
module ChannelFlow

using ChebyshevHelmoltzSolvers
import Adapt
import ChebyshevHelmoltzSolvers: solve!
import FFTW
import LinearAlgebra
import Random
import Serialization

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
include("fields/io.jl")
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
include("solvers/batchedinfluence.jl")
include("solvers/stokes.jl")

#//////////////////////////////////////////////////////////////////////////////#
#///               TIME INTEGRATION AND PROBLEM CONSTRUCTION                ///#
#//////////////////////////////////////////////////////////////////////////////#

include("forcing.jl")
include("timesteppers/cnrk2.jl")
include("problem.jl")

#//////////////////////////////////////////////////////////////////////////////#
#///                   INITIALIZATION AND POSTPROCESSING                    ///#
#//////////////////////////////////////////////////////////////////////////////#

include("pressure.jl")
include("initialization.jl")
include("postprocessing.jl")
include("adapt.jl")

end
