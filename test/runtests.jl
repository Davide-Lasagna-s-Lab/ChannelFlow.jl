# These tests check API contracts and numerical building blocks against
# manufactured analytic data. They do not validate turbulent channel
# statistics or establish the time integration convergence order.
using ChannelFlow
import ChebyshevHelmoltzSolvers
using ChebyshevHelmoltzSolvers: ChebCoeffs, diff!, endpoint_derivative
using FFTW
using LinearAlgebra
using Random
using Test
import Flows

const CF = ChannelFlow

# Shared helpers construct analytic reference coefficients and evaluate wall
# values and bulk integrals. Individual test files then cover storage,
# transforms, derivatives, nonlinear algebra, constrained solves and
# integration dispatch.
include("helpers.jl")

@testset "ChannelFlow interfaces and analytic numerical checks" begin
    include("test_fields.jl")
    include("test_fieldio.jl")
    include("test_ffts.jl")
    include("test_operators.jl")
    include("test_nonlinear.jl")
    include("test_helmoltz.jl")
    include("test_stokes.jl")
    include("test_projection.jl")
    include("test_postprocessing.jl")
    include("test_timestepping.jl")
end
