using ChannelFlow
using ChebyshevHelmoltzSolvers: ChebCoeffs, diff!
using FFTW
using LinearAlgebra
using Test
import Flows
const CF = ChannelFlow
include("../helpers.jl")
include("test_viscous_decay.jl")

include("test_tollmien_schlichting.jl")

include("test_waleffe_equilibrium.jl")
