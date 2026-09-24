module ChannelFlowCUDAExt

using CUDA, Adapt, LinearAlgebra
import ChannelFlow as CF
import ChebyshevHelmoltzSolvers as CH

const CuSpectral{T} = CF.SpectralField{T,A} where {T<:AbstractFloat,A<:CuArray{Complex{T},3}}
const CuPhysical{T} = CF.PhysicalField{T,A} where {T<:AbstractFloat,A<:CuArray{T,3}}
const CuInfluence = CF.BatchedInfluenceSolver{H,A} where {H,A<:CuArray}

include("transforms.jl")
include("operators.jl")
include("stokes.jl")
include("pressure.jl")
CF._same_storage(object, ::CuSpectral) = Adapt.adapt(CuArray, object)

# Rebuild transform plans on the device; transfer cached factors and fields.
function Adapt.adapt_structure(to::Type{<:CuArray}, p::CF.ChannelFlowProblem)
    scheme = Adapt.adapt(to, p.scheme)
    U = scheme.N[1]
    u = CF.PhysicalField(CUDA.zeros(Float64, CF.physicalsize(p.grid, CF.Padded())), p.grid)
    form = typeof(p.nlterm).parameters[2]()
    # Preserve an explicitly selected dense backend when moving the problem.
    backend = p.nlterm.fft.chebyplan isa CF.GEMMChebyshevPlan ? :gemm : :cufft
    nl = CF.NonLinearTerm(u, U, scheme.baseflow; form = form, chebbackend = backend)
    return CF.ChannelFlowProblem(
        p.grid,
        p.baseflow,
        nl,
        scheme,
        Adapt.adapt(to, p.forcing),
        p.constraint,
    )
end

end
