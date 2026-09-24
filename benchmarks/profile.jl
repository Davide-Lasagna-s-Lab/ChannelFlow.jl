# Sampling and kernel profiles of warmed, complete CNRK2 time steps.
using ChannelFlow, FFTW, LinearAlgebra, Profile
const DEVICE=isempty(ARGS) ? "cpu" : ARGS[1]
if DEVICE=="cuda"
    @eval using CUDA, Adapt
    include("profile_cuda.jl")
end
function profile_cpu(p, s)
    advance() = step!(p.scheme, p.nlterm, velocity(s), stagepressure(s), 0.0; p.constraint...)
    for _ = 1:10
        advance()
    end
    Profile.clear()
    Profile.@profile for _ = 1:200
        advance()
    end
    println("CPU FLAT PROFILE")
    Profile.print(format = :flat, sortedby = :count, mincount = 5)
    println("CPU CALL TREE")
    Profile.print(format = :tree, mincount = 10)
end
function main()
    N=parse(Int, get(ENV, "CHANNEL_PROFILE_N", "64"))
    FFTW.set_num_threads(1)
    BLAS.set_num_threads(1)
    p=CouetteFlow(
        Grid(N, N+1, N, 2π, 2π),
        1/400,
        0.002;
        fftwflags = FFTW.MEASURE,
        fftwtimelimit = 1,
    )
    s=roll_state(p, 0.1)
    if DEVICE=="cuda"
        CUDA.allowscalar(false)
        profile_cuda(adapt(CuArray, p), adapt(CuArray, s))
    else
        profile_cpu(p, s)
    end
end
main()
