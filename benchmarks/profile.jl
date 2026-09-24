# Sampling and kernel profiles of warmed, complete CNRK2 time steps.
using ChannelFlow, FFTW, LinearAlgebra, Profile
include("profile_summary.jl")
include("initial.jl")
const PROFILE_STEPS = parse(Int, get(ENV, "CHANNEL_PROFILE_STEPS", "20"))
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
    Profile.@profile for _ = 1:PROFILE_STEPS
        advance()
    end
    save_profile_summary(get(ENV,"CHANNEL_PROFILE_CSV",joinpath(@__DIR__,"results","cpu-phases.csv")))
    println("CPU FLAT PROFILE")
    Profile.print(format = :flat, sortedby = :count, mincount = 5)
    println("CPU CALL TREE")
    Profile.print(format = :tree, mincount = 10)
end
function main()
    N=parse(Int, get(ENV, "CHANNEL_PROFILE_N", "64"))
    threads = parse(Int, get(ENV, "CHANNEL_FFT_THREADS", "1"))
    FFTW.set_num_threads(threads)
    BLAS.set_num_threads(threads)
    println("device=$DEVICE grid=($N,$(N+1),$N) FFT/BLAS threads=$threads steps=$PROFILE_STEPS")
    println("Julia $(VERSION); host=$(gethostname()); source=$(get(ENV, "CHANNEL_SOURCE_COMMIT", "unspecified"))")
    p=CouetteFlow(
        Grid(N, N+1, N, 2π, 2π),
        1/400,
        0.002;
        fftwflags = FFTW.MEASURE,
        fftwtimelimit = 1,
    )
    s=initial(p.grid, p)
    if DEVICE=="cuda"
        CUDA.allowscalar(false)
        profile_cuda(adapt(CuArray, p), adapt(CuArray, s))
    else
        profile_cpu(p, s)
    end
end
main()
