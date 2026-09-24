# Run with the package environment on CPU, or test/cuda on an A100:
# julia --project=. benchmarks/step.jl cpu results.csv
# julia --project=test/cuda benchmarks/step.jl cuda results.csv
using ChannelFlow, FFTW, LinearAlgebra, Printf, Profile
const CF=ChannelFlow
const DEVICE=length(ARGS)>0 ? ARGS[1] : "cpu"
const OUTPUT=length(ARGS)>1 ? ARGS[2] : joinpath(@__DIR__, "results", "steps.csv")
const SOURCE_COMMIT = get(ENV, "CHANNEL_SOURCE_COMMIT") do
    readchomp(`git -C $(@__DIR__) rev-parse HEAD`)
end
const SAMPLES=parse(Int, get(ENV, "CHANNEL_SAMPLES", "100"))
const SIZES=parse.(Int, split(get(ENV, "CHANNEL_SIZES", "8,16,24,32,48,64,96,128"), ','))
const NTHREADS=parse(Int, get(ENV, "CHANNEL_FFT_THREADS", "1"))
FFTW.set_num_threads(NTHREADS)
BLAS.set_num_threads(NTHREADS)
if DEVICE=="cuda"
    @eval using CUDA, Adapt
    CUDA.allowscalar(false)
end
syncdevice() = DEVICE=="cuda" ? CUDA.synchronize() : nothing

function initial(g, p)
    # A smooth three-dimensional perturbation excites the complete nonlinear
    # path. Project once and reconstruct a consistent initial pressure;
    # initialization and planning are outside every timing interval.
    u=VectorField(
        PhysicalField(g, (x, y, z)->0.1*(1-y^2)*cos(x)*sin(z)),
        PhysicalField(g, (x, y, z)->0.05*(1-y^2)^2*cos(z)),
        PhysicalField(g, (x, y, z)->0.2*y*(1-y^2)*sin(z)),
    )
    U=project!(FFT(u), p)
    State(U, pressure(U, p))
end
function restore!(s, original)
    for (a, b) in zip(
        (velocity(s).components..., stagepressure(s)),
        (velocity(original).components..., stagepressure(original)),
    )
        copyto!(parent(a), parent(b))
    end
end
function measure(p, s, original)
    advance() = step!(p.scheme, p.nlterm, velocity(s), stagepressure(s), 0.0; p.constraint...)
    for _ = 1:5
        restore!(s, original)
        advance()
        syncdevice()
    end
    times=Float64[]
    bytes=Int[]
    allocs=Int[]
    for _ = 1:SAMPLES
        restore!(s, original)
        syncdevice()
        result=@timed begin
            advance()
            syncdevice()
        end
        push!(times, result.time)
        push!(bytes, result.bytes)
        push!(allocs, Base.gc_alloc_count(result.gcstats))
    end
    return minimum(times), sum(times)/length(times), minimum(bytes), minimum(allocs)
end
mkpath(dirname(OUTPUT))
open(OUTPUT, "w") do io
    println(
        io,
        "device,threads,Nx,Ny,Nz,samples,min_seconds,mean_seconds,host_bytes,host_allocations,commit",
    )
    for N in SIZES
        g=Grid(N, N+1, N, 2π, 2π)
        cpu=CouetteFlow(g, 1/400, 0.002; fftwflags = FFTW.MEASURE, fftwtimelimit = 1)
        state=initial(g, cpu)
        p=DEVICE=="cuda" ? adapt(CuArray, cpu) : cpu
        s=DEVICE=="cuda" ? adapt(CuArray, state) : state
        original=copy(s)
        seconds, average, bytes, allocs=measure(p, s, original)
        commit=SOURCE_COMMIT
        println(
            io,
            join(
                (DEVICE, NTHREADS, N, N+1, N, SAMPLES, seconds, average, bytes, allocs, commit),
                ',',
            ),
        )
        flush(io)
        @printf("%s N=%d: %.3f ms, %d host bytes\n", DEVICE, N, 1000seconds, bytes)
        # Release all large banks before the next size. GPU pooling is also
        # reclaimed so memory use reflects one problem, not the whole sweep.
        p=s=original=state=cpu=nothing
        GC.gc()
        DEVICE=="cuda" && CUDA.reclaim()
    end
end
