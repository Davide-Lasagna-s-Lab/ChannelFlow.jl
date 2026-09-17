# Run with julia --startup-file=no --project=. perf/benchmark_step.jl [--record] [--profile] [--cpu] [--measure]
using ChannelFlow, FFTW, Random, LinearAlgebra, Profile, SHA, Dates
import Flows, ChebyshevHelmoltzSolvers

# Hash source contents as well as Git revisions: locally developed dependencies
# may change independently of ChannelFlow. This identifies dirty measurements.
function source_identity(root)
    paths = sort(filter(p -> endswith(p, ".jl"),
        [joinpath(d, f) for (d, _, fs) in walkdir(joinpath(root, "src")) for f in fs]))
    return (commit=readchomp(`git -C $root rev-parse HEAD`),
            dirty=!isempty(readchomp(`git -C $root status --porcelain -- src Project.toml`)),
            hash=bytes2hex(sha256(join(read.(paths, String)))))
end

# Keep this case fixed when comparing commits: Couette, Waleffe box, Float64,
# convective form, dt=0.025, one Julia/BLAS thread and no trajectory monitor.
function benchmark_step(; record=false, profile=false, cpu=false, measure=false)
    root = dirname(dirname(pathof(ChannelFlow)))
    BLAS.set_num_threads(1)
    g = Grid(32, 35, 32,2π/1.14,2π/2.5)
    c = CouetteFlow(g,1/400,0.025; form=ChannelFlow.ConvectiveForm(),
                fftwflags=measure ? FFTW.MEASURE : FFTW.ESTIMATE)
    # Build the same seeded random projected input, without expensive FFTW
    # planning in random_state. Initialization is outside all measurements.
    Random.seed!(42)
    physical = PhysicalField(zeros(Float64,physicalsize(g,Padded())),g)
    initial = zero_state(g)
    fft = ForwardFFT!(physical; flags=FFTW.ESTIMATE)
    for u in velocity(initial).components
        randn!(parent(physical)); parent(physical) .*= 0.01
        fft(u,physical)
    end
    initial = project!(velocity(initial), c)
    state = copy(initial)
    reset() = begin
        for i=1:3
            copyto!(parent(velocity(state)[i]),parent(velocity(initial)[i]))
        end
        copyto!(parent(stagepressure(state)),parent(stagepressure(initial)))
    end
    direct() = step!(c.scheme,c.nlterm,velocity(state),stagepressure(state),0.0;
                     c.constraint...)
    flow = Flows.flow(c)
    cases = (("step",direct,1), ("flow_10_steps",()->flow(state,(0.0,0.25)),10))
    identity = source_identity(root)
    backend = source_identity(dirname(dirname(pathof(ChebyshevHelmoltzSolvers))))
    commit, dirty, fingerprint = identity
    for (name, call, steps) in cases
        measure && (name *= "_measure")
        for _=1:3; reset(); call(); end
        measurements = map(1:9) do _
            reset(); GC.gc()
            sample = @timed call()
            (ms=1000sample.time/steps, bytes=sample.bytes/steps,
             allocations=Base.gc_alloc_count(sample.gcstats)/steps)
        end
        median_ms = sort([m.ms for m in measurements])[5]
        best = minimum(m.ms for m in measurements)
        bytes = minimum(m.bytes for m in measurements)
        allocations = minimum(m.allocations for m in measurements)
        println("$name: median=$median_ms ms, min=$best ms, $bytes bytes, $allocations allocations / step")
        if record
            output = joinpath(@__DIR__,"history.csv")
            open(output,"a") do io
                if filesize(output)==0
                    println(io,"utc,commit,dirty_source,source_sha256,backend_commit,backend_dirty,backend_sha256,julia,cpu,julia_threads,case,median_ms,min_ms,bytes,allocations")
                end
                println(io,join((Dates.now(Dates.UTC),commit,dirty,fingerprint,backend...,VERSION,
                                 Sys.CPU_NAME,Threads.nthreads(),name,median_ms,best,bytes,allocations),','))
            end
        end
    end
    if cpu
        # Time individual kernels on allocated buffers. These are diagnostic
        # per-call costs, not additive stage timings (some operations overlap).
        u, n, grad, tmp, gradient = c.nlterm.cache
        kernels = (("nonlinear", () -> c.nlterm(0.0, velocity(state), c.scheme.N)),
                   ("gradient", () -> ChannelFlow.grad!(gradient, velocity(state))),
                   ("stokes", () -> ChannelFlow.solve!(c.scheme.solvers[1], velocity(state), stagepressure(state), c.scheme.R)),
                   ("inverse_scalar", () -> c.nlterm.ifft(u[1], tmp[1])),
                   ("forward_scalar", () -> c.nlterm.fft(tmp[1], u[1])),
                   ("inverse_chebyshev", () ->
                       LinearAlgebra.mul!(c.nlterm.ifft.resolved,
                                                      c.nlterm.ifft.chebyplan,
                                                      tmp[1])),
                   ("inverse_fourier", () -> FFTW.unsafe_execute!(c.nlterm.ifft.plan, parent(c.nlterm.ifft.padded), parent(u[1]))))
        kernel_lines = String[]
        for (name, kernel) in kernels
            kernel()
            times = [(@elapsed kernel())*1000 for _=1:11]
            line = string(name, ": median ", sort(times)[6], " ms/call")
            push!(kernel_lines, line)
            println(line)
        end
        reset(); Profile.clear()
        Profile.@profile for _ = 1:40
            direct()
        end
        open(joinpath(@__DIR__, "cpu-profile.txt"), "w") do io
            println(io, "ChannelFlow: $identity; backend: $backend")
            println(io, "FFTW planning: ", measure ? "MEASURE" : "ESTIMATE")
            foreach(line -> println(io, line), kernel_lines)
            Profile.print(io; format=:flat, sortedby=:count, mincount=10, C=true)
        end
    end
    if profile
        reset(); Profile.Allocs.clear()
        Profile.Allocs.@profile sample_rate=0.01 direct()
        totals = Dict{String,Tuple{Int,Int}}()
        for allocation in Profile.Allocs.fetch().allocs
            index = findfirst(allocation.stacktrace) do frame
                file = String(frame.file)
                occursin("/src/",file) &&
                    (occursin("ChannelFlow",file) || occursin("Chebyshev",file))
            end
            site = isnothing(index) ? string(allocation.type) : string(allocation.stacktrace[index].func, " at ", allocation.stacktrace[index].file, ":", allocation.stacktrace[index].line)
            bytes,count = get(totals,site,(0,0))
            totals[site] = (bytes+allocation.size,count+1)
        end
        open(joinpath(@__DIR__,"allocations.txt"),"w") do io
            println(io,"Commit: $commit; dirty_source=$dirty; source_sha256=$fingerprint")
            println(io,"Chebyshev backend: $backend")
            println(io,"One warmed direct step; 1% allocation sampling (counts and bytes below are sampled). Grouped by first solver frame.")
            for (site,(bytes,count)) in sort(collect(totals);by=x->-x[2][1])
                println(io,"$bytes bytes\t$count allocations\t$site")
            end
        end
    end
end

benchmark_step(record="--record" in ARGS, profile="--profile" in ARGS, cpu="--cpu" in ARGS, measure="--measure" in ARGS)
