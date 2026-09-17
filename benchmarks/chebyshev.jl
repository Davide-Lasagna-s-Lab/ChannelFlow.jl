# Run from the package root: julia --project=perf benchmarks/chebyshev.jl
using ChannelFlow, FFTW, LinearAlgebra, Random, Statistics
using PyPlot

# BLAS threads are independent of Julia's --threads setting. Keep FFTW
# single-threaded to isolate the benefit of threaded matrix multiplication.
const SIZES = parse.(Int, split(get(ENV, "NY", "17,33,49,65,97,129"), ','))
const BLAS_THREADS = parse.(Int, split(get(ENV, "BLAS_THREADS", "1,2,4"), ','))

function timings!(dest, plan, src; samples=9)
    for _ = 1:3
        mul!(dest, plan, src)
    end
    # Batch short transforms to reduce timer overhead; planning and compilation
    # are excluded. Report the median of independent batches.
    elapsed = @elapsed mul!(dest, plan, src)
    repeats = clamp(ceil(Int, 0.02/max(elapsed, 1e-9)), 1, 1000)
    GC.gc()
    times = map(1:samples) do _
        elapsed = @elapsed begin
            for _ = 1:repeats
                mul!(dest, plan, src)
            end
        end
        elapsed/repeats
    end
    bytes = @allocated mul!(dest, plan, src)
    return (; ms=1000median(times), bytes)
end

function benchmark()
    all(n -> n >= 3, SIZES) || error("NY values must be at least 3")
    all(>(0), BLAS_THREADS) || error("BLAS_THREADS values must be positive")
    oldblas, oldfftw = BLAS.get_num_threads(), FFTW.get_num_threads()
    output = joinpath(@__DIR__, "results")
    mkpath(output)
    rows = NamedTuple[]
    Random.seed!(42)
    try
        FFTW.set_num_threads(1)
        for Ny in SIZES
            # Grow the number of Fourier columns as well as the polynomial
            # degree: Nx=Nz=Ny-1, i.e. all three directions grow together.
            Nx = Nz = Ny-1
            g = Grid(Nx, Ny, Nz, 2π, 2π)
            src = SpectralField(g)
            randn!(parent(src))
            dest, reference = similar(src), similar(src)
            for (direction, constructor) in ((:forward, ChannelFlow.plan_cheb),
                                              (:inverse, ChannelFlow.plan_icheb))
                # FFTW.MEASURE may destroy its planning input. Always plan on
                # a disposable copy so every backend sees identical data.
                fftw = constructor(copy(src), :fftw; flags=FFTW.MEASURE)
                mul!(reference, fftw, src)
                gemm = constructor(src, :gemm)
                for (backend, threads, plan) in
                    vcat([(:fftw, 1, fftw)], [(:gemm, n, gemm) for n in BLAS_THREADS])
                    BLAS.set_num_threads(threads)
                    mul!(dest, plan, src)
                    # Both plans must implement the same scaling before their
                    # execution times can be meaningfully compared.
                    isapprox(parent(dest), parent(reference); rtol=1e-11, atol=1e-11) ||
                        error("Backend mismatch at Ny=$Ny, direction=$direction")
                    timing = timings!(dest, plan, src)
                    row = (; Nx, Ny, Nz, direction, backend,
                            blas_threads=BLAS.get_num_threads(), fftw_threads=1,
                            timing...)
                    push!(rows, row)
                    println(row)
                end
            end
        end
    finally
        BLAS.set_num_threads(oldblas)
        FFTW.set_num_threads(oldfftw)
    end

    open(joinpath(output, "chebyshev.csv"), "w") do io
        println(io, join(string.(keys(first(rows))), ','))
        for row in rows
            println(io, join(values(row), ','))
        end
    end
    open(joinpath(output, "environment.txt"), "w") do io
        println(io, "Julia: ", VERSION, "\nCPU: ", Sys.CPU_NAME)
        println(io, "Julia threads: ", Threads.nthreads(), "\nBLAS: ", BLAS.get_config())
        println(io, "FFTW: MEASURE, 1 thread; time excludes planning")
    end
    fig, axes = subplots(1, 2; figsize=(11, 4), constrained_layout=true)
    for (ax, direction) in zip(axes, (:forward, :inverse))
        for (backend, threads) in vcat([(:fftw, 1)], [(:gemm, n) for n in BLAS_THREADS])
            selected = sort(filter(r -> r.direction == direction &&
                r.backend == backend && r.blas_threads == threads, rows); by=r -> r.Ny)
            label = backend == :fftw ? "FFTW (1 thread)" : "GEMM ($threads threads)"
            ax.plot([r.Ny for r in selected], [r.ms for r in selected], "o-"; label)
        end
        ax.set(xlabel="Ny (Nx = Nz = Ny − 1)", ylabel="Time per transform [ms]",
               title=string(direction), yscale="log")
        ax.grid(true; alpha=0.3)
        ax.legend()
    end
    fig.savefig(joinpath(output, "chebyshev.png"); dpi=180)
    fig.savefig(joinpath(output, "chebyshev.svg"))
    close(fig)
    println("Results written to ", output)
end

benchmark()
