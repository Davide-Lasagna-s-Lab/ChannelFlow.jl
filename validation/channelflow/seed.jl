# Export identical projected velocity and modified pressure to the C++ driver.
# Physical arrays have x-fastest order, followed by z and y, with upper wall first.
using ChannelFlow, FFTW, LinearAlgebra
include("../../benchmarks/initial.jl")
FFTW.set_num_threads(1)
BLAS.set_num_threads(1)
N = parse(Int, ARGS[1])
p = CouetteFlow(Grid(N, N+1, N, 2π, 2π), 1/400, 0.002;
                fftwflags=FFTW.MEASURE, fftwtimelimit=1)
s = initial(p.grid, p)
function write_physical(path, s)
    open(path, "w") do io
        for field in (velocity(s).components..., stagepressure(s))
            write(io, parent(IFFT(field)))
        end
    end
end
write_physical(ARGS[2], s)
# A one-step reference allows comparison outside all timing intervals.
if length(ARGS) >= 3
    step!(p.scheme, p.nlterm, velocity(s), stagepressure(s), 0.0; p.constraint...)
    write_physical(ARGS[3], s)
end
