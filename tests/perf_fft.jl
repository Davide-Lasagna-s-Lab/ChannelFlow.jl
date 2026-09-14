using BenchmarkTools
using LinearAlgebra
using Printf
using FFTW

flags     = FFTW.EXHAUSTIVE
timelimit = 10
M         = 100
FFTW.set_num_threads(1)

# desired resolution
Nx, Ny, Nz = 64, 32, 64
Ns         = (Nx, Ny, Nz)

for is in ((1, 2, 3), (2, 1, 3), (1, 3, 2), (3, 1, 2), (2, 3, 1), (3, 2, 1))
    
    # indices
    ix, iy, iz = is
    
    # make plan
    a = zeros(Ns[ix], Ns[iy], Ns[iz])
    plan = plan_rfft(a, (ix, iz); flags=flags, timelimit=timelimit);
    out  = plan*a

    # time it
    t_min = minimum([@elapsed mul!(out, plan, a) for i = 1:M])

    @printf "layout = %s: t_min = %.8f ms\n" is 1000*t_min
end