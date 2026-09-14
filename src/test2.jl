using BenchmarkTools
using CanonicalFlows

function kernel(U, V, c)
    Threads.@threads for j = 1:length(U)
        @inbounds U[j] += c*V[j]
    end
    return nothing
end

N = 2^8
T = Float64

# real data
U = randn(T, N, N, N)
V = randn(T, N, N, N)
c = 1e-3

@btime kernel($U, $V, $c)

# field data
U = Field((N, N, N), CPU(), T)
V = Field((N, N, N), CPU(), T)
c = 1e-3

@btime kernel($U, $V, $c)

# complex data
U = randn(T, N>>1+1, N, N) + im*randn(T, N>>1+1, N, N)
V = randn(T, N>>1+1, N, N) + im*randn(T, N>>1+1, N, N)
c = 1e-3

@btime kernel($U, $V, $c)

# complex data
U = FTField((N, N, N), CPU(), T)
V = FTField((N, N, N), CPU(), T)
c = 1e-3

@btime kernel($U, $V, $c)