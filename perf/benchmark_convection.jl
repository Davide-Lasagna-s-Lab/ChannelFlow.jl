# Run with julia --startup-file=no --threads=1 --project=. perf/benchmark_convection.jl
# Compare the original field broadcast, native-array broadcast, and production
# contraction on the DNS padded grid. All outputs must match exactly; warm-up
# is excluded. Eleven batches of fifty calls reduce single-call timing noise.
using ChannelFlow, FFTW, Statistics, Random
const CF=ChannelFlow
function raw!(out,u,g)
    for i=1:3
        parent(out[i]) .= parent(u[1]).*parent(g[i,1]) .+ parent(u[2]).*parent(g[i,2]) .+ parent(u[3]).*parent(g[i,3])
    end
    out
end
# Preserve the old custom materializer explicitly after fixing the fields.
function old_materialize!(dest, bc)
    bc = Base.Broadcast.flatten(bc)
    @simd for k in eachindex(dest)
        @inbounds dest[k] = bc[k]
    end
    dest
end
function original!(out,u,g)
    b = Base.Broadcast.broadcasted
    @inbounds for i=1:3
        old_materialize!(out[i], b(+, b(*,u[1],g[i,1]),
                                     b(*,u[2],g[i,2]), b(*,u[3],g[i,3])))
    end
    out
end
function measure(f,out,u,g)
    f(out,u,g)
    samples=map(1:11) do _
        elapsed=@elapsed begin
            for _=1:50; f(out,u,g); end
        end
        1000elapsed/50
    end
    println(f,": median ms ",median(samples)," bytes ",@allocated(f(out,u,g)))
end
function main()
    grid=Grid(35,32,32,2π/1.14,2π/2.5)
    p=PhysicalField(zeros(physicalsize(grid,Padded())),grid)
    u=VectorField(p); g=CF.GradientField(p); out=VectorField(p)
    Random.seed!(42)
    for i=1:3
        randn!(parent(u[i]))
        for j=1:3; randn!(parent(g[i,j])); end
    end
    original!(out,u,g); expected=copy(out)
    for f in (original!,raw!,CF.dot!)
        f(out,u,g)
        @assert all(parent(out[i])==parent(expected[i]) for i=1:3)
        measure(f,out,u,g)
    end
end
main()
