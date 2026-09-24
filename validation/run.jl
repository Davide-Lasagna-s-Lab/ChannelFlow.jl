# Run from the repository root: julia --project=. validation/run.jl
# Optional argument selects viscous_decay, tollmien_schlichting or waleffe_equilibrium.
using ChannelFlow, FFTW, LinearAlgebra, Test
using ChebyshevHelmoltzSolvers: diff!
import Flows
const CF = ChannelFlow
include("../test/helpers.jl")

# CSV output comes from the same numerical cases and assertions used by CI.
const output = get(ENV, "CHANNEL_VALIDATION_OUTPUT", joinpath(@__DIR__, "results"))
mkpath(output)
const recorded = Set{String}()
function record_validation(name; values...)
    path = joinpath(output, name * ".csv")
    first = !(name in recorded)
    open(path, first ? "w" : "a") do io
        first && println(io, join(keys(values), ','))
        println(io, join(Base.values(values), ','))
    end
    push!(recorded, name)
end
open(joinpath(output, "environment.txt"), "w") do io
    println(io, "Julia ", VERSION)
    println(io, "Solver revision: ", readchomp(`git -C $(@__DIR__) rev-parse HEAD`))
    println(io, "Working-tree changes: ", readchomp(`git -C $(@__DIR__) status --porcelain`))
    println(io, "FFTW threads: 1; BLAS threads: 1; precision: Float64/ComplexF64")
end
FFTW.set_num_threads(1)
BLAS.set_num_threads(1)
cases = isempty(ARGS) ? ["viscous_decay", "tollmien_schlichting", "waleffe_equilibrium"] : ARGS
for name in cases
    name in ("viscous_decay", "tollmien_schlichting", "waleffe_equilibrium") || error("Unknown case: $name")
    include(name * ".jl")
end
