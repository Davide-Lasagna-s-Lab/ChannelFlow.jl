# Modal solvers are separated by their mathematical role. All definitions
# remain in ChannelFlow; these files do not introduce nested modules.
include("solvers/influence.jl")
include("solvers/meanmode.jl")
include("solvers/fourierstokes.jl")
