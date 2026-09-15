module CanonicalFlows

import Flows

include("grids.jl")

# basic scalar fields
include("fields/physicalfield.jl")
include("fields/spectralfield.jl")
include("fields/abstractfield.jl")

include("indexing.jl")
include("operators.jl")
include("fields/vectorfield.jl")
include("fields/gradientfield.jl")
include("state.jl")
include("ffts.jl")
include("nonlinear.jl")
include("helmoltz.jl")
include("channelflow.jl")
include("initialization.jl")
include("timesteppers/cnrk2.jl")
include("timesteppers/channel.jl")

end
