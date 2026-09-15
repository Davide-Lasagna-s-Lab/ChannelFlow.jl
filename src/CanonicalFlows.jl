module CanonicalFlows

import Flows

include("grids.jl")

# basic scalar fields
include("physicalfield.jl")
include("spectralfield.jl")

# generic field type
const AbstractField = Union{SpectralField, PhysicalField}

include("indexing.jl")
include("operators.jl")
include("vectorfield.jl")
include("gradientfield.jl")
include("ffts.jl")
include("nonlinear.jl")
include("helmoltz.jl")
include("projection.jl")
include("timestepping.jl")
include("channelflow.jl")

end
