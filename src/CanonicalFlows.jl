module CanonicalFlows

# basic scalar fields
include("physicalfield.jl")
include("spectralfield.jl")

# generic field type
const AbstractField = Union{SpectralField, PhysicalField}

include("indexing.jl")
include("operators.jl")
include("explicit.jl")
include("vectorfield.jl")
include("gradientfield.jl")
# include("ffts.jl")

end