# 
abstract type AbstractGrid end

struct CartesianGrid <: AbstractGrid
    x1::R
    x2::R
    x3::R
end