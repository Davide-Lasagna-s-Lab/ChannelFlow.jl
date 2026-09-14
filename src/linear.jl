# add method to this function
import Flows
import LinearAlgebra

struct ImplicitTerm
    Re::Float64
end

# Methods to satisfy the Flows interface
function LinearAlgebra.mul!(OUT::FT,
                              L::ImplicitTerm,
                              U::FT) where {FT<:FTField}
    laplacian!(OUT, U)
    OUT .*= 1/L.Re
    return OUT
end


function Flows.ImcA!(L::ImplicitTerm,
                     c::Real,
                     Y::FT,
                     OUT::FT) where {FT<:FTField} 
    invlaplacian!(OUT, Y, c/L.Re)
    return OUT
end