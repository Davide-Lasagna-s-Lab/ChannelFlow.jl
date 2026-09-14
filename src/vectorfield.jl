struct VectorField{F <: AbstractField}
    components::NTuple{3, F}
end

VectorField(u::F) = VectorField(similar(u), similar(u), similar(u))

"""
    Base.getindex(U::VectorField, i::Int)

Return the `i` component of the velocity field `U`.
"""
Base.getindex(U::VectorField, i::Int) = U.components[i]


function div!(out::ScalarField{F}, u::VectorField{F}) where {F}
    out .= ddx1!(u) .+ ddx2!(u) .+ ddx3!(u)
end


"""
    outer!(out::GradientField, a::VectorField, b::VectorField)

Store in `out` the outer product of two vectors `a` and `b` defined in physical space.
"""
function outer!(out::GradientField{F},
                  a::VectorField{F},
                  b::VectorField{F}) where {T, SIZE, F<:PhysicalField{T, SIZE}}
    for i = 1:3, j=1:3
        out[i][j] .= a[i].*b[j]
    end
    return out
end

function dot!(out::VectorField, u::VectorField, grad::GradientField)
    @inbounds for i = 1:3
        out[i] .= a[1].*b[i][1] .+ a[2].*b[i][2] .+ a[3].*b[i][3]
    end
end

function div!(out::ScalarField, u::VectorField)
    out .= ddx!(u) .+ ddz!(u) .+ ddy!(u)
end


function div!(out::VectorField, u::GradientField)
    @inbounds for i = 1:3
        div!(out[i], u[i])
    end
end

function ddx!(GRAD::GradientField{FT},
                 U::VectorField{FT}) where {FT<:FTField}
    ddx!(GRAD[1][1], U[1])
    ddx!(GRAD[1][2], U[2])
    ddx!(GRAD[1][3], U[3])
end
