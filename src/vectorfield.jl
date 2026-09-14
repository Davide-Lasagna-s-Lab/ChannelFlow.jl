export VectorField

struct VectorField{F <: AbstractField}
    components::NTuple{3, F}
end

"""
    VectorField(u::AbstractField)

Allocate three independent components from the scalar-field prototype `u`.
"""
VectorField(u::F) where {F<:AbstractField} =
    VectorField((similar(u), similar(u), similar(u)))

"""Allocate a `VectorField` similar to `U`."""
Base.similar(U::VectorField) = VectorField(U[1])

"""
    Base.getindex(U::VectorField, i::Int)

Return the `i` component of the velocity field `U`.
"""
Base.getindex(U::VectorField, i::Int) = U.components[i]
