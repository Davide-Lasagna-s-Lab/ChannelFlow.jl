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

"""Treat the three components as the broadcast axes of `U`."""
Base.size(::VectorField) = (3,)
Base.axes(::VectorField) = (Base.OneTo(3),)
Base.broadcastable(U::VectorField) = U

struct VectorFieldStyle <: Base.Broadcast.AbstractArrayStyle{1} end
Base.BroadcastStyle(::Type{<:VectorField}) = VectorFieldStyle()

"""Fuse the scalar broadcast within each component; scalar arguments are shared."""
function Base.Broadcast.materialize!(dest::VectorField,
                                       bc::Base.Broadcast.Broadcasted{<:VectorFieldStyle})
    bc_ = Base.Broadcast.flatten(bc)
    for i = 1:3
        args = map(arg -> arg isa VectorField ? arg[i] : arg, bc_.args)
        dest[i] .= bc_.f.(args...)
    end
    return dest
end
