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

"""Copy all three components, retaining the grid and owning independent data."""
Base.copy(U::VectorField) = VectorField(map(copy, U.components))

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

# Scalar RK coefficients preserve the component-wise vector-field broadcast.
Base.BroadcastStyle(::VectorFieldStyle, ::Base.Broadcast.DefaultArrayStyle{0}) = VectorFieldStyle()

"""Fuse the scalar broadcast within each component; scalar arguments are shared."""
function Base.Broadcast.materialize!(dest::VectorField,
                                       bc::Base.Broadcast.Broadcasted{<:VectorFieldStyle})
    return copyto!(dest, bc)
end

"""Evaluate a component-wise broadcast, including unpacked Flows.Coupled expressions."""
function Base.copyto!(dest::VectorField,
                       bc::Base.Broadcast.Broadcasted)
    bc_ = Base.Broadcast.flatten(bc)
    for i = 1:3
        args = map(arg -> arg isa VectorField ? arg[i] : arg, bc_.args)
        dest[i] .= bc_.f.(args...)
    end
    return dest
end
