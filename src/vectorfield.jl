export VectorField

struct VectorField{F <: AbstractField}
    components::NTuple{3, F}
end

"""
    VectorField(u::AbstractField)

Allocate three independent, zero-initialised components with the size, element
type and grid of the scalar-field prototype `u`. Its values are not copied.
"""
VectorField(u::F) where {F<:AbstractField} =
    VectorField(ntuple(_ -> fill!(similar(u), zero(eltype(u))), 3))

"""Allocate independent, uninitialised components with the same sizes and grids."""
Base.similar(U::VectorField) = VectorField(map(similar, U.components))

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
