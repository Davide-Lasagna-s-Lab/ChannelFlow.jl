export PhysicalField

"""
    PhysicalField(grid, [T=Float64])

Allocate a physical scalar field with storage order `(y, x, z)`. The first
index is contiguous and the array size is `(Ny, Nx, Nz)`.
"""
struct PhysicalField{T<:AbstractFloat,
                     A<:DenseArray{T, 3},
                     G<:Grid} <: DenseArray{T, 3}
    data::A
    grid::G
    function PhysicalField(data::A, grid::G) where {T<:AbstractFloat,
                                                    A<:DenseArray{T, 3},
                                                    G<:Grid}
        size(data) == gridsize(grid) ||
            throw(DimensionMismatch("data must have size (Ny, Nx, Nz)"))
        return new{T, A, G}(data, grid)
    end

end

"""
    PhysicalField(grid, f, [T=Float64])

Construct a physical field by evaluating `f(x,y,z)` at every grid point. The
periodic coordinates cover `[0,Lx)` and `[0,Lz)`.
"""
function PhysicalField(grid::G, f::Function, ::Type{T}=Float64) where {
                       G<:Grid, T<:AbstractFloat}
    y, x, z = points(grid)
    return PhysicalField(T.(f.(x, y, z)), grid)
end

PhysicalField(grid::Grid, ::Type{T}=Float64) where {T<:AbstractFloat} =
    PhysicalField(grid, (x, y, z) -> zero(T), T)

"""Return an independent copy of `u` on the same grid."""
Base.copy(   u::PhysicalField) = PhysicalField(   copy(parent(u)), grid(u))

"""Allocate an uninitialised field with the same size and grid as `u`."""
Base.similar(u::PhysicalField) = PhysicalField(similar(parent(u)), grid(u))

"""Return the grid associated with `u`."""
grid(u::PhysicalField) = u.grid

"""Use linear indexing as the default indexing style."""
Base.IndexStyle(::Type{<:PhysicalField}) = Base.IndexLinear()

"""Return the value of `u` at indices `I`."""
Base.@propagate_inbounds function Base.getindex(u::PhysicalField,
                                                I::Integer...)
    @boundscheck checkbounds(u, I...)
    @inbounds val = u.data[I...]
    return val
end

"""Set the value of `u` at indices `I` and return `val`."""
Base.@propagate_inbounds function Base.setindex!(u::PhysicalField,
                                               val, I::Integer...)
    @boundscheck checkbounds(u, I...)
    @inbounds u.data[I...] = val
    return val
end

"""Return the dimensions of the underlying physical array."""
Base.size(u::PhysicalField) = size(parent(u))

"""Return the array storing the values of `u`."""
Base.parent(u::PhysicalField) = u.data

const PhysicalFieldStyle = Broadcast.ArrayStyle{PhysicalField}

"""Return the broadcast style used by `PhysicalField`."""
Base.BroadcastStyle(::Type{<:PhysicalField}) = PhysicalFieldStyle()

"""Evaluate a broadcast expression directly into `dest`."""
@inline function Base.Broadcast.materialize!(dest::PhysicalField,
                                               bc::Broadcast.Broadcasted{<:PhysicalFieldStyle})
    bc_ = Broadcast.flatten(bc)
    @simd for i in eachindex(dest)
        @inbounds dest[i] = bc_[i]
    end
    return dest
end
