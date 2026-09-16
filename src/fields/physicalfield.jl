export PhysicalField

#//////////////////////////////////////////////////////////////////////////////#
#///                PHYSICAL FIELD STORAGE AND CONSTRUCTION                 ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    PhysicalField(data, grid)

Wrap a resolved or 3/2-padded physical scalar field with storage order
`(y, x, z)`. The first index is contiguous.
"""
struct PhysicalField{T<:AbstractFloat,
                     A<:DenseArray{T, 3},
                     G<:Grid} <: DenseArray{T, 3}
    data::A
    grid::G
    function PhysicalField(data::A, grid::G) where {T<:AbstractFloat,
                                                    A<:DenseArray{T, 3},
                                                    G<:Grid}
        size(data) in (physicalsize(grid, NotPadded()), physicalsize(grid, Padded())) ||
            throw(DimensionMismatch("data must have the resolved or padded grid size"))
        return new{T, A, G}(data, grid)
    end

end

"""
    PhysicalField(grid, f, [T=Float64])

Construct a physical field by evaluating `f(x,y,z)` at every grid point. The
periodic coordinates cover `[0,Lx)` and `[0,Lz)`.
"""
PhysicalField(grid::Grid, f::Function, ::Type{T}=Float64) where {T<:AbstractFloat} =
    PhysicalField(grid, f, NotPadded(), T)

"""Sample `f(x,y,z)` on the grid with the selected padding."""
function PhysicalField(grid::Grid, f::Function, tag::Union{Padded, NotPadded},
                       ::Type{T}=Float64) where {T<:AbstractFloat}
    y, x, z = points(grid, tag)
    return PhysicalField(T.(f.(x, y, z)), grid)
end

"""Allocate a zero physical field on the resolved grid, with real element type `T`."""
PhysicalField(grid::Grid, ::Type{T}=Float64) where {T<:AbstractFloat} =
    PhysicalField(grid, (x, y, z) -> zero(T), T)

"""
    PhysicalField(grid, tag::Union{Padded, NotPadded}, [T=Float64])

Allocate a zero physical field with the selected padding and real element
type `T`, in storage order `(y, x, z)`.
"""
PhysicalField(grid::Grid, tag::Union{Padded, NotPadded},
              ::Type{T}=Float64) where {T<:AbstractFloat} =
    PhysicalField(zeros(T, physicalsize(grid, tag)), grid)

#//////////////////////////////////////////////////////////////////////////////#
#///                     ARRAY INTERFACE AND ALLOCATION                     ///#
#//////////////////////////////////////////////////////////////////////////////#

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

#//////////////////////////////////////////////////////////////////////////////#
#///                         IN-PLACE BROADCASTING                          ///#
#//////////////////////////////////////////////////////////////////////////////#

const PhysicalFieldStyle = Broadcast.ArrayStyle{PhysicalField}

"""Return the broadcast style used by `PhysicalField`."""
Base.BroadcastStyle(::Type{<:PhysicalField}) = PhysicalFieldStyle()

# Use Julia's standard materialize!/copyto! path: it validates broadcast
# axes and chooses efficient iteration, including singleton expansion.
