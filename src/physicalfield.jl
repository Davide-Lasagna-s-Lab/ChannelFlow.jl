using DecomposedArrays
using FDGrids
import Base.Threads: @threads

import MPI

export PhysicalField

struct PhysicalField{T<:AbstractFloat,
                     A<:DenseArray{T, 3}} <: DenseArray{T, 3}
    data::A
    PhysicalField(data::A) where {T,
                                  A<:DenseArray{T, 3}} =
        new{T, A}(data)
end

"""
PhysicalField(comm::MPI.Comm,
          gridsize::NTuple{3, Int},
                  ::Type{T<:AbstractFloat} = Float64) where {T}

Construct a PhysicalField object of global size `gridsize` with a floating point data
type `T` that defaults to `Float64`. Data is decomposed in slabs distributed over
processors in the communicator `comm`. By default slabs are distributed along the third
dimension, corresponding to the wall normal direction.

Example
-------
    a = PhysicalField(MPI.COMM_WORLD, (1024, 512, 256), Float32)

Indexing
--------
A `PhysicalField` is indexable with three indices as

    a[i, j, k]

By default, the first two indices are the two wall parallel directions, where
the first is typically the streamwise direction. The third index defines the
wall normal direction.
"""
PhysicalField(comm::MPI.Comm,
          gridsize::NTuple{3, Int},
                  ::Type{T<:AbstractFloat} = Float64) where {T} =
    PhysicalField(SlabArray(comm, gridsize, T, 3))

# copy and similar
Base.copy(   U::PhysicalField) = PhysicalField(   copy(parent(U)))
Base.similar(U::PhysicalField) = PhysicalField(similar(parent(U)))

# index by a scalar by default
Base.IndexStyle(::Type{<:PhysicalField}) = Base.IndexLinear()

# indexing
Base.@propagate_inbounds function Base.getindex(u::PhysicalField, 
                                                I::Integer...)
    @boundscheck checkbounds(u, I...)
    @inbounds val = u.data[I...]
    return val
end

Base.@propagate_inbounds function Base.setindex!(u::PhysicalField,
                                               val, I::Integer...)
    @boundscheck checkbounds(u, I...)
    @inbounds u.data[I...] = val
    return val
end

# size of local data
Base.size(u::PhysicalField) = size(parent(u))

# get underlying storage
Base.parent(u::PhysicalField) = u.data

# overload the broadcasting machinery
const PhysicalFieldStyle = Broadcast.ArrayStyle{PhysicalField}
Base.BroadcastStyle(::Type{<:PhysicalField}) = PhysicalFieldStyle()

@inline function Base.Broadcast.materialize!(dest::PhysicalField,
                                               bc::Broadcast.Broadcasted{<:PhysicalFieldStyle})
    bc_ = Broadcast.flatten(bc)
    @simd for i in eachindex(dest)
        @inbounds dest[i] = bc_[i]
    end
    return dest
end