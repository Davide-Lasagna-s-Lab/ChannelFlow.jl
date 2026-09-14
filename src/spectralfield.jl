using DecomposedArrays
using FDGrids

import MPI

export SpectralField

struct SpectralField{T<:AbstractFloat,
                     A<:DenseArray{Complex{T}, 3},
                     G<:FDGrid} <: DenseArray{Complex{T}, 3}
    data::A                        # the underlying storage
    grid::G                        # an object describing the grid in the wall normal direction
    domainsize::NTuple{2, Float64} # domain size along the homogeneous dimensions
    SpectralField(data::A,
            domainsize::NTuple{2, Reak},
                  grid::G) where {T,
                                  G<:Grid,
                                  A<:DenseArray{Complex{T}, 3}} =
        new{T, A, G}(data, grid, Float64.(domainsize))
end

# constructors
function SpectralField(comm::MPI.Comm,
                   gridsize::NTuple{3, Integer},
                 domainsize::NTuple{2, Real},
                       grid::FDGrid,
                           ::Type{T<:AbstractFloat} = Float64) where {T}
    # size of global data after transform
    gsize = (gridsize[1], gridsize[2]>>1+1, gridsize[3])
    return SpectralField(SlabArray(comm, gsize, Complex{T}, 3), domainsize, grid)
end

# Accessor functions
grid(U::SpectralField) = U.grid
domainsize(U::SpectralField) = U.domainsize
domainsize(U::SpectralField, i::Int) = U.domainsize[i]

# what is this for ?
gridsize(U::SpectralField{T, <:DecomposedArray}) where {T} = globalsize(parent(U))
gridsize(U::SpectralField{T, <:DenseArray})      where {T} =       size(parent(U))

# copy and similar
Base.copy(   U::SpectralField) = SpectralField(   copy(parent(U)), grid(U))
Base.similar(U::SpectralField) = SpectralField(similar(parent(U)), grid(U))

# indexing by a scalar by default
Base.IndexStyle(::Type{<:SpectralField}) = Base.IndexLinear()

# indexing
Base.@propagate_inbounds function Base.getindex(U::SpectralField, I::Integer...)
    @boundscheck checkbounds(U, I...)
    @inbounds val = U.data[I...]
    return val
end

Base.@propagate_inbounds function Base.setindex!(U::SpectralField, val, I::Integer...)
    @boundscheck checkbounds(U, I...)
    @inbounds U.data[I...] = val
    return val
end

# size of data local to this processor
Base.size(U::SpectralField) = size(parent(U))

# get underlying storage
Base.parent(U::SpectralField) = U.data

# overload the broadcasting machinery
const SpectralFieldStyle = Broadcast.ArrayStyle{SpectralField}
Base.BroadcastStyle(::Type{<:SpectralField}) = SpectralFieldStyle()

@inline function Base.Broadcast.materialize!(dest::SpectralField,
                                               bc::Broadcast.Broadcasted{<:SpectralFieldStyle})
    bc_ = Broadcast.flatten(bc)
    @simd for i in eachindex(dest)
        @inbounds dest[i] = bc_[i]
    end
    return dest
end