export SpectralField

struct SpectralField{T<:AbstractFloat,
                     A<:DenseArray{Complex{T}, 3},
                     G<:Grid} <: DenseArray{Complex{T}, 3}
    data::A
    grid::G

    SpectralField(data::A, grid::G) where {
        T<:AbstractFloat, A<:DenseArray{Complex{T}, 3}, G<:Grid} =
        new{T, A, G}(data, grid)
end

# Accessor functions
grid(U::SpectralField) = U.grid

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

# size of the spectral data
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
