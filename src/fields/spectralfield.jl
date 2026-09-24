export SpectralField

#//////////////////////////////////////////////////////////////////////////////#
#///                SPECTRAL FIELD STORAGE AND CONSTRUCTION                 ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    SpectralField(data, grid)

Wrap Fourier--Chebyshev coefficients in storage order `(kx, kz, n)`.
Index `n+1` stores the ordinary coefficient multiplying `T_n` in the mapped
wall-normal coordinate; the third dimension contains coefficients, not nodal
values. The `x` spectrum is real-transform half-storage and `z` is full-storage.
Fourier padding changes only the first and second dimensions.
"""
struct SpectralField{T<:AbstractFloat,
                     A<:DenseArray{Complex{T}, 3},
                     G<:Grid} <: DenseArray{Complex{T}, 3}
    data::A
    grid::G

    SpectralField(data::A, grid::G) where {
        T<:AbstractFloat, A<:DenseArray{Complex{T}, 3}, G<:Grid} =
        new{T, A, G}(data, grid)
end

"""Allocate a zero spectral field on the resolved grid."""
SpectralField(grid::Grid, ::Type{T}=Float64) where {T<:AbstractFloat} =
    SpectralField(zeros(Complex{T}, spectralsize(grid, NotPadded())), grid)

#//////////////////////////////////////////////////////////////////////////////#
#///                     ARRAY INTERFACE AND ALLOCATION                     ///#
#//////////////////////////////////////////////////////////////////////////////#

"""Return the grid associated with `U`."""
grid(U::SpectralField) = U.grid

"""Copy coefficients into independent storage on the same grid."""
Base.copy(   U::SpectralField) = SpectralField(   copy(parent(U)), grid(U))
"""Allocate uninitialised coefficients with the same shape, type and grid."""
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

"""Return coefficient-array dimensions `(Nxh, Nz, Ny)`."""
Base.size(U::SpectralField) = size(parent(U))

"""Return the underlying coefficient array, without copying."""
Base.parent(U::SpectralField) = U.data

#//////////////////////////////////////////////////////////////////////////////#
#///                         IN-PLACE BROADCASTING                          ///#
#//////////////////////////////////////////////////////////////////////////////#

# overload the broadcasting machinery
const SpectralFieldStyle = Broadcast.ArrayStyle{SpectralField}
Base.BroadcastStyle(::Type{<:SpectralField}) = SpectralFieldStyle()

# Use Julia's standard materialize!/copyto! path: it validates broadcast
# axes and chooses efficient iteration, including singleton expansion.
