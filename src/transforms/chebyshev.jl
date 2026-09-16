#//////////////////////////////////////////////////////////////////////////////#
#///                      CHEBYSHEV TRANSFORM BACKENDS                      ///#
#//////////////////////////////////////////////////////////////////////////////#

# Dense BLAS multiplication is faster than many short DCT-I transforms at
# the wall-normal resolutions used by the present DNS. Larger systems retain
# FFTW's asymptotically cheaper algorithm.
struct DenseChebyshevTransform{A}
    matrix::A
end

# Allocation-free two-dimensional view of contiguous `(y, kx, kz)` storage.
# BLAS sees every Fourier column as one matrix column without constructing a
# `ReshapedArray` at each transform call.
struct SpectralMatrixView{T, A<:DenseArray{T, 3}} <: AbstractMatrix{T}
    data::A
end

Base.size(A::SpectralMatrixView) =
    (size(A.data, 1), length(A.data) ÷ size(A.data, 1))
Base.strides(A::SpectralMatrixView) = (1, size(A, 1))
Base.IndexStyle(::Type{<:SpectralMatrixView}) = IndexLinear()
Base.getindex(A::SpectralMatrixView, i::Int) = A.data[i]
Base.setindex!(A::SpectralMatrixView, value, i::Int) = (A.data[i] = value)
Base.unsafe_convert(::Type{Ptr{T}}, A::SpectralMatrixView{T}) where {T} = pointer(A.data)

"""
    chebyshev_transform(prototype; inverse=false, flags, timelimit)

Build the wall-normal DCT-I backend for resolved Fourier columns. Use a dense
BLAS matrix through Ny=35 (the measured small-grid crossover) and FFTW above
that size. The inverse matrix includes endpoint input weights; output scaling
is applied by the enclosing Fourier--Chebyshev transform in either direction.
"""
function chebyshev_transform(prototype::SpectralField;
                               inverse::Bool=false,
                                 flags::Integer=FFTW.EXHAUSTIVE,
                             timelimit::Real=FFTW.NO_TIMELIMIT)
    Ny = size(prototype, 1)
    if Ny <= 35
        T = eltype(prototype)
        matrix = T[(inverse ? 2 : (j == 1 || j == Ny ? 1 : 2)) *
                   cospi((i-1)*(j-1)/(Ny-1)) for i = 1:Ny, j = 1:Ny]
        return DenseChebyshevTransform(matrix)
    end
    return FFTW.plan_r2r!(parent(prototype), FFTW.REDFT00, (1,);
                          flags=flags, timelimit=timelimit)
end

"""Apply a forward wall-normal transform, preserving `src`; buffers must be distinct."""
function forward_chebyshev!(     dest::SpectralField,
                            transform::DenseChebyshevTransform,
                                  src::SpectralField)
    LinearAlgebra.BLAS.gemm!('N', 'N', true, transform.matrix,
                             SpectralMatrixView(parent(src)), false,
                             SpectralMatrixView(parent(dest)))
    return dest
end

function forward_chebyshev!(dest::SpectralField, plan, src::SpectralField)
    copyto!(parent(dest), parent(src))
    FFTW.unsafe_execute!(plan, parent(dest), parent(dest))
    return dest
end

# The inverse matrix already contains its endpoint input weights.
inverse_chebyshev!(     dest::SpectralField,
                  transform::DenseChebyshevTransform,
                        src::SpectralField) = forward_chebyshev!(dest, transform, src)

"""Apply FFTW DCT-I after doubling endpoint coefficients in the destination buffer."""
function inverse_chebyshev!(dest::SpectralField, plan, src::SpectralField)
    copyto!(parent(dest), parent(src))
    @views parent(dest)[1, :, :] .*= 2
    @views parent(dest)[end, :, :] .*= 2
    FFTW.unsafe_execute!(plan, parent(dest), parent(dest))
    return dest
end
