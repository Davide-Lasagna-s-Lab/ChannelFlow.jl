#//////////////////////////////////////////////////////////////////////////////#
#///                         CHEBYSHEV TRANSFORM PLANS                       ///#
#//////////////////////////////////////////////////////////////////////////////#

"""Dense wall-normal transform plan; endpoint input weights are in the matrix."""
struct GEMMChebyshevPlan{A}
    matrix::A
end

"""FFTW DCT-I plan with a compile-time inverse endpoint-weighting flag."""
struct FFTWChebyshevPlan{INVERSE, P}
    plan::P
end

#//////////////////////////////////////////////////////////////////////////////#
#///                        SPECTRAL MATRIX VIEW                            ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    SpectralMatrixView(U::SpectralField)

Expose contiguous `(y, kx, kz)` field storage as a `(Ny, Nxh*Nz)` matrix
without copying. This lightweight view gives BLAS access to every Fourier
column without allocating a ReshapedArray on each transform call.
"""
struct SpectralMatrixView{T, F<:SpectralField} <: AbstractMatrix{T}
    field::F

    function SpectralMatrixView(U::SpectralField{T, A}) where {T, A<:DenseArray}
        return new{eltype(U), typeof(U)}(U)
    end
end

function Base.size(A::SpectralMatrixView)
    Ny, Nxh, Nz = size(A.field)
    return (Ny, Nxh*Nz)
end
Base.strides(A::SpectralMatrixView) = (1, size(A, 1))
Base.IndexStyle(::Type{<:SpectralMatrixView}) = IndexLinear()
Base.getindex(A::SpectralMatrixView, i::Int) = parent(A.field)[i]
Base.setindex!(A::SpectralMatrixView, value, i::Int) = (parent(A.field)[i] = value)
Base.unsafe_convert(::Type{Ptr{T}}, A::SpectralMatrixView{T}) where {T} = pointer(parent(A.field))

#//////////////////////////////////////////////////////////////////////////////#
#///                          PLAN CONSTRUCTION                             ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    plan_cheb(U::SpectralField, backend=:auto; flags=FFTW.EXHAUSTIVE, timelimit=FFTW.NO_TIMELIMIT)

Plan the forward, unnormalised wall-normal DCT-I for the layout and type of
`U`. Execute with `mul!(dest, plan, src)` using distinct matching buffers.
The enclosing ForwardFFT! applies coefficient normalization afterwards.
Select `:gemm` or `:fftw` explicitly. The default `:auto` uses GEMM through
Ny=35 and FFTW above that measured crossover. FFTW planning may overwrite
the supplied `U`; `flags` and `timelimit` apply only to FFTW.
"""
plan_cheb(U::SpectralField, backend::Symbol=:auto; kwargs...) =
    _plan_cheb(U, Val(false), backend; kwargs...)

"""
    plan_icheb(U::SpectralField, backend=:auto; flags=FFTW.EXHAUSTIVE, timelimit=FFTW.NO_TIMELIMIT)

Plan inverse wall-normal evaluation, including doubled endpoint inputs.
Execute with `mul!(dest, plan, src)` using distinct matching buffers.
The result is twice the evaluated Chebyshev series: InverseFFT! applies
its remaining factor of 1/2 during Fourier padding. Backend selection and
planning side effects match [`plan_cheb`](@ref).
"""
plan_icheb(U::SpectralField, backend::Symbol=:auto; kwargs...) =
    _plan_cheb(U, Val(true), backend; kwargs...)

function _plan_cheb(U::SpectralField, ::Val{INVERSE}, backend::Symbol;
                   flags::Integer=FFTW.EXHAUSTIVE,
                   timelimit::Real=FFTW.NO_TIMELIMIT) where {INVERSE}
    Ny = size(U, 1)
    backend in (:auto, :gemm, :fftw) ||
        throw(ArgumentError("backend must be :auto, :gemm or :fftw"))
    if backend == :gemm || (backend == :auto && Ny <= 35)
        T = eltype(U)
        matrix = T[2cospi((i-1)*(j-1)/(Ny-1)) for i = 1:Ny, j = 1:Ny]
        # Forward DCT-I counts endpoint inputs once; the inverse counts them twice.
        if !INVERSE
            @views matrix[:, 1] ./= 2
            @views matrix[:, end] ./= 2
        end
        return GEMMChebyshevPlan(matrix)
    end
    plan = FFTW.plan_r2r!(parent(U), FFTW.REDFT00, (1,);
                         flags=flags, timelimit=timelimit)
    return FFTWChebyshevPlan{INVERSE, typeof(plan)}(plan)
end

#//////////////////////////////////////////////////////////////////////////////#
#///                            PLAN EXECUTION                              ///#
#//////////////////////////////////////////////////////////////////////////////#

"""Apply the dense plan, preserving `src`; source and destination must not alias."""
function LinearAlgebra.mul!(dest::SpectralField, plan::GEMMChebyshevPlan, src::SpectralField)
    LinearAlgebra.BLAS.gemm!('N', 'N', true, plan.matrix, SpectralMatrixView(src), false, SpectralMatrixView(dest))
    return dest
end

"""Apply the FFTW plan to distinct buffers, preserving the source coefficients."""
function LinearAlgebra.mul!(dest::SpectralField, plan::FFTWChebyshevPlan{INV}, src::SpectralField) where {INV}
    copyto!(parent(dest), parent(src))
    if INV
        @views parent(dest)[1, :, :] .*= 2
        @views parent(dest)[end, :, :] .*= 2
    end
    FFTW.unsafe_execute!(plan.plan, parent(dest), parent(dest))
    return dest
end
