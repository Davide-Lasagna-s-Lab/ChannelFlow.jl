#//////////////////////////////////////////////////////////////////////////////#
#///                      FOURIER AND CHEBYSHEV PLANS                       ///#
#//////////////////////////////////////////////////////////////////////////////#

# DCT-I is a complex FFT of the even extension. This works for complex
# Fourier amplitudes without splitting real and imaginary parts.
struct CUDAChebyshevPlan{INVERSE,A,P}
    work::A
    plan::P
end

function CF._plan_cheb(U::CuSpectral, ::Val{INV}, backend::Symbol; kwargs...) where {INV}
    Ny = size(U, 3)
    if backend == :gemm
        matrix = ComplexF64[2cospi((i-1)*(j-1)/(Ny-1)) for i = 1:Ny, j = 1:Ny]
        if !INV
            matrix[:, 1] ./= 2
            matrix[:, end] ./= 2
        end
        return CF.GEMMChebyshevPlan(CuArray(matrix))
    end
    backend in (:cufft, :fftw, :auto) ||
        throw(ArgumentError("CUDA Chebyshev backend must be :cufft or :gemm"))
    work = CUDA.zeros(eltype(U), size(U, 1)*size(U, 2), 2(Ny-1))
    plan = CUDA.CUFFT.plan_fft!(work, (2,))
    return CUDAChebyshevPlan{INV,typeof(work),typeof(plan)}(work, plan)
end

function LinearAlgebra.mul!(
    dest::CuSpectral,
    p::CUDAChebyshevPlan{INV},
    src::CuSpectral,
) where {INV}
    Ny = size(src, 3)
    a = CF.spectralmatrix(src)
    @views p.work[:, 1:Ny] .= a
    @views p.work[:, (Ny+1):end] .= a[:, (Ny-1):-1:2]
    if INV
        @views p.work[:, 1] .*= 2
        @views p.work[:, Ny] .*= 2
    end
    mul!(p.work, p.plan, p.work)
    @views CF.spectralmatrix(dest) .= p.work[:, 1:Ny]
    return dest
end

function CF.ForwardFFT!(u::CuPhysical{T}; chebbackend = :cufft, kwargs...) where {T}
    g = CF.grid(u)
    Nx, Nz, Ny = CF.physicalsize(g, CF.Padded())
    input = CUDA.zeros(T, Nx, Nz, Ny)
    padded = CF.SpectralField(CUDA.zeros(Complex{T}, CF.spectralsize(g, CF.Padded())), g)
    resolved = CF.SpectralField(CUDA.zeros(Complex{T}, CF.spectralsize(g, CF.NotPadded())), g)
    plan = CUDA.CUFFT.plan_rfft(input, CF.FFT_DIMS)
    return CF.ForwardFFT!(
        plan,
        CF.plan_cheb(resolved, chebbackend),
        padded,
        resolved,
        inv(T(Nx*Nz*(Ny-1))),
    )
end

function CF.InverseFFT!(U::CuSpectral{T}; chebbackend = :cufft, kwargs...) where {T}
    g = CF.grid(U)
    padded = CF.SpectralField(CUDA.zeros(Complex{T}, CF.spectralsize(g, CF.Padded())), g)
    resolved = CF.SpectralField(CUDA.zeros(Complex{T}, CF.spectralsize(g, CF.NotPadded())), g)
    Nx = CF.physicalsize(g, CF.Padded())[1]
    plan = CUDA.CUFFT.plan_brfft(parent(padded), Nx, CF.FFT_DIMS)
    return CF.InverseFFT!(plan, CF.plan_icheb(resolved, chebbackend), padded, resolved)
end

function CF.normalize_forward!(U::CuSpectral, normalization)
    parent(U) .*= normalization
    @views parent(U)[:, :, 1] .*= 0.5
    @views parent(U)[:, :, end] .*= 0.5
    CF.zero_nyquist!(U)
    return U
end
