#//////////////////////////////////////////////////////////////////////////////#
#///                      FOURIER AND CHEBYSHEV PLANS                       ///#
#//////////////////////////////////////////////////////////////////////////////#

# DCT-I is a complex FFT of the even extension. This works for complex
# Fourier amplitudes without splitting real and imaginary parts.
struct CUDAChebyshevPlan{INVERSE,A,P}
    work::A # B × 2(Ny-1) complex even-extension workspace
    plan::P # in-place cuFFT along the coefficient axis
end

function CF._plan_cheb(U::CuSpectral, ::Val{INV}; flags=nothing, timelimit=nothing) where {INV}
    Ny = size(U, 3)
    work = CUDA.zeros(eltype(U), size(U, 1)*size(U, 2), 2(Ny-1))
    plan = CUDA.CUFFT.plan_fft!(work, (2,))
    return CUDAChebyshevPlan{INV,typeof(work),typeof(plan)}(work, plan)
end

# Fill the even extension in one pass. Each thread owns one coefficient;
# adjacent threads access adjacent Fourier systems. Endpoint doubling for
# inverse evaluation is folded into the same write, preserving src.
function _cheb_extend!(work, a, inverse)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    if i <= length(work)
        B, Ny = size(a)
        s = (i-1)%B + 1
        j = (i-1)÷B + 1
        k = j <= Ny ? j : 2Ny-j
        @inbounds work[i] = inverse && (k == 1 || k == Ny) ? 2a[s,k] : a[s,k]
    end
    return nothing
end

function LinearAlgebra.mul!(
    dest::CuSpectral,
    p::CUDAChebyshevPlan{INV},
    src::CuSpectral,
) where {INV}
    Ny = size(src, 3)
    @cuda threads=256 blocks=cld(length(p.work),256) _cheb_extend!(
        p.work, CF.spectralmatrix(src), INV)
    mul!(p.work, p.plan, p.work)
    @views CF.spectralmatrix(dest) .= p.work[:, 1:Ny]
    return dest
end

function CF.ForwardFFT!(u::CuPhysical{T}; flags=nothing, timelimit=nothing) where {T}
    g = CF.grid(u)
    Nx, Nz, Ny = CF.physicalsize(g, CF.Padded())
    input = CUDA.zeros(T, Nx, Nz, Ny)
    padded = CF.SpectralField(CUDA.zeros(Complex{T}, CF.spectralsize(g, CF.Padded())), g)
    resolved = CF.SpectralField(CUDA.zeros(Complex{T}, CF.spectralsize(g, CF.NotPadded())), g)
    plan = CUDA.CUFFT.plan_rfft(input, CF.FFT_DIMS)
    return CF.ForwardFFT!(
        plan,
        CF.plan_cheb(resolved),
        padded,
        resolved,
        inv(T(Nx*Nz*(Ny-1))),
    )
end

function CF.InverseFFT!(U::CuSpectral{T}; flags=nothing, timelimit=nothing) where {T}
    g = CF.grid(U)
    padded = CF.SpectralField(CUDA.zeros(Complex{T}, CF.spectralsize(g, CF.Padded())), g)
    resolved = CF.SpectralField(CUDA.zeros(Complex{T}, CF.spectralsize(g, CF.NotPadded())), g)
    Nx = CF.physicalsize(g, CF.Padded())[1]
    plan = CUDA.CUFFT.plan_brfft(parent(padded), Nx, CF.FFT_DIMS)
    return CF.InverseFFT!(plan, CF.plan_icheb(resolved), padded, resolved)
end

# Normalization, endpoint weights and Nyquist filtering share one pass.
# This avoids revisiting endpoint/Nyquist planes in four additional kernels.
function _normalize_forward!(a, normalization, xnyquist, znyquist)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    if i <= length(a)
        nx, nz, ny = size(a)
        ix = (i-1)%nx + 1
        iz = ((i-1)÷nx)%nz + 1
        iy = (i-1)÷(nx*nz) + 1
        endpoint = iy == 1 || iy == ny ? 0.5 : 1.0
        @inbounds a[i] = ix == xnyquist || iz == znyquist ?
            zero(eltype(a)) : a[i]*(normalization*endpoint)
    end
    return nothing
end

function CF.normalize_forward!(U::CuSpectral, normalization)
    Nx, Nz, _ = CF.physicalsize(CF.grid(U), CF.NotPadded())
    xnyquist = iseven(Nx) ? (Nx >> 1)+1 : 0
    znyquist = iseven(Nz) ? (Nz >> 1)+1 : 0
    @cuda threads=256 blocks=cld(length(U),256) _normalize_forward!(
        parent(U), normalization, xnyquist, znyquist)
    return U
end
