#//////////////////////////////////////////////////////////////////////////////#
#///                        SPECTRAL DIFFERENTIATION                       /// #
#//////////////////////////////////////////////////////////////////////////////#

function _fourier!(out, a, direction, add, scale)
    i = (blockIdx().x-1)*blockDim().x+threadIdx().x
    if i <= length(a)
        nx, nz = size(a, 1), size(a, 2)
        ix = (i-1)%nx
        iz = ((i-1)÷nx)%nz
        mode = direction == 1 ? ix : (iz <= nz÷2 ? iz : iz-nz)
        value = im*(scale*mode)*a[i]
        out[i] = add ? out[i]+value : value
    end
    return
end
function CF.ddx1!(out::CuSpectral, U::CuSpectral, add::Bool = false)
    scale = 2π/CF.domainsize(CF.grid(U))[1]
    @cuda threads=256 blocks=cld(length(U), 256) _fourier!(parent(out), parent(U), 1, add, scale)
    return out
end
function CF.ddx3!(out::CuSpectral, U::CuSpectral, add::Bool = false)
    scale = 2π/CF.domainsize(CF.grid(U))[3]
    @cuda threads=256 blocks=cld(length(U), 256) _fourier!(parent(out), parent(U), 3, add, scale)
    return out
end

# One thread follows the coefficient recurrence for one Fourier system.
# Adjacent threads access adjacent systems, giving coalesced memory accesses.
function _derivative!(out, a, add, laplace, nx, nz, α², β²)
    s = (blockIdx().x-1)*blockDim().x+threadIdx().x
    if s <= size(a, 1)
        iz = (s-1)÷nx
        l = iz <= nz÷2 ? iz : iz-nz
        κ² = α²*((s-1)%nx)^2+β²*l^2
        an=dn=dn2=en=en2=zero(eltype(a))
        for j = size(a, 2):-1:1
            value = a[s, j]
            d = dn2+2j*an
            e = en2+2j*dn
            result = laplace ? (j==1 ? e/2 : e)-κ²*value : (j==1 ? d/2 : d)
            out[s, j] = add ? out[s, j]+result : result
            an=value
            dn2, dn=dn, d
            en2, en=en, e
        end
    end
    return
end
function CF.ddx2!(out::CuSpectral, U::CuSpectral, add::Bool = false)
    B = size(U, 1)*size(U, 2)
    @cuda threads=256 blocks=cld(B, 256) _derivative!(
        CF.spectralmatrix(out),
        CF.spectralmatrix(U),
        add,
        false,
        size(U, 1),
        size(U, 2),
        0.0,
        0.0,
    )
    return out
end
function CF.laplacian!(out::CuSpectral, U::CuSpectral)
    Lx, _, Lz=CF.domainsize(CF.grid(U))
    B = size(U, 1)*size(U, 2)
    @cuda threads=256 blocks=cld(B, 256) _derivative!(
        CF.spectralmatrix(out),
        CF.spectralmatrix(U),
        false,
        true,
        size(U, 1),
        size(U, 2),
        (2π/Lx)^2,
        (2π/Lz)^2,
    )
    return out
end
function CF._batchdiff!(out::CUDA.AnyCuMatrix, a::CUDA.AnyCuMatrix)
    B=size(a, 1)
    @cuda threads=256 blocks=cld(B, 256) _derivative!(out, a, false, false, B, 1, 0.0, 0.0)
    return out
end

function _curl!(o1, o2, o3, u1, u2, u3, α, β)
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    if i<=length(u1)
        nx, nz=size(u1, 1), size(u1, 2)
        ix=(i-1)%nx
        iz=((i-1)÷nx)%nz
        l=iz<=nz÷2 ? iz : iz-nz
        o1[i] -= im*l*β*u2[i]
        o2[i] = im*(l*β*u1[i]-ix*α*u3[i])
        o3[i] = im*ix*α*u2[i]-o3[i]
    end
    return
end
function CF.curl!(out::CF.VectorField{F}, U::CF.VectorField{F}) where {F<:CuSpectral}
    CF.ddx2!(out[1], U[3])
    CF.ddx2!(out[3], U[1])
    Lx, _, Lz=CF.domainsize(CF.grid(U[1]))
    @cuda threads=256 blocks=cld(length(U[1]), 256) _curl!(
        map(parent, out.components)...,
        map(parent, U.components)...,
        2π/Lx,
        2π/Lz,
    )
    return out
end

#//////////////////////////////////////////////////////////////////////////////#
#///                       ALLOCATING DEVICE TRANSFORMS                     ///#
#//////////////////////////////////////////////////////////////////////////////#

function CF.FFT(u::CuPhysical{T}; kwargs...) where {T}
    U=CF.SpectralField(
        CUDA.zeros(Complex{T}, CF.spectralsize(CF.grid(u), CF.NotPadded())),
        CF.grid(u),
    )
    return CF.ForwardFFT!(u; kwargs...)(U, u)
end
function CF.IFFT(U::CuSpectral{T}; kwargs...) where {T}
    u=CF.PhysicalField(CUDA.zeros(T, CF.physicalsize(CF.grid(U), CF.Padded())), CF.grid(U))
    return CF.InverseFFT!(U; kwargs...)(u, U)
end
function CF.FFT(u::CF.VectorField{F}; kwargs...) where {F<:CuPhysical}
    return CF.VectorField(map(f -> CF.FFT(f; kwargs...), u.components))
end
function CF.IFFT(U::CF.VectorField{F}; kwargs...) where {F<:CuSpectral}
    return CF.VectorField(map(f -> CF.IFFT(f; kwargs...), U.components))
end

# Exact physical inner products retain the same Chebyshev mass matrix as CPU.
# Only the reduced scalar returns to the host; allocating diagnostics are
# deliberately separate from the allocation-free integration workspaces.
function LinearAlgebra.dot(U::CuSpectral{T}, V::CuSpectral{T}) where {T}
    CF.grid(U)==CF.grid(V) && size(U)==size(V) || throw(DimensionMismatch("fields must match"))
    nx, nz, ny=size(U)
    Nx=CF.grid(U).physicalsize[1]
    moment(n) = iseven(n) ? T(1)/(1-n^2) : zero(T)
    M=CuArray(Complex{T}[(moment(m+n)+moment(abs(m-n)))/2 for m = 0:(ny-1), n = 0:(ny-1)])
    weights=CuArray(
        vec([
            ((iseven(Nx)&&ix==nx)||(iseven(nz)&&iz==nz÷2+1)) ? 0.0 : (ix==1 ? 1.0 : 2.0) for
            ix = 1:nx, iz = 1:nz
        ]),
    )
    work=CF.spectralmatrix(V)*M
    return sum(real.(conj.(CF.spectralmatrix(U)) .* work) .* weights)
end

# Wall power needs only the horizontal mean profiles, not a full-field download.
CF._meanprofile(U::CuSpectral) = Array(view(parent(U), 1, 1, :))
