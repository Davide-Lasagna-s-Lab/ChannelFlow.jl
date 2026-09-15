export ddx1!, ddx2!, ddx3!

#//////////////////////////////////////////////////////////////////////////////#
#///                     STREAMWISE FOURIER DERIVATIVE                      ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    ddx1!(OUT::S, U::S, add::Bool=false) where {S<:SpectralField}

Compute the derivative of the scalar field `U` along the first spatial
direction. The result is added in the output argument `OUT` if `add`
is true, or just overwritten in the output argument otherwise.
"""
function ddx1!(OUT::S, U::S, add::Bool=false) where {S<:SpectralField}
    Lx, _, _ = domainsize(grid(OUT))
    α = 2π/Lx
    if add
        @loop_jk_i size(OUT) OUT[_i, _j, _k] += im * j * α * U[_i, _j, _k]
    else
        @loop_jk_i size(OUT) OUT[_i, _j, _k]  = im * j * α * U[_i, _j, _k]
    end
    return OUT
end

#//////////////////////////////////////////////////////////////////////////////#
#///                    WALL-NORMAL CHEBYSHEV DERIVATIVE                    ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    ddx2!(OUT::S, U::S, add::Bool=false) where {S<:SpectralField}

Compute the derivative of the scalar field `U` along the second spatial
direction, i.e. the wall-normal direction. Use the backward Chebyshev
recurrence from Channelflow's `chebyshev.cpp::diff`, scaled to the physical
wall interval. Add the derivative when `add=true`; otherwise overwrite `OUT`.
`OUT` and `U` may be the same field.
"""
function ddx2!(OUT::S, U::S, add::Bool=false) where {S<:SpectralField}
    Ny, Nxh, Nz = size(U)
    y = grid(U).y
    scale = 4/(first(y)-last(y))

    @inbounds for iz = 1:Nz, ix = 1:Nxh
        a_next = d_next = d_next2 = zero(eltype(U))
        for n = Ny-1:-1:0
            # Save a_n before writing OUT, so the recurrence also works in
            # place. d_next2 holds d_{n+2}, independently of the output.
            a = U[n+1, ix, iz]
            d = d_next2 + scale*(n+1)*a_next
            value = n == 0 ? d/2 : d
            add ? (OUT[n+1, ix, iz] += value) : (OUT[n+1, ix, iz] = value)
            a_next = a
            d_next2, d_next = d_next, d
        end
    end
    return OUT
end

#//////////////////////////////////////////////////////////////////////////////#
#///                      SPANWISE FOURIER DERIVATIVE                       ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    ddx3!(OUT::S, U::S, add::Bool=false) where {S<:SpectralField}

Compute the derivative of the scalar field `U` along the third spatial
direction, i.e. the spanwise direction. The result is added to `OUT` when
`add=true` and overwrites it otherwise.
"""
function ddx3!(OUT::S, U::S, add::Bool=false) where {S<:SpectralField}
    _, _, Lz = domainsize(grid(OUT))
    β = 2π/Lz
    if add
        @loop_jk_i size(U) OUT[_i, _j, _k] += im * k * β * U[_i, _j, _k]
    else
        @loop_jk_i size(U) OUT[_i, _j, _k]  = im * k * β * U[_i, _j, _k]
    end
    return OUT
end

#//////////////////////////////////////////////////////////////////////////////#
#///                      FOURIER-CHEBYSHEV LAPLACIAN                       ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    laplacian!(OUT::S, U::S) where {S<:SpectralField}

Overwrite `OUT` with `∂yy U - (α²*kx² + β²*kz²)*U` in coefficient space.
Fuse two Chebyshev derivative recurrences, equivalent to Channelflow's
`diff2`, to avoid a temporary field. `OUT` and `U` may be the same field.
"""
function laplacian!(OUT::S, U::S) where {S<:SpectralField}
    Ny, Nxh, Nz = size(U)
    y = grid(U).y
    scale = 4/(first(y)-last(y))
    Lx, _, Lz = domainsize(grid(OUT))
    α² = (2π/Lx)^2
    β² = (2π/Lz)^2

    @inbounds for iz = 1:Nz, ix = 1:Nxh
        kx = ix-1
        kz = iz <= (Nz >> 1)+1 ? iz-1 : iz-1-Nz
        k² = α²*kx^2 + β²*kz^2
        a_next = d_next = d_next2 = e_next = e_next2 = zero(eltype(U))
        for n = Ny-1:-1:0
            a = U[n+1, ix, iz]
            d = d_next2 + scale*(n+1)*a_next
            e = e_next2 + scale*(n+1)*d_next
            OUT[n+1, ix, iz] = (n == 0 ? e/2 : e) - k²*a
            a_next = a
            d_next2, d_next = d_next, d
            e_next2, e_next = e_next, e
        end
    end

    return OUT
end
