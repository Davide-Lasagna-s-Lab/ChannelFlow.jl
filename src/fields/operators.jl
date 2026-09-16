export ddx1!, ddx2!, ddx3!, curl!

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

#//////////////////////////////////////////////////////////////////////////////#
#///                         SPECTRAL VECTOR CURL                           ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    curl!(OUT::VectorField, U::VectorField)

Overwrite `OUT` with the spectral curl of `U`. The two wall-normal
derivatives use the Chebyshev recurrence. A single subsequent traversal adds
all streamwise and spanwise Fourier contributions, avoiding separate
derivative and sign-change passes for each component.
"""
function curl!(OUT::VectorField{S},
                 U::VectorField{S}) where {S<:SpectralField}
    # curl(U)_1 = dy(U_3) - dz(U_2)
    # curl(U)_3 = dx(U_2) - dy(U_1)
    ddx2!(OUT[1], U[3])
    ddx2!(OUT[3], U[1])

    Ny, Nxh, Nz = size(U[1])
    Lx, _, Lz = domainsize(grid(U[1]))
    α, β = 2π/Lx, 2π/Lz
    @inbounds for iz = 1:Nz, ix = 1:Nxh, iy = 1:Ny
        kx = ix-1
        kz = iz <= (Nz >> 1)+1 ? iz-1 : iz-1-Nz
        OUT[1][iy, ix, iz] -= im*kz*β*U[2][iy, ix, iz]
        OUT[2][iy, ix, iz]  = im*(kz*β*U[1][iy, ix, iz] - kx*α*U[3][iy, ix, iz])
        OUT[3][iy, ix, iz]  = im*kx*α*U[2][iy, ix, iz] - OUT[3][iy, ix, iz]
    end
    return OUT
end

#//////////////////////////////////////////////////////////////////////////////#
#///                       SPECTRAL VELOCITY GRADIENT                       ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    grad!(GRAD::GradientField, U::VectorField)

Compute the spectral velocity gradient of `U` and store it in `GRAD`.

The entries follow `GRAD[i,j] = ∂Uᵢ/∂xⱼ`, where `x₁ = x` is streamwise,
`x₂ = y` is wall-normal and `x₃ = z` is spanwise. The `x` and `z`
derivatives are computed in Fourier space and the `y` derivative by a
Chebyshev coefficient recurrence. The function returns `GRAD`.
"""
function grad!(GRAD::GradientField{S},
                  U::VectorField{S}) where {S<:SpectralField}
    for i = 1:3
        ddx1!(GRAD[i, 1], U[i])
        ddx2!(GRAD[i, 2], U[i])
        ddx3!(GRAD[i, 3], U[i])
    end
    return GRAD
end

#//////////////////////////////////////////////////////////////////////////////#
#///               PHYSICAL TENSOR PRODUCTS AND CONTRACTIONS                ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    outer!(out::GradientField, a::VectorField, b::VectorField)

Compute the pointwise outer product of the physical vector fields `a` and `b`
and store it in `out`, so that `out[i,j] = a[i] * b[j]`. All fields must have
the same physical field type. The function returns `out`.
"""
function outer!(out::GradientField{F},
                  a::VectorField{F},
                  b::VectorField{F}) where {F<:PhysicalField}
    @inbounds for i = 1:3, j = 1:3
        out[i, j] .= a[i].*b[j]
    end
    return out
end

"""
    dot!(out::VectorField, u::VectorField, grad::GradientField)

Contract the physical vector field `u` with the second index of `grad` and
store the result in `out`:

    out[i] = sum(u[j] * grad[i,j] for j = 1:3)

For `grad[i,j] = ∂Uᵢ/∂xⱼ`, this evaluates `(u ⋅ ∇)U`. The operation is
pointwise and returns `out`.
"""
function dot!(out::VectorField{F},
                u::VectorField{F},
             grad::GradientField{F}) where {F<:PhysicalField}
    @inbounds for i = 1:3
        out[i] .= u[1].*grad[i, 1] .+ u[2].*grad[i, 2] .+ u[3].*grad[i, 3]
    end
    return out
end

#//////////////////////////////////////////////////////////////////////////////#
#///                 SPECTRAL VECTOR AND TENSOR DIVERGENCE                  ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    div!(OUT::S, U::VectorField{S}) where {S<:SpectralField}

Compute the spectral divergence of `U` and store it in `OUT`:

    ∂U₁/∂x₁ + ∂U₂/∂x₂ + ∂U₃/∂x₃

The second component and coordinate are wall-normal. The other two derivatives
are evaluated in Fourier space. The function returns `OUT`.
"""
function div!(OUT::S,
                U::VectorField{S}) where {S<:SpectralField}
    ddx1!(OUT, U[1])
    ddx2!(OUT, U[2], true)
    ddx3!(OUT, U[3], true)
    return OUT
end

"""
    div!(OUT::VectorField{S}, GRAD::GradientField{S}) where {S<:SpectralField}

Compute the divergence of each row of the spectral tensor `GRAD` and store the
three results in `OUT`. In components,

    OUT[i] = sum(∂GRAD[i,j]/∂xⱼ for j = 1:3)

The function returns `OUT`.
"""
function div!(OUT::VectorField{S},
             GRAD::GradientField{S}) where {S<:SpectralField}
    @inbounds for i = 1:3
        div!(OUT[i], GRAD[i])
    end
    return OUT
end
