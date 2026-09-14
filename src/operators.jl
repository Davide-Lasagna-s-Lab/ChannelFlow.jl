import LinearAlgebra

export ddx1!, ddx2!, ddx3!

"""
    ddx1!(OUT::S, U::S, add::Bool=false) where {S<:SpectralField}

Compute the derivative of the scalar field `U` along the first spatial
direction. The result is added in the output argument `OUT` if `add`
is true, or just overwritten in the output argument otherwise.
"""
function ddx1!(OUT::S, U::S, add::Bool=false) where {S<:SpectralField}
    α = 2π/domainsize(grid(OUT), 1)
    if add
        @loop_jk_i size(OUT) OUT[_i, _j, _k] += im * j * α * U[_i, _j, _k]
    else
        @loop_jk_i size(OUT) OUT[_i, _j, _k]  = im * j * α * U[_i, _j, _k]
    end
    return OUT
end

"""
    ddx2!(OUT::S, U::S, add::Bool=false) where {S<:SpectralField}

Compute the derivative of the scalar field `U` along the second spatial
direction, i.e. the wall-normal direction. The result is added to `OUT` when
`add=true` and overwrites it otherwise.
"""
function ddx2!(OUT::S, U::S, add::Bool=false) where {S<:SpectralField}
    LinearAlgebra.mul!(parent(OUT), grid(OUT).D1, parent(U), Val(1), Val(add))
    return OUT
end

"""
    ddx3!(OUT::S, U::S, add::Bool=false) where {S<:SpectralField}

Compute the derivative of the scalar field `U` along the third spatial
direction, i.e. the spanwise direction. The result is added to `OUT` when
`add=true` and overwrites it otherwise.
"""
function ddx3!(OUT::S, U::S, add::Bool=false) where {S<:SpectralField}
    β = 2π/domainsize(grid(OUT), 2)
    if add
        @loop_jk_i size(U) OUT[_i, _j, _k] += im * k * β * U[_i, _j, _k]
    else
        @loop_jk_i size(U) OUT[_i, _j, _k]  = im * k * β * U[_i, _j, _k]
    end
    return OUT
end

# function invlaplacian!(OUT::FTField{T, SIZE}, U::FTField{T, SIZE}) where {T, SIZE}
#     @loop_jk n m OUT[_k, _j] = - U[_k, _j] / (j^2 + k^2)
#     @inbounds OUT[WaveNumber(0, 0)] = 0
#     return OUT
# end

# function invlaplacian!(OUT::FTField{T, SIZE}, U::FTField{T, SIZE}, c::Real) where {T, SIZE}
#     @loop_jk n m OUT[_k, _j] = U[_k, _j] / (1 + c * (j^2 + k^2))
#     @inbounds OUT[WaveNumber(0, 0)] = 0
#     return OUT
# end


function laplacian!(OUT::S, U::S) where {S<:SpectralField}
    α² = (2π/domainsize(grid(OUT), 1))^2
    β² = (2π/domainsize(grid(OUT), 2))^2

    @loop_jk_i size(U) OUT[_i, _j, _k] =
        -(α²*j^2 + β²*k^2) * U[_i, _j, _k]
    LinearAlgebra.mul!(parent(OUT), grid(OUT).D2, parent(U), Val(1), Val(true))

    return OUT
end
