import LinearAlgebra

export ddx1!, ddx2!, ddx3!

"""
    ddx1!(OUT::S, U::S, add::Bool=false) where {S<:SpectralField}

Compute the derivative of the scalar field `U` along the first spatial
direction. The result is added in the output argument `OUT` if `add`
is true, or just overwritten in the output argument otherwise.
"""
function ddx1!(OUT::S, U::S, add::Bool=false) where {S<:SpectralField}
    α = 2π/domainsize(OUT, 1)
    if add
        @loop_jk_i size(OUT) OUT[_j, _k, _i] += im * j * α * U[_j, _k, _i]
    else
        @loop_jk_i size(OUT) OUT[_j, _k, _i]  = im * j * α * U[_j, _k, _i]
    end
    return OUT
end

"""
    ddx2!(OUT::S, U::S, add::Bool=false) where {S<:SpectralField}

Compute the derivative of the scalar field `U` along the second spatial
direction. The result is added to the output argument `OUT` if `add`
is true, or just overwritten in the output argument otherwise.
"""
function ddx2!(OUT::S, U::S, add::Bool=false) where {S<:SpectralField}
    β = 2π/domainsize(OUT, 2)
    if add
        @loop_jk_i size(U) OUT[_j, _k, _i] += im * k * β * U[_j, _k, _i]
    else
        @loop_jk_i size(U) OUT[_j, _k, _i]  = im * k * β * U[_j, _k, _i]
    end
    return OUT
end

"""
    ddx3!(OUT::S, U::S, add::Bool=false) where {S<:SpectralField}

Compute the derivative of the scalar field `U` along the third spatial
direction, i.e. the wall normal direction. The result is written
in the output argument `OUT`. The type parameter `SIZE` is assumed to
describe the global size of the physical grid.
"""
function ddx3!(OUT::S,
                 U::S,
            tcache::TransposeCache) where {T, A, S<:SpectralField{T, A}}
    if A <: DecomposedArray
        # transpose
        transpose!(parent(tcache.tmp[1]), tcache.plan, parent(U))

        # differentiate along direction 3
        mul!(parent(tcache.tmp[2]),
             diffmat(grid(OUT), 1),
             parent(tcache.tmp[1]), Val(3))

        # then transpose back to OUT
        transpose!(parent(OUT), tcache.plan, tcache.TMP_OUT)
    else
        mul!(parent(OUT), diffmat(grid(OUT), 1), parent(U), Val(3))
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


function laplacian!(OUT::SpectralField{T},
                      U::SpectralField{T}, 
                  cache::TransposeCache) where {T}

    # TODO: overlap transpose and jk derivatives
    α² = (2π/domainsize(OUT, 2))^2
    β² = (2π/domainsize(OUT, 2))^2

    # check input state
    (slabdim(OUT) == 3) && (slabdim(U) == 3) ||
        throw(ArgumentError("invalid slab state"))

    # differentiate along wall parallel directions first
    @loop_jk_i size(U) OUT[_j, _k, _i]  =  - (α²*j^2 + β²*k^2) * U[_j, _k, _i]

    # transpose
    transpose!(cache.TMP_IN, cache.plan, U)

    # differentiate along direction 3
    mul!(cache.TMP_OUT, OUT.grid.D2, cache.TMP_IN, Val(3))

    # then transpose back to OUT
    transpose!(OUT, cache.plan, cache.TMP_OUT)

    return OUT
end