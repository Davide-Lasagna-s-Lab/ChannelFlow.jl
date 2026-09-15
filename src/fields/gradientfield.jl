struct GradientField{F<:AbstractField, V<:VectorField{F}}
    components::NTuple{3, V}
end

"""
    GradientField(u::AbstractField)

Allocate a three-by-three tensor field from the scalar-field prototype `u`.
Each entry has the same storage, grid and numerical type as `u`, but owns
independent data.
"""
GradientField(u::F) where {F<:AbstractField} =
    GradientField((VectorField(u), VectorField(u), VectorField(u)))


"""
    Base.getindex(GRAD::GradientField, i::Int)

Return row `i` of `GRAD` as a `VectorField`. For a velocity gradient, row `i`
contains the derivatives of velocity component `i`.
"""
Base.getindex(GRAD::GradientField, i::Int) = GRAD.components[i]

"""
    Base.getindex(GRAD::GradientField, i::Int, j::Int)

Return entry `(i, j)` of `GRAD`. The first index selects the velocity component
and the second selects the derivative direction.
"""
Base.getindex(GRAD::GradientField, i::Int, j::Int) = GRAD.components[i][j]


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
