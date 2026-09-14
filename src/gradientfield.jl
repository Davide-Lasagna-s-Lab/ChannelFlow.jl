struct GradientField{F<:AbstractField, V<:VectorField{F}}
    components::NTuple{3, V}
end

GradientField(u::F) where {F<:AbstractField} =
    Gradient((VectorField(u), VectorField(u), VectorField(u)))


"""
    Base.getindex(GRAD::GradientField, i::Int)

Return the `i` row of the velocity gradient tensor `GRAD`.
"""
Base.getindex(GRAD::GradientField, i::Int) = GRAD.components[i]

"""
    Base.getindex(GRAD::GradientField, i::Int, j::Int)

Return the (`i`, `j`) entry of the velocity gradient tensor `GRAD`.
"""
Base.getindex(GRAD::GradientField, i::Int, col::Int) = GRAD.components[i][col]


"""
    grad!(GRAD::GradientField{F}, U::VectorField{F})
        where {T, SIZE, F<:SpectralField{T, SIZE}}

Compute the velocity gradient tensor `GRAD` of the velocity field `U`. Note that
to calculate the wall normal derivatives, the slabs are transposed along the (x1, x3) 
directions and then eventually transposed back to the (x1, x2) directions.
"""
function grad!(GRAD::GradientField{F},
                  U::VectorField{F}) where {T, SIZE, F<:SpectralField{T, SIZE}}

    # launch blocking reordering of first component
    reorder!(U[1])
    reorder!(GRAD[2, 1])

    # then immediately launch nonblocking reordering of the second
    r1 = init_reorder!(U[2])
    r2 = init_reorder!(GRAD[2, 2])

    # compute derivative of first component
    ddx3!(GRAD[2, 1], U[1])

    # wait for second component
    waitall!(r1, r2)

    # upon arrival launch nonblocking reordering of the second component
    r1 = init_reorder!(U[3])
    r2 = init_reorder!(GRAD[2, 3])

    # compute derivative of first component
    ddx3!(GRAD[2, 2], U[2])

    # wait for last component to arrive
    waitall!(r1, r2)

    # then execute
    ddx3!(GRAD[2, 2], U[2])

    # reorder slabs along wall parallel directions
    r1 = init_reorder!(U[1])
    r2 = init_reorder!(U[2])
    r3 = init_reorder!(U[3])
    r4 = init_reorder!(GRAD[2, 1])
    r5 = init_reorder!(GRAD[2, 2])
    r6 = init_reorder!(GRAD[2, 3])

    # then compute derivative along wall parallel directions
    for i = 1:3
        ddx1!(GRAD[1, i], U[i])
        ddx2!(GRAD[3, i], U[i])
    end

    # wait for completion before returning
    waitall!(r1, r2, r3, r4, r5, r6)

    return GRAD
end


"""
    outer!(out::GradientField, a::VectorField, b::VectorField)

Store in `out` the outer product of two vectors `a` and `b`.
"""
function outer!(out::GradientField{F},
                  a::VectorField{F},
                  b::VectorField{F}) where {T, SIZE, F<:PhysicalField{T, SIZE}}
    @inbounds for i = 1:3, j=1:3
        out[i][j] .= a[i].*b[j]
    end
    return out
end

function dot!(out::VectorField{F},
                u::VectorField{F},
             grad::GradientField{F}) where {T, SIZE, F<:PhysicalField{T, SIZE}}
    @inbounds for i = 1:3
        out[i] .= u[1].*grad[i][1] .+ u[2].*grad[i][2] .+ u[3].*grad[i][3]
    end
end

"""
    div!(OUT::F, U::VectorField{F}) where {T, SIZE, F<:SpectralField{T, SIZE}}

Compute the divergence of the vector field `U` and store the result in `OUT`. This 
function operates in the spectral domain, hence the output contains the Fourier 
transformed divergence field along the first two spatial directions. Note that to 
calculate derivatives along the wall normal direction the data is transposed.
"""
function div!(OUT::F,
                U::VectorField{F}) where {T, SIZE, F<:SpectralField{T, SIZE}}

    # launch blocking reordering of first component
    reorder!(U[3]); reorder!(OUT)

    # compute derivative of first component
    ddx3!(OUT, U[3])

    # reorder slabs along wall parallel directions
    reorder!(U[3]); reorder!(OUT)

    # add the other two derivatives
    ddx1!(OUT, U[1], true)
    ddx2!(OUT, U[2], true)

    return OUT
end

# other option with no temporary
function div!(OUT::F,
                U::VectorField{F}) where {T, SIZE, F<:SpectralField{T, SIZE}}

    # launch blocking reordering of first component
    plan(TMP0, U[3])

    # compute derivative of first component
    ddx3!(TMP1, TMP0)

    # reorder slabs along wall parallel directions
    reorder!(TMP1, OUT)

    # add the other two derivatives
    ddx1!(OUT, U[1], true)
    ddx2!(OUT, U[2], true)

    return OUT
end

"""
    div!(OUT::VectorField{F}, GRAD::GradientField{F}) where {T, SIZE, F<:SpectralField{T, SIZE}}

Compute the divergence of every row of the gradient tensor `GRAD` and store the 
result in the components of `OUT`. Note that to calculate derivatives along the 
wall normal direction the data is transposed.
"""
function div!(OUT::VectorField{F},
             GRAD::GradientField{F}) where {T, SIZE, F<:SpectralField{T, SIZE}}
    @inbounds for i = 1:3
        div!(OUT[i], GRAD[i])
    end
end
