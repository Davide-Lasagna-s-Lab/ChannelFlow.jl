#//////////////////////////////////////////////////////////////////////////////#
#///             TENSOR FIELD CONSTRUCTION AND COMPONENT ACCESS             ///#
#//////////////////////////////////////////////////////////////////////////////#

"""Three-by-three tensor stored by rows: `G[i,j] = ∂Uᵢ/∂xⱼ` for a gradient."""
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
