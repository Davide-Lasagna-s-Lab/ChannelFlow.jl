export State, velocity, stagepressure

#//////////////////////////////////////////////////////////////////////////////#
#///                    VELOCITY AND STAGE-PRESSURE STATE                   ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    State(U, Q)

Store spectral perturbation velocity and the CNRK2 stage-pressure field
without copying either. Access them with `velocity(state)` and
`stagepressure(state)`. The name distinguishes this algebraic field from an
independently evolved variable and, for rotational form, from physical
pressure: the stored value is modified pressure including total kinetic
energy. [`pressure`](@ref) reconstructs the appropriate pressure from a
divergence-free, no-slip initial velocity field.
"""
struct State{V, P}
         velocity::V
    stagepressure::P
end

#//////////////////////////////////////////////////////////////////////////////#
#///                      STATE ACCESS AND ALLOCATION                       ///#
#//////////////////////////////////////////////////////////////////////////////#

"""Return the stored perturbation velocity, without copying."""
velocity(state::State) = state.velocity

"""Return the algebraic pressure used by the time stepper, without copying."""
stagepressure(state::State) = state.stagepressure

Base.copy(state::State) = State(copy(state.velocity), copy(state.stagepressure))
Base.similar(state::State) = State(similar(state.velocity), similar(state.stagepressure))
