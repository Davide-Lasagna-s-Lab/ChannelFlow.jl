export State, velocity, pressure

#//////////////////////////////////////////////////////////////////////////////#
#///                      VELOCITY AND PRESSURE STATE                       ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    State(U, P)

Store spectral perturbation velocity and pressure without copying either.
Access them with `velocity(state)` and `pressure(state)`. The default
rotational form stores modified pressure, including total kinetic energy;
`project!` initially stores an auxiliary projection multiplier instead.
"""
struct State{V, P}
    velocity::V
    pressure::P
end

#//////////////////////////////////////////////////////////////////////////////#
#///                      STATE ACCESS AND ALLOCATION                       ///#
#//////////////////////////////////////////////////////////////////////////////#

"""Return the stored perturbation velocity, without copying."""
velocity(state::State) = state.velocity
"""Return the stored pressure field, without copying."""
pressure(state::State) = state.pressure
Base.size(::State) = (2,)
Base.copy(state::State) = State(copy(state.velocity), copy(state.pressure))
Base.similar(state::State) = State(similar(state.velocity), similar(state.pressure))
