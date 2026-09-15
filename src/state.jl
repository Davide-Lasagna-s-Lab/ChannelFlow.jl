export State, velocity, pressure

"""Coupled DNS state containing perturbation velocity and pressure."""
struct State{V, P}
    velocity::V
    pressure::P
end

velocity(state::State) = state.velocity
pressure(state::State) = state.pressure
Base.size(::State) = (2,)
Base.copy(state::State) = State(copy(state.velocity), copy(state.pressure))
Base.similar(state::State) = State(similar(state.velocity), similar(state.pressure))
