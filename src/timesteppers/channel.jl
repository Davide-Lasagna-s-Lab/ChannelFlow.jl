#//////////////////////////////////////////////////////////////////////////////#
#///                      FLOWS INTEGRATION INTERFACE                       ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    Flows.flow(channel::ChannelFlowProblem)

Construct a forward Flows operator using this channel's three-stage CNRK2
method and nominal fixed time step. Pass a `State` created by
`zero_state(channel.grid)` (or an independent copy) to the returned operator.
Flows monitors and trajectory storage use the coupled velocity/pressure state.

Flows may shorten the last step to reach the requested endpoint. That step
uses a temporary CNRK2 cache with its actual duration; the configured cache
and nominal step remain available for subsequent calls.
"""
Flows.flow(channel::ChannelFlowProblem) =
    Flows.flow(channel, channel.scheme, Flows.TimeStepConstant(channel.scheme.dt))

#//////////////////////////////////////////////////////////////////////////////#
#///                      FLOWS TIME-STEPPING ADAPTER                       ///#
#//////////////////////////////////////////////////////////////////////////////#

"""Bridge the Flows stepping interface to the pressure-coupled DNS stages."""
function Flows.step!(scheme::CNRK2{S, F, B, C},
                        sys::Flows.System{1, D, CF, Nothing},
                          t::Real,
                         dt::Real,
                      state::State{VectorField{F}, F},
                     ::Nothing) where {S, F, B, C, D, CF<:ChannelFlowProblem}
    channel = sys.g
    g = scheme.solvers[1].grid
    grid(velocity(state)[1]) === g && channel.scheme.solvers[1].grid === g ||
        throw(ArgumentError("channel, method and state must share a grid"))

    # A shortened terminal step needs factors built for its actual duration.
    active = dt == scheme.dt ? scheme : CNRK2(channel.grid, scheme.baseflow, scheme.nu, dt)
    step!(active, channel.nlterm, velocity(state), pressure(state), t;
          forcing=channel.forcing, channel.constraint...)
    return state
end
