#//////////////////////////////////////////////////////////////////////////////#
#///                      FLOWS INTEGRATION INTERFACE                       ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    Flows.flow(channel::ChannelFlowProblem)

Construct a forward Flows operator using this channel's three-stage CNRK2
method and nominal fixed time step. Pass a `State` created by
`zero_state(channel.grid)` (or an independent copy) to the returned operator.
Flows monitors and trajectory storage use velocity together with its algebraic
stage pressure. Access the latter with `stagepressure(state)`.

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
function Flows.step!(scheme::CNRK2{S, F, P},
                        sys::Flows.System{1, D, CF, Nothing},
                          t::Real,
                         dt::Real,
                      state::State{VectorField{F}, F},
                           ::Nothing) where {S, F, P, D, CF<:ChannelFlowProblem}
    # just an alias
    channel = sys.g

    # a shortened terminal step needs factors built for its actual duration.
    active = dt == scheme.dt ? scheme : _same_storage(CNRK2(channel.grid, channel.baseflow, scheme.nu, dt), velocity(state)[1])

    # delegate the stepping to the CNRK2 step! method
    step!(active, channel.nlterm, velocity(state), stagepressure(state), t;
          forcing=channel.forcing, channel.constraint...)
    return state
end
