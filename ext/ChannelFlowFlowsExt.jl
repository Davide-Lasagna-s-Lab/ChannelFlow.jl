module ChannelFlowFlowsExt

using ChannelFlow
import Flows

# Keep Flows' method hierarchy outside the standalone DNS time-stepper.
struct FlowMethod{S} <: Flows.AbstractMethod{State, Flows.NormalMode, 3}
    scheme::S
end

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
    Flows.flow(channel, FlowMethod(channel.scheme), Flows.TimeStepConstant(channel.scheme.dt))

#//////////////////////////////////////////////////////////////////////////////#
#///                      FLOWS TIME-STEPPING ADAPTER                       ///#
#//////////////////////////////////////////////////////////////////////////////#

"""Bridge the Flows stepping interface to the pressure-coupled DNS stages."""
function Flows.step!(method::FlowMethod,
                        sys::Flows.System{1, D, C, Nothing},
                          t::Real,
                         dt::Real,
                      state::State,
                           ::Nothing) where {D, C<:ChannelFlowProblem}
    channel = sys.g
    scheme = method.scheme

    # a shortened terminal step needs factors built for its actual duration.
    active = dt == scheme.dt ? scheme : ChannelFlow._same_storage(CNRK2(channel.grid, channel.baseflow, scheme.nu, dt), velocity(state)[1])

    # delegate the stepping to the CNRK2 step! method
    step!(active, channel.nlterm, velocity(state), stagepressure(state), t;
          forcing=channel.forcing, channel.constraint...)
    return state
end

end
