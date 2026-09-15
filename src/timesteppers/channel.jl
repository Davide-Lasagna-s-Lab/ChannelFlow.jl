"""Advance a channel state with its configured CNRK2 scheme."""
function step!(channel::ChannelFlow,
               state::State{VectorField{F}, F},
               t::Real) where {F<:SpectralField{Float64}}
    return step!(channel.scheme, channel.nlterm, velocity(state), pressure(state), t;
                 forcing=channel.forcing, channel.constraint...)
end

"""Bridge the Flows stepping interface to the pressure-coupled DNS stages."""
function Flows.step!(scheme::CNRK2{S, F, B, C},
                     sys::Flows.System{1, D, CF, Nothing},
                     t::Real,
                     dt::Real,
                     state::State{VectorField{F}, F},
                     ::Nothing) where {S, F, B, C, D, CF<:ChannelFlow}
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
