export ChannelFlow

"""
    ChannelFlow(grid, nu, dt;
                form=ConvectiveForm(), forcing=nothing,
                pressuregradient=nothing, bulkvelocity=nothing,
                fftwflags=FFTW.MEASURE, fftwtimelimit=FFTW.NO_TIMELIMIT)

Assemble a serial channel/Couette DNS with a fixed nominal time step. The
base profile and domain are supplied by `grid`; `nu` is kinematic viscosity.
Own the nonlinear operator and its FFT plans, the CN-RK2 caches and modal
solvers, the optional forcing callback, and the mean-flow constraint.

`state = Flows.couple(U, P)` stores perturbation velocity and stage pressure.
Both are initially zero. Initialise them before integration to satisfy no slip,
continuity and any prescribed bulk velocity. For a rotational nonlinear form,
`P` must include the total kinetic energy per unit mass; it is not generally
zero even for a laminar base state.

Specify either a constant `pressuregradient=(dPdx, dPdz)` or total
`bulkvelocity=(Ubulk, Wbulk)`. Omitting both selects zero pressure gradient.
The `forcing(t, U, F)` callback overwrites all three spectral components of
an additional acceleration, following [`step!`](@ref).

Advance the owned state with `step!(channel, t)`, or construct the Flows
operator and integrate it over a time span:
```julia
channel = ChannelFlow(grid, nu, dt; pressuregradient=(-2nu, 0))
I = Flows.flow(channel)
I(channel.state, (0.0, 1.0))
```
The example pressure gradient sustains `Ub(y)=1-y²`. Other profiles or forcing
require their own balance. Time is passed explicitly and is not stored in
`ChannelFlow`; all copies of the state retain their pressure history.
"""
struct ChannelFlow{Z, NL, S, F, C}
         state::Z
        nlterm::NL
        scheme::S
       forcing::F
    constraint::C

    function ChannelFlow(            grid::Grid,
                                       nu::Real,
                                       dt::Real;
                                     form::NonlinearityForm=ConvectiveForm(),
                                  forcing=nothing,
                         pressuregradient::Union{Nothing, NTuple{2, Real}}=nothing,
                             bulkvelocity::Union{Nothing, NTuple{2, Real}}=nothing,
                                fftwflags::Integer=FFTW.MEASURE,
                            fftwtimelimit::Real=FFTW.NO_TIMELIMIT)
        isnothing(pressuregradient) || isnothing(bulkvelocity) ||
            throw(ArgumentError("specify either pressuregradient or bulkvelocity"))
        constraint = isnothing(bulkvelocity) ?
            (; pressuregradient=isnothing(pressuregradient) ? (0.0, 0.0) : Float64.(pressuregradient)) :
            (; bulkvelocity=Float64.(bulkvelocity))
        all(isfinite, first(values(constraint))) ||
            throw(ArgumentError("mean-flow constraint values must be finite"))

        P = SpectralField(zeros(ComplexF64, spectralsize(grid, NotPadded())), grid)
        U = VectorField(P)
        for field in U.components
            fill!(parent(field), 0)
        end
        scheme = CNRK2(U, nu, dt)
        nlterm = NonLinearTerm(PhysicalField(grid), P;
                              form=form, fftwflags=fftwflags, fftwtimelimit=fftwtimelimit)
        state = Flows.couple(U, P)
        return new{typeof(state), typeof(nlterm), typeof(scheme), typeof(forcing), typeof(constraint)}(
            state, nlterm, scheme, forcing, constraint)
    end
end

"""
    step!(channel::ChannelFlow, t; stagecache=nothing)

Advance `channel.state` by the configured time step. Return
`(t + dt, (dPdx, dPdz))`; the pressure derivatives come from the final stage.
Optionally record the three coupled velocity/pressure states in a Flows cache.
"""
function step!(   channel::ChannelFlow,
                        t::Real;
               stagecache::Union{Nothing, Flows.AbstractStageCache{3}}=nothing)
    return step!(channel.scheme, channel.nlterm, channel.state[1], channel.state[2], t;
                 forcing=channel.forcing, stagecache=stagecache, channel.constraint...)
end

"""
    Flows.flow(channel::ChannelFlow)

Construct a forward Flows operator using this channel's three-stage CN-RK2
method and nominal fixed time step. Integrate `channel.state` or an independent
`copy(channel.state)` on the same grid. Flows monitors, trajectory storage and
three-stage caches use the coupled velocity/pressure state.

Flows may shorten the last step to reach the requested endpoint. That step
uses a temporary CN-RK2 cache with its actual duration; the configured cache
and nominal step remain available for subsequent calls.
"""
Flows.flow(channel::ChannelFlow) =
    Flows.flow(channel, channel.scheme, Flows.TimeStepConstant(channel.scheme.dt))

"""Bridge Flows' forward stepping interface to the pressure-coupled DNS stages."""
function Flows.step!(scheme::CNRK2{S, F},
                        sys::Flows.System{1, D, CF, Nothing},
                          t::Real,
                         dt::Real,
                      state::Flows.Coupled{2, Tuple{VectorField{F}, F}},
                      cache::Union{Nothing, Flows.AbstractStageCache{3}}) where {S, F, D, CF<:ChannelFlow}
    channel = sys.g
    g = scheme.solvers[1].grid
    grid(state[1][1]) === g && channel.scheme.solvers[1].grid === g ||
        throw(ArgumentError("channel, method and state must share a grid"))

    # Flows.Steps can supply a shorter terminal step. Its temporal shifts
    # require different factors; never use the nominal-step factors for it.
    active = dt == scheme.dt ? scheme : CNRK2(state[1], scheme.nu, dt)
    step!(active, channel.nlterm, state[1], state[2], t;
          forcing=channel.forcing, stagecache=cache, channel.constraint...)
    return state
end
