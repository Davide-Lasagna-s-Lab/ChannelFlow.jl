export ChannelFlow, initial_state, baseflow

"""
    ChannelFlow(grid, profile, nu, dt;
                form=ConvectiveForm(), forcing=nothing,
                pressuregradient=nothing, bulkvelocity=nothing,
                fftwflags=FFTW.MEASURE, fftwtimelimit=FFTW.NO_TIMELIMIT)

Assemble a serial channel/Couette DNS with a fixed nominal time step. The
domain is supplied by `grid`; `profile(y)` is the stationary streamwise base
velocity and `nu` is kinematic viscosity. The profile is converted to
Chebyshev coefficients and stored in `ChannelFlow`.
Own the nonlinear operator and its FFT plans, the CNRK2 caches and modal
solvers, the optional forcing callback, and the mean-flow constraint.

`initial_state(channel)` creates a coupled perturbation velocity and stage
pressure. Initialise this state before integration to satisfy no slip,
continuity and any prescribed bulk velocity. For a rotational nonlinear form,
`P` must include the total kinetic energy per unit mass; it is not generally
zero even for a laminar base state.

The CNRK2 momentum source contains `-grad(P)` from the preceding stage.
Pressure is therefore retained with velocity between stages and time steps;
copies and saved states need both fields to resume the same discrete evolution.
It is still determined by the incompressibility solve, not advanced by an
independent pressure evolution equation.

Specify either a constant `pressuregradient=(dPdx, dPdz)` or total
`bulkvelocity=(Ubulk, Wbulk)`. Omitting both selects zero pressure gradient.
The `forcing(t, U, F)` callback overwrites all three spectral components of
an additional acceleration, following [`step!`](@ref).

Create a state and integrate it directly, or construct the Flows operator:
```julia
channel = ChannelFlow(grid, y -> 1-y^2, nu, dt;
                      pressuregradient=(-2nu, 0))
state = initial_state(channel)
I = Flows.flow(channel)
I(state, (0.0, 1.0))
```
The example pressure gradient sustains `Ub(y)=1-y²`. Other profiles or forcing
require their own balance. Time is passed explicitly and is not stored in
`ChannelFlow`; each state copy retains its own pressure history.
"""
struct ChannelFlow{G, B, NL, S, F, C}
          grid::G
      baseflow::B
        nlterm::NL
        scheme::S
       forcing::F
    constraint::C

    function ChannelFlow(            grid::Grid,
                                    profile::Function,
                                       nu::Real,
                                       dt::Real;
                                     form::NonlinearityForm=ConvectiveForm(),
                                  forcing=nothing,
                         pressuregradient::Union{Nothing, NTuple{2, Real}}=nothing,
                             bulkvelocity::Union{Nothing, NTuple{2, Real}}=nothing,
                                fftwflags::Integer=FFTW.MEASURE,
                            fftwtimelimit::Real=FFTW.NO_TIMELIMIT)
        # Store only the selected mean-flow constraint. Its name is reused
        # directly as a keyword by the time-stepper and the mean-mode solve.
        isnothing(pressuregradient) || isnothing(bulkvelocity) ||
            throw(ArgumentError("specify either pressuregradient or bulkvelocity"))
        constraint = isnothing(bulkvelocity) ?
            (; pressuregradient=isnothing(pressuregradient) ? (0.0, 0.0) : Float64.(pressuregradient)) :
            (; bulkvelocity=Float64.(bulkvelocity))
        all(isfinite, first(values(constraint))) ||
            throw(ArgumentError("mean-flow constraint values must be finite"))

        # Build scalar prototypes only to initialise the nonlinear operator and
        # its FFT plans. The resulting state is created separately by
        # `initial_state`, so one channel can evolve several states.
        baseflow = chebyshev_coefficients(grid, profile)
        P = SpectralField(zeros(ComplexF64, spectralsize(grid, NotPadded())), grid)

        # Precompute the three stage-specific Stokes factorisations, wall
        # influence responses, RK workspaces and the base-flow viscous term.
        scheme = CNRK2(grid, baseflow, nu, dt)

        # The scalar prototypes supply types and grid information. The
        # nonlinear operator allocates form-specific caches and padded FFT plans.
        nlterm = NonLinearTerm(PhysicalField(grid), P, baseflow;
                              form=form, fftwflags=fftwflags, fftwtimelimit=fftwtimelimit)

        return new{typeof(grid), typeof(baseflow), typeof(nlterm), typeof(scheme), typeof(forcing), typeof(constraint)}(
            grid, baseflow, nlterm, scheme, forcing, constraint)
    end
end

"""Return the Chebyshev coefficients of the stationary profile owned by `channel`."""
baseflow(channel::ChannelFlow) = channel.baseflow

"""
    initial_state(channel::ChannelFlow)

Allocate a zero coupled state `(U, P)` for `channel`. `U` is the perturbation
velocity and `P` is the stage pressure required by CNRK2. The caller may then
populate and project `U`, and initialise `P`, before integration.
"""
function initial_state(channel::ChannelFlow)
    P = SpectralField(zeros(ComplexF64, spectralsize(channel.grid, NotPadded())), channel.grid)
    return Flows.couple(VectorField(P), P)
end

"""
    step!(channel::ChannelFlow, state, t)

Advance the supplied coupled state by the configured time step. Return
`(t + dt, (dPdx, dPdz))`; the pressure derivatives come from the final stage.
The state is modified in place and is owned by the caller.
"""
function step!(channel::ChannelFlow,
               state::Flows.Coupled{2, Tuple{VectorField{F}, F}},
               t::Real) where {F<:SpectralField{Float64}}
    return step!(channel.scheme, channel.nlterm, state[1], state[2], t;
                 forcing=channel.forcing, channel.constraint...)
end

"""
    Flows.flow(channel::ChannelFlow)

Construct a forward Flows operator using this channel's three-stage CNRK2
method and nominal fixed time step. Pass any coupled state created by
`initial_state(channel)` (or an independent copy) to the returned operator.
Flows monitors and trajectory storage use the coupled velocity/pressure state.

Flows may shorten the last step to reach the requested endpoint. That step
uses a temporary CNRK2 cache with its actual duration; the configured cache
and nominal step remain available for subsequent calls.
"""
Flows.flow(channel::ChannelFlow) =
    Flows.flow(channel, channel.scheme, Flows.TimeStepConstant(channel.scheme.dt))

"""Bridge Flows' forward stepping interface to the pressure-coupled DNS stages."""
function Flows.step!(scheme::CNRK2{S, F, B, C},
                        sys::Flows.System{1, D, CF, Nothing},
                          t::Real,
                         dt::Real,
                      state::Flows.Coupled{2, Tuple{VectorField{F}, F}},
                           ::Nothing) where {S, F, B, C, D, CF<:ChannelFlow}
    channel = sys.g
    g = scheme.solvers[1].grid
    grid(state[1][1]) === g && channel.scheme.solvers[1].grid === g ||
        throw(ArgumentError("channel, method and state must share a grid"))

    # Flows.Steps can supply a shorter terminal step. Its temporal shifts
    # require different factors; never use the nominal-step factors for it.
    active = dt == scheme.dt ? scheme : CNRK2(channel.grid, scheme.baseflow, scheme.nu, dt)
    step!(active, channel.nlterm, state[1], state[2], t;
          forcing=channel.forcing, channel.constraint...)
    return state
end
