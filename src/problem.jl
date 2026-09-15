export ChannelFlowProblem

#//////////////////////////////////////////////////////////////////////////////#
#///                 CHANNEL CONFIGURATION AND CONSTRUCTION                 ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    ChannelFlowProblem(grid, profile, nu, dt;
                form=ConvectiveForm(), forcing=nothing,
                pressuregradient=nothing, bulkvelocity=nothing,
                fftwflags=FFTW.MEASURE, fftwtimelimit=FFTW.NO_TIMELIMIT)

Assemble a serial channel/Couette DNS with a fixed nominal time step. The
domain is supplied by `grid`; `profile(y)` is the stationary streamwise base
velocity and `nu` is kinematic viscosity. The profile is converted to
Chebyshev coefficients and stored in `ChannelFlowProblem`.
Own the nonlinear operator and its FFT plans, the CNRK2 caches and modal
solvers, the optional forcing callback, and the mean-flow constraint.

`zero_state(channel.grid)` creates a coupled perturbation velocity and stage
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

Create a state and integrate it with the Flows operator:
```julia
channel = ChannelFlowProblem(grid, y -> 1-y^2, nu, dt;
                      pressuregradient=(-2nu, 0))
state = zero_state(channel.grid)
I = Flows.flow(channel)
I(state, (0.0, 1.0))
```
The example pressure gradient sustains `Ub(y)=1-y²`. Other profiles or forcing
require their own balance. Time is passed explicitly and is not stored in
`ChannelFlowProblem`; each state copy retains its own pressure history.
"""
struct ChannelFlowProblem{G, B, NL, S, F, C}
          grid::G
      baseflow::B
        nlterm::NL
        scheme::S
       forcing::F
    constraint::C

    function ChannelFlowProblem(            grid::Grid,
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
        # `zero_state`, so one channel can evolve several states.
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

#//////////////////////////////////////////////////////////////////////////////#
#///                      FLOWS INTEGRATION INTERFACE                       ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    Flows.flow(channel::ChannelFlowProblem)

Construct a forward Flows operator using this channel's three-stage CNRK2
method and nominal fixed time step. Pass any coupled state created by
`zero_state(channel.grid)` (or an independent copy) to the returned operator.
Flows monitors and trajectory storage use the coupled velocity/pressure state.

Flows may shorten the last step to reach the requested endpoint. That step
uses a temporary CNRK2 cache with its actual duration; the configured cache
and nominal step remain available for subsequent calls.
"""
Flows.flow(channel::ChannelFlowProblem) =
    Flows.flow(channel, channel.scheme, Flows.TimeStepConstant(channel.scheme.dt))
