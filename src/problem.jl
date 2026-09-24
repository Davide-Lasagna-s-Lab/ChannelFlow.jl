export ChannelFlowProblem, CouetteFlow, PoiseuilleFlow

#//////////////////////////////////////////////////////////////////////////////#
#///                 CHANNEL CONFIGURATION AND CONSTRUCTION                 ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    ChannelFlowProblem(grid, profile, nu, dt;
                form=RotatingForm(), forcing=NoForcing(),
                pressuregradient=nothing, bulkvelocity=nothing,
                fftwflags=FFTW.MEASURE, fftwtimelimit=FFTW.NO_TIMELIMIT)

Assemble a Poiseuille/Couette DNS with a fixed nominal time step. The
domain is supplied by `grid`; `profile(y)` is the stationary streamwise base
velocity and `nu` is kinematic viscosity. The profile is converted to
values at the wall-normal collocation points and stored in
`ChannelFlowProblem`; spectral coefficients are cached inside its operators.
Own the nonlinear operator and its FFT plans, the CNRK2 caches and modal
solvers, the forcing callback, and the mean-flow constraint.

`zero_state(channel.grid)` creates a perturbation velocity and its algebraic
stage pressure. Initialise this state before integration to satisfy no slip,
continuity and any prescribed bulk velocity. The default rotational form uses
modified pressure, so `P` must include the total kinetic energy per unit mass;
it is not generally zero even for a laminar base state.

The CNRK2 momentum source contains `-grad(P)` from the preceding stage.
Pressure is therefore retained with velocity between stages and time steps;
copies and saved states need both fields to resume the same discrete evolution.
It is still determined by the incompressibility solve, not advanced by an
independent pressure evolution equation.

Specify either a constant `pressuregradient=(dPdx, dPdz)` or total
`bulkvelocity=(Ubulk, Wbulk)`. Omitting both selects zero pressure gradient.
The `forcing(t, U, F)` callback adds its spectral acceleration to `F`,
preserving its existing contents and leaving `U` unchanged; see [`step!`](@ref).

CPU Fourier and Chebyshev transforms use FFTW. Planning options are
controlled by `fftwflags` and `fftwtimelimit`. GPU transforms use cuFFT.

Create a state and integrate it with the optional Flows operator:
```julia
channel = ChannelFlowProblem(grid, y -> 1-y^2, nu, dt;
                      pressuregradient=(-2nu, 0))
state = zero_state(channel.grid)
# Modified pressure of the laminar profile: |Ub|²/2 (up to a constant).
stagepressure(state)[1, 1, :] .= parent(chebcoeffs(
    (1 .- channel.grid.y.^2).^2 ./ 2))
import Flows
I = Flows.flow(channel)
I(state, (0.0, 1.0))
```
The example pressure gradient sustains `Ub(y)=1-y²`. Other profiles or forcing
require their own balance. Time is passed explicitly and is not stored in
`ChannelFlowProblem`; each state copy retains its own stage-pressure history.
"""
struct ChannelFlowProblem{G, B, NL, S, F, C}
          grid::G
      baseflow::B
        nlterm::NL
        scheme::S
       forcing::F
    constraint::C

    ChannelFlowProblem(grid, baseflow, nonlinear, scheme, forcing, constraint) =
        new{typeof(grid),typeof(baseflow),typeof(nonlinear),typeof(scheme),typeof(forcing),typeof(constraint)}(
            grid,baseflow,nonlinear,scheme,forcing,constraint)

    function ChannelFlowProblem(            grid::Grid,
                                         profile::Function,
                                              nu::Real,
                                              dt::Real;
                                            form::NonlinearityForm=RotatingForm(),
                                         forcing=NoForcing(),
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
        baseflow = Float64.(profile.(grid.y))
        P = SpectralField(grid)

        # Precompute the three stage-specific Stokes factorisations, wall
        # influence responses, RK workspaces and the base-flow viscous term.
        scheme = CNRK2(grid, baseflow, nu, dt)

        # The scalar prototypes supply types and grid information. The
        # nonlinear operator allocates form-specific caches and padded FFT plans.
        nlterm = NonLinearTerm(PhysicalField(grid), P, scheme.baseflow;
                              form=form,
                              fftwflags=fftwflags, fftwtimelimit=fftwtimelimit)

        return new{typeof(grid), typeof(baseflow), typeof(nlterm), typeof(scheme), typeof(forcing), typeof(constraint)}(
            grid, baseflow, nlterm, scheme, forcing, constraint)
    end
end

#//////////////////////////////////////////////////////////////////////////////#
#///                       STANDARD LAMINAR PROFILES                        ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    CouetteFlow(grid, nu, dt; kwargs...)

Construct a [`ChannelFlowProblem`](@ref) with base velocity `Ub(y)=y`,
corresponding to walls moving at velocities ±1. The default pressure gradient
is zero. Forward all keyword arguments to `ChannelFlowProblem`, including
`bulkvelocity`, `forcing`, `form` and FFT planning options.
"""
CouetteFlow(grid::Grid, nu::Real, dt::Real; kwargs...) =
    ChannelFlowProblem(grid, identity, nu, dt; kwargs...)

"""
    PoiseuilleFlow(grid, nu, dt; bulkvelocity=nothing, pressuregradient=..., kwargs...)

Construct a [`ChannelFlowProblem`](@ref) with base velocity `Ub(y)=1-y^2`,
unit centreline velocity and stationary walls. By default, the pressure
gradient `(-2nu, 0)` sustains this profile.

Supplying `bulkvelocity` selects constant total bulk velocity instead; for
this laminar profile its value is `(2/3, 0)`. An explicit `pressuregradient`
overrides the default. Specifying both constraints is rejected by
`ChannelFlowProblem`. Forward all remaining keywords unchanged.
"""
function PoiseuilleFlow(grid::Grid, nu::Real, dt::Real;
                   bulkvelocity=nothing,
                   pressuregradient=isnothing(bulkvelocity) ? (-2nu, 0) : nothing,
                   kwargs...)
    return ChannelFlowProblem(grid, y -> 1-y^2, nu, dt;
                              bulkvelocity=bulkvelocity,
                              pressuregradient=pressuregradient, kwargs...)
end
