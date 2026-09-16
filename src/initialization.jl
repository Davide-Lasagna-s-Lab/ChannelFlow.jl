export zero_state, random_state, project!, pressure

#//////////////////////////////////////////////////////////////////////////////#
#///                         ZERO STATE ALLOCATION                          ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    zero_state(grid::Grid)

Allocate `State(U, Q)` with zero spectral perturbation velocity and stage
pressure. Access these fields with `velocity(state)` and
`stagepressure(state)`.
A laminar base profile belongs to the problem, so it is not added here.
Zero stage pressure need not be a consistent initial value for time stepping.
"""
zero_state(grid::Grid) = State(VectorField(SpectralField(grid)), SpectralField(grid))

#//////////////////////////////////////////////////////////////////////////////#
#///                     RANDOM VELOCITY INITIALIZATION                     ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    random_state(problem::ChannelFlowProblem, epsilon)

Create three random physical velocity components on the padded grid, transform
and project them onto the divergence-free, no-slip space. Return a `State`
with the physical or modified pressure appropriate to `problem`.
`epsilon` scales Gaussian samples before projection; it does not prescribe
the final RMS or kinetic energy. Use Julia's default RNG; call `Random.seed!`
beforehand for reproducibility.
"""
function random_state(problem::ChannelFlowProblem, epsilon::Real)
    # sanity check
    epsilon >= 0 || throw(ArgumentError("epsilon must be non-negative"))
    
    # random function
    eps_rand(x, y, z) = epsilon * Random.randn()

    # create non-zero-divergence velocity field in physical space
    u = VectorField(PhysicalField(problem.grid, eps_rand, Padded()),
                    PhysicalField(problem.grid, eps_rand, Padded()),
                    PhysicalField(problem.grid, eps_rand, Padded()))
    
    # transform to spectral space and project 
    U = project!(FFT(u), problem)

    # obtain pressure field
    P = pressure(U, problem)

    return State(U, P)
end

#//////////////////////////////////////////////////////////////////////////////#
#///                   DIVERGENCE-FREE NO-SLIP PROJECTION                   ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    project!(U::VectorField, problem::ChannelFlowProblem)

Project the spectral perturbation velocity `U` onto the divergence-free,
no-slip space, overwrite its three components, and return `U`.
The auxiliary multiplier is discarded; use [`pressure`](@ref) afterwards
to reconstruct the pressure from the projected velocity.
Use the existing Fourier--Chebyshev tau and influence-matrix solvers for
the Stokes projection
```text
-ΔUnew + grad(phi) + G = -ΔU,
div(Unew) = 0,  Unew(y=±1) = 0.
```
The unit coefficient and zero temporal shift define an auxiliary stationary
problem, independent of the DNS viscosity and time step. Applying the same
Laplacian to the input avoids introducing a viscous filtering step. In the
continuous formulation this minimises the gradient seminorm of the velocity
correction; it is not an orthogonal L² projection. Here the equations use
the same tau discretisation as [`StokesSolver`](@ref).

The mean-flow constraint and base profile are taken from `problem`. A field
already satisfying the constraints is unchanged up to solve roundoff.
Excluded Nyquist planes are set to zero. The projection acts on the
perturbation field and does not add or remove the base profile.

Require resolved `ComplexF64` fields on one grid, odd `Ny ≥ 3`, independent
component storage and Fourier conjugate symmetry for a real velocity.
Allocate factors and workspaces on each call: intended for initialisation.
"""
function project!(      U::VectorField{F},
                  problem::ChannelFlowProblem) where {F<:SpectralField{Float64}}
    g = grid(U[1])
    g === problem.grid || throw(ArgumentError("velocity and problem must share a grid"))
    for field in U.components
        grid(field) === g || throw(ArgumentError("velocity components must share a grid"))
        size(field) == spectralsize(g, NotPadded()) ||
            throw(DimensionMismatch("velocity must have the resolved spectral size"))
    end

    # First make the supplied perturbation velocity satisfy continuity,
    # no slip and, when requested, the total bulk-flow constraint.
    solver = StokesSolver(g, 1.0, 0.0)
    R, phi = similar(U), similar(U[1])
    # Form the complete source before solve! overwrites any velocity component.
    for i = 1:3
        laplacian!(R[i], U[i])
        R[i] .*= -1
    end
    bulkvelocity = haskey(problem.constraint, :bulkvelocity) ?
                   problem.constraint.bulkvelocity : nothing
    solve!(solver, U, phi, R;
           bulkvelocity=bulkvelocity,
           baseflow=parent(problem.scheme.baseflow))

    return U
end

#//////////////////////////////////////////////////////////////////////////////#
#///                        PRESSURE RECONSTRUCTION                         ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    pressure(U::VectorField, problem::ChannelFlowProblem, t=0)

Reconstruct and return the spectral pressure from a divergence-free, no-slip
perturbation velocity without modifying `U`. The caller must project the
velocity first if needed. Workspaces and solver factors are allocated per call.
`t` is passed to nonlinear and forcing callbacks.

Form the pressure-free acceleration
```text
A = nonlinear(U + Ub) + forcing + nu*Delta(U + Ub).
```
An instantaneous Stokes projection then solves
```text
a + grad(P) = A,  div(a) = 0,  a(y=+-1) = 0.
```
Consequently `P` is the ordinary physical pressure for the convective,
divergence and alternating forms. For the rotational form it is the modified
pressure `P = p + abs(U + Ub)^2/2`, because the kinetic-energy gradient has
already been absorbed into the nonlinear formulation. The spatially uniform
streamwise and spanwise pressure gradients remain in `problem.constraint`;
`P` stores only the periodic pressure field.

"""
function pressure(      U::VectorField{F},
                  problem::ChannelFlowProblem,
                        t::Real=0) where {F<:SpectralField{Float64}}
    g = grid(U[1])
    g === problem.grid || throw(ArgumentError("velocity and problem must share a grid"))
    R = similar(U)
    # Assemble every pressure-free contribution to the instantaneous
    # acceleration. Reuse solver_output for forcing, viscous derivatives
    # and the projected acceleration.
    solver_output = similar(U)
    nonlinear_flag = problem.nlterm.flag[]
    try
        problem.nlterm(t, U, R)
    finally
        # Pressure reconstruction must not consume one AlternatingForm call;
        # the first time-integration stage must see the same selected form.
        problem.nlterm.flag[] = nonlinear_flag
    end
    if !isnothing(problem.forcing)
        problem.forcing(t, U, solver_output)
        R .+= solver_output
    end

    for i = 1:3
        laplacian!(solver_output[i], U[i])
        R[i] .+= problem.scheme.nu .* solver_output[i]
    end
    @views R[1][:, 1, 1] .+= problem.scheme.nu .* parent(problem.scheme.basecurvature)

    # With nu=0 and lambda=1 the Stokes system is the instantaneous Helmholtz
    # projection a + grad(P) = R. Only P is retained in the initial state.
    pressure = similar(U[1])
    pressure_solver = StokesSolver(g, 0.0, 1.0)
    solve!(pressure_solver, solver_output, pressure, R)
    return pressure
end
