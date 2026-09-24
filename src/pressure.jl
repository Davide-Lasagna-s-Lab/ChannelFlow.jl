export project!, pressure

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

Require resolved `ComplexF64` fields on one grid, odd `Ny ≥ 5`, independent
component storage and Fourier conjugate symmetry for a real velocity.
Allocate factors and workspaces on each call: intended for initialisation.
"""
function project!(U::VectorField{F}, problem::ChannelFlowProblem) where {F<:SpectralField}

    # First make the supplied perturbation velocity satisfy continuity,
    # no slip and, when requested, the total bulk-flow constraint.
    solver = _same_storage(StokesSolver(problem.grid, 1.0, 0.0), U[1])
    R, phi = similar(U), similar(U[1])
    
    # Form the complete source before solve! overwrites any velocity component.
    laplacian!(R, U); R .*= -1

    bulkvelocity = haskey(problem.constraint, :bulkvelocity) ?
                   problem.constraint.bulkvelocity : nothing

    solve!(solver, U, phi, R;
           bulkvelocity=bulkvelocity,
           baseflow=problem.scheme.baseflow)

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

Form the non-viscous acceleration
```text
A = nonlinear(U + Ub) + forcing.
```
Solve the scalar pressure problem on the fixed wall-normal interval [-1, 1]:
```text
Delta(P) = div(A),  dP/dy = A_y + nu*d²v/dy² at y = ±1.
```
Incompressibility eliminates viscosity from the interior source. No slip
eliminates horizontal derivatives of the normal velocity at the walls, leaving
only its second wall-normal derivative there. The parallel base flow has no
normal component and contributes no viscous pressure term.
These wall conditions impose zero normal acceleration for stationary wall
velocities; no tangential acceleration condition is imposed. Nonzero Fourier
modes use the structured Chebyshev Helmholtz tau solver with Neumann
boundary conditions. The mean mode integrates the retained normal acceleration
and fixes the volume mean pressure to zero. Its highest source coefficient
is a tau residual, as in the mean-mode momentum solver.
This routine allocates and factorises per call for initialisation.
Consequently `P` is the ordinary physical pressure for the convective and
divergence forms. For the rotational form it is the modified
pressure `P = p + abs(U + Ub)^2/2`, because the kinetic-energy gradient has
already been absorbed into the nonlinear formulation. The spatially uniform
streamwise and spanwise pressure gradients remain in `problem.constraint`;
`P` stores only the periodic pressure field.

"""
function pressure(U::VectorField{F}, problem::ChannelFlowProblem, t::Real=0) where {F<:SpectralField{Float64}}
    # this is the work array
    R = similar(U)

    # nonlinearity overwrites R; forcing adds to it
    problem.nlterm(t, U, R)
    problem.forcing(t, U, R)

    return _pressure_poisson(R, U[2], problem.scheme.nu)
end

"""
    _pressure_poisson(A::VectorField, v::SpectralField, nu)

Return periodic pressure from `Delta(P) = div(A)` and
`P_y(±1) = A_y(±1) + nu*v_yy(±1)`, for divergence-free, no-slip velocity.
Use ordinary Chebyshev coefficients, retain the lowest `Ny-2` differential
equations for nonzero modes, and omit excluded Nyquist planes. The zero mode
satisfies `P_y = A_y` through degree `Ny-2`, with zero volume mean pressure.
"""
function _pressure_poisson(A::VectorField{F}, v::F, nu::Real) where {F<:SpectralField{Float64}}

    # Grid geometry and resolved Fourier dimensions; y is the last array axis.
    g = grid(A[1])
    Nxh, Nz, Ny = size(A[1])
    Ny >= 3 || throw(ArgumentError("pressure reconstruction requires Ny ≥ 3"))
    Lx, _, Lz = domainsize(g)
    Nx, _, _ = physicalsize(g, NotPadded())

    # Initially P holds the Poisson source. Each modal solve overwrites its
    # own column with pressure coefficients, leaving A and v unchanged.
    P = similar(A[1])
    div!(P, A)

    # Reuse the structured Neumann tau solver for each nonzero Fourier mode.
    # Real and imaginary sources share factors and a real coefficient buffer.
    solver = HelmoltzSolver(Ny-1; neum=true)
    work = zeros(Ny)
    solution = similar(work)

    # NONZERO FOURIER MODES
    # Solve (Dyy - k² - l²) P = div(A) with wall-normal momentum conditions.
    for iz = 1:Nz, ix = 1:Nxh

        # The singular mean mode is handled below; Nyquist planes are excluded.
        (ix == 1 && iz == 1) && continue
        ((iseven(Nx) && ix == Nxh) ||
         (iseven(Nz) && iz == (Nz >> 1)+1)) && continue

        # x contains the nonnegative half-spectrum; z follows signed FFT order.
        k = (2π/Lx)*(ix-1)
        l = (2π/Lz)*(iz <= (Nz >> 1)+1 ? iz-1 : iz-1-Nz)
        update!(solver, 1.0, k^2 + l^2)

        rhs = view(parent(P), ix, iz, :)
        normal = view(parent(A[2]), ix, iz, :)

        # WALL DATA: P_y = A_y + nu*v_yy at y = +1 and -1.
        # Both derivatives point along +y, rather than the outward wall normal.
        # T_n(±1) = (±1)^n evaluates the normal acceleration.
        # T_n''(+1) = n²(n²-1)/3 and T_n''(-1) = (-1)^n T_n''(+1).
        # Evaluate the viscous wall terms directly without a derivative field.
        upper = sum(normal)
        lower = sum((-1)^(j-1)*normal[j] for j = 1:Ny)

        for n = 2:Ny-1
            wall = nu * (n^2*(n^2-1)/3) * v[ix, iz, n+1]
            upper += wall
            lower += iseven(n) ? wall : -wall
        end

        # REAL PART: reuse the modal factors with a real coefficient workspace.
        parent(work) .= real.(rhs)
        solve!(solver, solution, work, real(upper), real(lower))

        # Preserve the imaginary source until its own solve is complete.
        rhs .= complex.(parent(solution), imag.(rhs))

        # IMAGINARY PART: solve with the same factors, then assemble complex P.
        parent(work) .= imag.(rhs)
        solve!(solver, solution, work, imag(upper), imag(lower))
        rhs .= complex.(real.(rhs), parent(solution))
    end

    # ZERO FOURIER MODE
    # For the mean mode, incompressibility and impermeability give v = 0
    # throughout the channel, so there is no viscous contribution. Thus
    # P_y = A_y. Truncate the unrepresentable highest-degree primitive and
    # choose the constant coefficient to give zero physical volume mean.
    normal = view(parent(A[2]), 1, 1, :)
    mean = view(parent(P), 1, 1, :)

    # Integrate the retained Chebyshev series using neighbouring coefficients.
    # The constant source has a different normalization from higher degrees.
    for n = 1:Ny-1
        lower = n == 1 ? normal[1] : normal[n]/2
        upper = n+1 < Ny-1 ? normal[n+2]/2 : zero(eltype(normal))
        mean[n+1] = (lower-upper)/n
    end

    # Fix the additive constant by the physical volume mean, not by setting
    # the zeroth Chebyshev coefficient alone to zero.
    mean[1] = 0
    mean[1] = -_bulkmean(mean)

    # Restore the spectral convention after all retained modes are solved.
    zero_nyquist!(P)

    return P
end
