export CNRK2, step!

# Channelflow's RungeKuttaDNS coefficients (dnsalgo.cpp), for signed N = -u⋅∇u.
const CNRK2_A = (0.0, -5/9, -153/128)
const CNRK2_B = (1/3, 15/16, 8/15)
const CNRK2_C = (1/6, 5/24, 1/8)

"""
    CNRK2(U::VectorField, nu, dt)

Allocate Gibson's three-stage CN-RK2 time-stepper for spectral velocity `U`,
viscosity `nu > 0` and fixed, finite `dt > 0`. `U` supplies the grid and
`ComplexF64` storage type; its values are preserved.

Cache three [`FourierStokesSolver`](@ref) instances with temporal shifts
`1/(C_j*dt)`, the accumulated explicit term `Q`, the current explicit term
`N`, the stage source `R`, and the stationary base-flow term `nu*Ub''`.
`N` is reused for pressure derivatives after updating `Q`.

This is the three-stage RK/CN algorithm in Channelflow's `RungeKuttaDNS`,
with overall order two. Rebuild the cache when `dt`, viscosity, grid or base
profile changes. The caller retains velocity and stage pressure between steps.
"""
struct CNRK2{S, F<:SpectralField{Float64}, C} <: Flows.AbstractMethod{Flows.Coupled{2, Tuple{VectorField{F}, F}}, Flows.NormalMode, 3}
             nu::Float64
             dt::Float64
        solvers::NTuple{3, S}
              Q::VectorField{F}
              N::VectorField{F}
              R::VectorField{F}
    baseviscous::C

    function CNRK2( U::VectorField{F},
                   nu::Real,
                   dt::Real) where {F<:SpectralField{Float64}}
        nu, dt = Float64(nu), Float64(dt)
        isfinite(dt) && dt > 0 || throw(ArgumentError("dt must be finite and positive"))
        g = grid(U[1])
        for field in U.components
            grid(field) === g || throw(ArgumentError("velocity components must share a grid"))
            size(field) == spectralsize(g, NotPadded()) ||
                throw(DimensionMismatch("velocity must have the resolved spectral size"))
        end
        solvers = ntuple(j -> FourierStokesSolver(g, nu, inv(CNRK2_C[j]*dt)), 3)
        Q, N, R = ntuple(_ -> similar(U), 3)
        baseviscous = ChebyshevHelmoltzSolvers.ChebCoeffs(copy(baseflow(g)))
        ChebyshevHelmoltzSolvers.diff!(baseviscous, baseviscous)
        ChebyshevHelmoltzSolvers.diff!(baseviscous, baseviscous)
        parent(baseviscous) .*= nu
        return new{typeof(solvers[1]), F, typeof(baseviscous)}(
            nu, dt, solvers, Q, N, R, baseviscous)
    end
end

"""
    _bulkpressuregradient(scheme, U, N)

Return the instantaneous uniform pressure derivatives required for constant
bulk velocity. Use the mean of the total viscous and signed explicit
acceleration. Viscosity is evaluated from the two wall shears; `N` includes
any added body force. For divergence-free, periodic convection its bulk
contribution vanishes, recovering Channelflow's `NSE::linear` wall-stress rule.
"""
function _bulkpressuregradient(scheme::CNRK2,
                                    U::VectorField,
                                    N::VectorField)
    return ntuple(2) do i
        component = i == 1 ? 1 : 3
        u = _chebcolumn(U[component], 1, 1)
        shear = ChebyshevHelmoltzSolvers.endpoint_derivative(u, :right) -
                ChebyshevHelmoltzSolvers.endpoint_derivative(u, :left)
        base = i == 1 ? _bulkmean(scheme.baseviscous) : 0.0
        return scheme.nu*real(shear)/2 + base + real(_bulkmean(_chebcolumn(N[component], 1, 1)))
    end
end

"""
    step!(scheme::CNRK2, nonlinear, U, P, t;
          forcing=nothing, pressuregradient=nothing, bulkvelocity=nothing,
          stagecache=nothing)

Advance perturbation velocity `U` and stage pressure `P` by `scheme.dt`.
Return `(t + scheme.dt, (dPdx, dPdz))`, with the uniform pressure derivatives
from the final implicit stage. `U` and `P` must be resolved fields on the
scheme's grid, with distinct storage that does not overlap its caches.

`nonlinear(tstage, U, N)` must overwrite `N` with signed nonlinear
acceleration, using the total velocity including the base flow. An existing
[`NonLinearTerm`](@ref) supplies this contract. If provided,
`forcing(tstage, U, F)` must overwrite all three components of `F` with an
additional spectral acceleration. Both callbacks preserve `U` and are
sampled at `t + (0, 1/3, 3/4)*dt`.

Use either `pressuregradient=(dPdx, dPdz)` or total
`bulkvelocity=(Ubulk, Wbulk)`, as in [`FourierStokesSolver`](@ref).
Omitting both imposes zero pressure gradient. For a bulk constraint, the
initial velocity must already satisfy the target; every stage enforces it.

Each stage updates `Q = A_j*Q + N` and solves
```text
(lambda_j - nu*Δ) Unew + grad(Pnew) + Gnew =
    lambda_j*U + nu*ΔU - grad(P) - Gold + (B_j/C_j)*Q + 2nu*Ub''*e_x,
lambda_j = 1/(C_j*dt).
```
The base diffusion appears twice because it contributes to both CN halves;
the new uniform gradient is supplied to the implicit mean-mode solve.
`Q` is reset at the first stage of every step. `P` is overwritten at each
stage and retained for the next, as in Gibson's `RungeKuttaDNS`/`NSE` split.

Supply a consistent initial pressure for accuracy from the first step. For
[`RotatingForm`](@ref), `P` is the modified pressure including total kinetic
energy per unit mass; otherwise it is the ordinary kinematic pressure.
The scheme's workspaces and callbacks must not be used concurrently.

If `stagecache` is a `Flows.AbstractStageCache{3}`, save independent copies
of `Flows.couple(U, P)` before the three explicit evaluations. Stage recording
allocates snapshots only when requested; it does not enable tangent or adjoint
integration, which requires separate implementations.
"""
function step!(          scheme::CNRK2{S, F},
                      nonlinear,
                              U::VectorField{F},
                              P::F,
                              t::Real;
                        forcing=nothing,
               pressuregradient::Union{Nothing, NTuple{2, Real}}=nothing,
                   bulkvelocity::Union{Nothing, NTuple{2, Real}}=nothing,
                     stagecache::Union{Nothing, Flows.AbstractStageCache{3}}=nothing) where {S, F<:SpectralField{Float64}}
    isnothing(pressuregradient) || isnothing(bulkvelocity) ||
        throw(ArgumentError("specify either pressuregradient or bulkvelocity"))
    g = scheme.solvers[1].grid
    for field in (U.components..., P)
        grid(field) === g || throw(ArgumentError("fields must use the scheme's grid"))
        size(field) == spectralsize(g, NotPadded()) ||
            throw(DimensionMismatch("fields must have the resolved spectral size"))
    end
    Q, N, R = scheme.Q, scheme.N, scheme.R
    gradients = isnothing(pressuregradient) ? (0.0, 0.0) : pressuregradient
    state = Flows.couple(U, P)
    stages = isnothing(stagecache) ? nothing : Vector{typeof(state)}(undef, 3)

    for j = 1:3
        isnothing(stagecache) || (stages[j] = copy(state))
        tstage = t + (0.0, 1/3, 3/4)[j]*scheme.dt
        nonlinear(tstage, U, N)
        if !isnothing(forcing)
            forcing(tstage, U, R)
            N .+= R
        end
        oldgradient = isnothing(bulkvelocity) ? gradients : _bulkpressuregradient(scheme, U, N)

        # The first stage must overwrite Q: 0*uninitialised storage is unsafe,
        # and nonlinear history from the previous step must not survive.
        if j == 1
            Q .= N
        else
            Q .= CNRK2_A[j] .* Q .+ N
        end

        lambda = inv(CNRK2_C[j]*scheme.dt)
        weight = CNRK2_B[j]/CNRK2_C[j]
        for (i, derivative!) in enumerate((ddx1!, ddx2!, ddx3!))
            laplacian!(R[i], U[i])
            derivative!(N[i], P)
            R[i] .= lambda .* U[i] .+ scheme.nu .* R[i] .- N[i] .+ weight .* Q[i]
        end
        @views R[1][:, 1, 1] .+= 2 .* parent(scheme.baseviscous)
        R[1][1, 1, 1] -= oldgradient[1]
        R[3][1, 1, 1] -= oldgradient[2]

        gradients = solve!(scheme.solvers[j], U, P, R;
                           pressuregradient=pressuregradient, bulkvelocity=bulkvelocity)
    end
    isnothing(stagecache) || push!(stagecache, t, scheme.dt, Tuple(stages))
    return t+scheme.dt, gradients
end
