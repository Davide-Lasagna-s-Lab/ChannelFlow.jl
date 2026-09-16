export MeanModeSolver

#//////////////////////////////////////////////////////////////////////////////#
#///                  MEAN MODE AND BULK FLOW CONSTRAINTS                   ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    _bulkmean(a::ChebCoeffs)

Return `integral(a(y), y=-1:1)/2` from ordinary Chebyshev coefficients.
Only even degrees contribute: the mean of `T_n` is `1/(1-n²)` for even `n`.
"""
function _bulkmean(a::ChebCoeffs)
    value = a[0]
    for n = 2:2:length(a)-1
        value += a[n]/(1-n^2)
    end
    return value
end

"""
    MeanModeSolver(Ny, nu, lambda)

Cache the `(kx, kz) = (0, 0)` Stokes solve on `[-1, 1]`, with homogeneous
velocity wall values. `Ny ≥ 3` is the coefficient count, `nu ≥ 0` the
viscosity and `lambda ≥ 0` the temporal shift; both parameters must be finite
and cannot vanish simultaneously.

Allow `nu = 0` when the system is used as an instantaneous pressure
projection with positive `lambda`; the pair cannot both be zero.
One real Helmholtz factorisation serves both horizontal velocities and both
parts of their complex coefficients. Cache the response `c` to
`(nu*D² - lambda)c = 1`, with `c(±1) = 0`, and its bulk mean. This response
allows a prescribed flux to be imposed without another Helmholtz solve.
Rebuild the cache when any constructor argument changes.
"""
struct MeanModeSolver{H, C}
         velocity::H
         response::C
    response_mean::Float64
             work::C

    function MeanModeSolver(    Ny::Int,
                                nu::Real,
                            lambda::Real)
        Ny ≥ 3 || throw(ArgumentError("at least three Chebyshev coefficients are required"))
        nu, lambda = Float64(nu), Float64(lambda)
        isfinite(nu) && nu >= 0 || throw(ArgumentError("nu must be finite and non-negative"))
        isfinite(lambda) && lambda ≥ 0 ||
            throw(ArgumentError("lambda must be finite and nonnegative"))
        nu > 0 || lambda > 0 ||
            throw(ArgumentError("nu and lambda cannot both be zero"))

        velocity = HelmoltzSolver(Ny-1, Float64)
        update!(velocity, nu, lambda)
        response = ChebCoeffs(Ny-1, Float64)
        work = similar(response)
        response[0] = 1
        solve!(velocity, response, 0, 0)
        response_mean = _bulkmean(response)
        isfinite(response_mean) && response_mean != 0 ||
            throw(ArgumentError("the uniform-gradient response has no finite nonzero bulk mean"))
        return new{typeof(velocity), typeof(response)}(
            velocity, response, response_mean, work)
    end
end

"""
    solve!(solver::MeanModeSolver, u, v, w, p, Rx, Ry, Rz;
           pressuregradient=nothing, bulkvelocity=nothing)

Overwrite the zero Fourier mode, solving
```text
(lambda - nu*D²)u + dPdx = Rx,
(lambda - nu*D²)w + dPdz = Rz,
v = 0,  Dp = Ry,  u(±1) = w(±1) = 0.
```
Return `(dPdx, dPdz)`. Set `pressuregradient=(dPdx, dPdz)` to prescribe the
two uniform pressure derivatives, or `bulkvelocity=(umean, wmean)` to infer
them from the two bulk velocities. These options are mutually exclusive;
omitting both selects zero pressure gradient. The gradients appear with a
plus sign on the momentum left-hand side, so a negative `dPdx` drives flow
in the positive streamwise direction.

The velocities and bulk targets refer to the perturbation. The caller must
subtract the base-flow bulk mean from a total-flow target and include base
flow contributions in `R`. No base profile is added by this modal solver.

Reconstruct `p` by integrating `Ry` through degree `P-1`, with `P = Ny-1`,
and impose zero bulk mean on `p`. Degree `P` of `Ry` is the normal-momentum
tau residual: representing its integral would require degree `P+1`.
Pressure wall values are determined by momentum, not prescribed.

All seven fields must be `ChebCoeffs{ComplexF64}` of the solver's degree and
the same concrete storage type. Preserve the sources; outputs must have
distinct storage and must not overlap inputs or solver workspaces. For real
physical fields the zero-mode coefficients are real. Real and imaginary
parts are solved separately; bulk constraints apply to the real part.
The factorisation and response are reused; calls must not run concurrently.
"""
function solve!(          solver::MeanModeSolver,
                               u::Z,
                               v::Z,
                               w::Z,
                               p::Z,
                              Rx::Z,
                              Ry::Z,
                              Rz::Z;
                pressuregradient::Union{Nothing, NTuple{2, Real}}=nothing,
                    bulkvelocity::Union{Nothing, NTuple{2, Real}}=nothing) where {Z<:ChebCoeffs{ComplexF64}}
    length(p) == length(solver.work) ||
        throw(DimensionMismatch("mode coefficients must match the solver's degree"))
    isnothing(pressuregradient) || isnothing(bulkvelocity) ||
        throw(ArgumentError("specify either pressuregradient or bulkvelocity"))

    # Integrate the retained normal source. Omitting Ry[P] avoids feeding
    # its unrepresentable degree-P+1 integral back into lower coefficients.
    P = length(p) - 1
    for n = 1:P
        lower = n == 1 ? Ry[0] : Ry[n-1]/2
        upper = n+1 < P ? Ry[n+1]/2 : zero(eltype(Ry))
        p[n] = (lower-upper)/n
    end
    p[0] = 0
    p[0] = -_bulkmean(p)
    fill!(parent(v), 0)

    gradients = isnothing(pressuregradient) ? (0.0, 0.0) : pressuregradient
    return ntuple(2) do i
        out, source = i == 1 ? (u, Rx) : (w, Rz)
        for part in (real, imag)
            parent(solver.work) .= .-part.(parent(source))
            solve!(solver.velocity, solver.work, 0, 0)
            if part === real
                parent(out) .= parent(solver.work)
            else
                parent(out) .+= im .* parent(solver.work)
            end
        end

        # Superpose the cached unit-gradient response. For a bulk target,
        # its mean gives the scalar constraint on the unknown gradient.
        gradient = isnothing(bulkvelocity) ? gradients[i] :
                   (bulkvelocity[i]-real(_bulkmean(out)))/solver.response_mean
        parent(out) .+= gradient .* parent(solver.response)
        return gradient
    end
end
