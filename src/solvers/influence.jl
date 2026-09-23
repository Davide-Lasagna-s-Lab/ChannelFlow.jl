export InfluenceModeSolver, solve!

#//////////////////////////////////////////////////////////////////////////////#
#///                NONZERO MODE FACTORIZATION AND RESPONSES                ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    InfluenceModeSolver

Cache the real Chebyshev tau systems for the pressure and wall-normal velocity
of one nonzero Fourier mode on `y ∈ [-1, 1]`.

The cache stores two scalar Helmholtz factorisations, two wall influence
responses, their inverse influence matrix, and the auxiliary pressure/velocity
pair used by Gibson's tau correction. All coefficients and workspaces use
`Float64`; the same cache can process real and imaginary parts sequentially.

Construct it with [`InfluenceModeSolver(Ny, kx, kz, nu, lambda)`](@ref) and apply
it with [`solve!`](@ref). The mean mode requires a separate solver.
"""
struct InfluenceModeSolver{H, C}
             pressure::H                # factors for D² - kappa2
             velocity::H                # factors for nu*D² - lambda (full shift)
        pressure_plus::C                # pressure response to p(+1) = 1
       pressure_minus::C                # pressure response to p(-1) = 1
        velocity_plus::C                # velocity driven by pressure_plus'
       velocity_minus::C                # velocity driven by pressure_minus'
    influence_inverse::Matrix{Float64}  # rows/columns ordered as +1, -1
        pressure_zero::C                # auxiliary pressure for the tau correction
        velocity_zero::C                # auxiliary velocity for the tau correction
               sigma0::Vector{Float64}  # auxiliary tau coefficients at Ny-2, Ny-1
               lambda::Float64          # full shift: constructor lambda + nu*kappa2
                 work::C                # derivative and right-hand-side workspace
                   kx::Float64          # physical streamwise wavenumber
                   kz::Float64          # physical spanwise wavenumber
                cache::NTuple{4, C}     # real p, v, div(R), Ry for complex solves

    """
        InfluenceModeSolver(Ny, kx, kz, nu, lambda)

    Construct and factorise the pressure/velocity systems for a fixed Fourier mode
    and time-stepping coefficient. `Ny ≥ 5` is the odd number of Chebyshev
    coefficients, `kx` and `kz` are physical wavenumbers (including the factors
    `2π/Lx` and `2π/Lz`), `nu` is viscosity, and `lambda` is the temporal shift.
    The mode must be nonzero; its squared wavenumber is `kappa2 = kx² + kz²`.

    With `D = d/dy`, the scalar operators are `D² - kappa2` for pressure and
    `nu*D² - shift` for velocity, where `shift = lambda + nu*kappa2`. The stored
    `solver.lambda` is this full shift, matching the coefficient called
    `lambda_` in Channelflow's `TauSolver`.

    Precompute the wall influence responses and the auxiliary tau problem used by
    `TauSolver::solve_P_and_v`. An odd `Ny` makes the highest polynomial degree
    `P = Ny - 1` even, as required by the parity assignment in that correction.
    Rebuild the cache when the resolution, wavenumber, viscosity or shift changes.

    Throw `ArgumentError` for an unsupported `Ny`, the mean mode, or an
    influence matrix that fails the scale or relative-determinant checks.
    """
    function InfluenceModeSolver(    Ny::Int,
                                     kx::Real,
                                     kz::Real,
                                     nu::Real,
                                 lambda::Real)
        Ny ≥ 5 && isodd(Ny) ||
            throw(ArgumentError("Gibson's tau correction requires odd Ny ≥ 5"))
        kx, kz = Float64(kx), Float64(kz)
        kappa2 = kx^2 + kz^2
        kappa2 > 0 ||
            throw(ArgumentError("InfluenceModeSolver requires kappa2 > 0; handle the mean mode separately"))

        P = Ny - 1
        pressure = HelmoltzSolver(P, Float64)
        velocity = HelmoltzSolver(P, Float64)
        pplus, pminus, vplus, vminus, pzero, vzero, work =
            ntuple(_ -> zeros(Float64, P+1), 7)
        cache = ntuple(_ -> zeros(Float64, P+1), 4)
        shift = Float64(lambda + nu*kappa2)

        update!(pressure, 1, kappa2)
        update!(velocity, nu, shift)

        # Each homogeneous pressure solution has a
        # unit value at one wall and zero at the other; its derivative drives
        # a velocity response with zero values at both walls.
        # The backend takes boundary values in the order (+1, -1).
        solve!(pressure, pplus, work, 1, 0)
        diff!(work, pplus)
        solve!(velocity, vplus, work, 0, 0)

        fill!(work, 0)
        solve!(pressure, pminus, work, 0, 1)
        diff!(work, pminus)
        solve!(velocity, vminus, work, 0, 0)

        # Rows select v'(+1), v'(-1); columns select the plus/minus response.
        # Multiplying this matrix by the two pressure amplitudes gives the
        # resulting change in the wall-normal velocity derivative at the walls.
        influence = [
            diff(vplus, :right) diff(vminus, :right)
            diff(vplus, :left)  diff(vminus, :left)
        ]

        # Scale before checking the determinant: small response amplitudes alone
        # do not imply singularity. Undo the scaling when caching the inverse.
        scale = maximum(abs, influence)
        isfinite(scale) && scale > 0 ||
            throw(ArgumentError("the influence matrix has no finite nonzero scale"))
        influence ./= scale
        A, B = influence[1, 1], influence[1, 2]
        C, D = influence[2, 1], influence[2, 2]
        detA = A*D - B*C
        abs(detA) > eps(Float64)*max(abs(A*D), abs(B*C)) ||
            throw(ArgumentError("the influence matrix is numerically singular"))
        influence_inverse = [D -B; -C A] ./ detA ./ scale

        # Gibson's auxiliary B0 problem uses (T_{P-1} + T_P)' as the pressure
        # source. Both parities fit in one pair (pzero, vzero), so each of the
        # two momentum tau terms can later be corrected independently.
        fill!(work, 0)
        work[P] = work[P+1] = 1
        diff!(work)
        solve!(pressure, pzero, work, 0, 0)
        diff!(work, pzero)
        solve!(velocity, vzero, work, 0, 0)

        solver = new{typeof(pressure), typeof(pplus)}(
            pressure, velocity, pplus, pminus, vplus, vminus, influence_inverse,
            pzero, vzero, zeros(2), shift, work, kx, kz, cache)
        _influence_correction!(solver, pzero, vzero)

        # Both terms must use the influence-corrected auxiliary solution.
        # Mixing the old pressure derivative with the corrected velocity
        # leaves a discrete divergence residual, especially at low degree.
        # At degrees P-1 and P, vzero'' vanishes; pzero'[P] also vanishes.
        diff!(work, pzero)
        solver.sigma0[1] = shift*vzero[P] + work[P]
        solver.sigma0[2] = shift*vzero[P+1]
        return solver
    end
end

#//////////////////////////////////////////////////////////////////////////////#
#///                       WALL INFLUENCE CORRECTION                        ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    _influence_correction!(solver, p, v)

Modify the Chebyshev coefficients `p` and `v` so that `v'(+1) = v'(-1) = 0`.
The input pair must already satisfy the particular pressure/velocity tau
equations and `v(+1) = v(-1) = 0`.

Use the cached inverse influence matrix to obtain two pressure amplitudes,
then add the corresponding homogeneous responses to both fields. These
responses preserve the interior tau equations and the wall values of `v`.
The final pressure wall values are determined by the correction.

Return the modified pair `(p, v)`; the cached responses are preserved.
"""
function _influence_correction!(solver::InfluenceModeSolver{H, C},
                                     p::C,
                                     v::C) where {H, C}
    # Cancel the current derivative residuals, ordered as upper/lower wall.
    b₊ = -diff(v, :right)
    b₋ = -diff(v, :left)
    A = solver.influence_inverse
    δ₊ = A[1, 1]*b₊ + A[1, 2]*b₋
    δ₋ = A[2, 1]*b₊ + A[2, 2]*b₋

    # A pressure response and its velocity response share one amplitude.
    p .+= δ₊ .* solver.pressure_plus .+
                  δ₋ .* solver.pressure_minus
    v .+= δ₊ .* solver.velocity_plus .+
                  δ₋ .* solver.velocity_minus
    return p, v
end

#//////////////////////////////////////////////////////////////////////////////#
#///                       NONZERO MODE STOKES SOLVES                       ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    solve!(solver::InfluenceModeSolver, u, v, w, p, Rx, Ry, Rz)

Solve one nonzero complex Fourier mode of the primitive-variable Stokes system
```text
(solver.lambda - nu*D²) (u, v, w) + (i*kx*p, Dp, i*kz*p) = (Rx, Ry, Rz),
i*kx*u + Dv + i*kz*w = 0,
u(±1) = v(±1) = w(±1) = 0.
```
Here `D = d/dy` and `solver.lambda` includes `nu*(kx² + kz²)`. This is
the modal solve performed by Channelflow's `TauSolver::solve`.

All input and output fields are `AbstractVector{ComplexF64}` with the solver's
degree and the same concrete storage type. Overwrite `u, v, w, p` and return
`(u, v, w, p)`; preserve `Rx, Ry, Rz`. Outputs must have distinct storage
and must not overlap inputs or the solver's cache.

Solve the real and imaginary pressure/normal-velocity pairs separately,
including wall influence and tau corrections. Then solve for `u` and `w`
using the same real velocity factors. All temporary coefficient vectors are
cached; one solver instance must not be used concurrently.
"""
function solve!(solver::InfluenceModeSolver,
                     u::Z,
                     v::Z,
                     w::Z,
                     p::Z,
                    Rx::Z,
                    Ry::Z,
                    Rz::Z) where {Z<:AbstractVector{ComplexF64}}
    length(p) == length(solver.work) ||
        throw(DimensionMismatch("mode coefficients must match the solver's degree"))
    pc, vc, r, ry = solver.cache

    # Re(div R) = D Re(Ry) - kx Im(Rx) - kz Im(Rz);
    # Im(div R) = D Im(Ry) + kx Re(Rx) + kz Re(Rz).
    # Keep the real/imaginary selection scalar: iterating over function
    # objects here makes broadcast construction type-unstable in this hot loop.
    for imaginary in (false, true)
        @inbounds for n in eachindex(ry)
            ry[n] = imaginary ? imag(Ry[n]) : real(Ry[n])
        end
        diff!(r, ry)
        @inbounds for n in eachindex(r)
            r[n] += imaginary ? solver.kx*real(Rx[n]) + solver.kz*real(Rz[n]) :
                               -solver.kx*imag(Rx[n]) - solver.kz*imag(Rz[n])
        end
        solve!(solver, pc, vc, r, ry)

        @inbounds for n in eachindex(p)
            if imaginary
                p[n] += im*pc[n]
                v[n] += im*vc[n]
            else
                p[n] = pc[n]
                v[n] = vc[n]
            end
        end
    end

    # The horizontal momentum sources are i*k*p - R. Reuse the p/v
    # workspaces for their real/imaginary parts and the cached velocity factors.
    for (out, source, k) in ((u, Rx, solver.kx), (w, Rz, solver.kz))
        pc .= -k .* imag.(p) .- real.(source)
        vc .=  k .* real.(p) .- imag.(source)
        solve!(solver.velocity, ry, pc, 0, 0)
        solve!(solver.velocity, r, vc, 0, 0)
        out .= complex.(ry, r)
    end
    return u, v, w, p
end

"""
    solve!(solver::InfluenceModeSolver, p, v, r, Ry)

Overwrite `p` and `v` with the pressure and wall-normal velocity of one
nonzero Fourier mode, using Gibson's influence-matrix and tau corrections.
All four arguments are ordinary real coefficient vectors of degree
`P = Ny - 1`, indexed from `1` to `P+1`.

The particular problems are
```text
(D² - kappa2) p = r,
(nu*D² - solver.lambda) v = Dp - Ry,
```
where `D = d/dy`. For a momentum source `R`, `r` is the corresponding real
or imaginary part of its full Fourier divergence
`i*kx*Rx + D*Ry + i*kz*Rz`, and `Ry` is the matching part of the wall-normal
source. Here `kx` and `kz` are physical wavenumbers.

The influence correction imposes `v = Dv = 0` at both walls. The subsequent
tau correction accounts for the two highest momentum residual coefficients
and their contribution to the pressure equation, preserving these wall
conditions. Pressure wall values are not prescribed independently.

Return `(p, v)`. Inputs `r` and `Ry` are preserved: all four vectors must use
distinct storage and must not alias the solver's cache. The solve overwrites
internal workspaces, so one solver instance must not be used concurrently.
"""
function solve!(solver::InfluenceModeSolver{H, C},
                     p::C,
                     v::C,
                     r::C,
                    Ry::C) where {H, C}
    # Zero pressure wall values select a particular solution; they are
    # replaced by the values required by the influence correction below.
    solve!(solver.pressure, p, r, 0, 0)

    # Form Dp - Ry and solve for v with homogeneous wall values.
    diff!(solver.work, p)
    solver.work .-= Ry
    solve!(solver.velocity, v, solver.work, 0, 0)

    _influence_correction!(solver, p, v)

    # Scalar tau solves leave degrees P-1 and P unconstrained. Evaluate the
    # residual solver.lambda*v + Dp - nu*D²v - Ry at those degrees.
    # D²v has degree at most P-2, so its contribution is exactly zero here.
    P = length(p) - 1
    diff!(solver.work, p)
    sigmaNm1 = (solver.lambda*v[P] + solver.work[P] - Ry[P]) /
               (1 - solver.sigma0[1])
    sigmaN = (solver.lambda*v[P+1] + solver.work[P+1] - Ry[P+1]) /
             (1 - solver.sigma0[2])

    # sigma_m = sigma1_m/(1 - sigma0_m) includes the auxiliary response's
    # own tau contribution. Since P is even and differentiation swaps
    # parity, the pressure correction uses the opposite parity to velocity.
    @inbounds for n = 1:P+1
        p[n] += (iseven(n-1) ? sigmaNm1 : sigmaN) * solver.pressure_zero[n]
        v[n] += (iseven(n-1) ? sigmaN : sigmaNm1) * solver.velocity_zero[n]
    end
    return p, v
end
