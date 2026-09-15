import ChebyshevHelmoltzSolvers

export InfluenceModeSolver, MeanModeSolver, FourierStokesSolver, solve!

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
    and time-stepping coefficient. `Ny ≥ 3` is the odd number of Chebyshev
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
        Ny ≥ 3 && isodd(Ny) ||
            throw(ArgumentError("Gibson's tau correction requires odd Ny ≥ 3"))
        kx, kz = Float64(kx), Float64(kz)
        kappa2 = kx^2 + kz^2
        kappa2 > 0 ||
            throw(ArgumentError("InfluenceModeSolver requires kappa2 > 0; handle the mean mode separately"))

        P = Ny - 1
        pressure = ChebyshevHelmoltzSolvers.HelmoltzSolver(P, Float64)
        velocity = ChebyshevHelmoltzSolvers.HelmoltzSolver(P, Float64)
        pplus, pminus, vplus, vminus, pzero, vzero, work =
            ntuple(_ -> ChebyshevHelmoltzSolvers.ChebCoeffs(P, Float64), 7)
        cache = ntuple(_ -> ChebyshevHelmoltzSolvers.ChebCoeffs(P, Float64), 4)
        shift = Float64(lambda + nu*kappa2)

        ChebyshevHelmoltzSolvers.update!(pressure, 1, kappa2)
        ChebyshevHelmoltzSolvers.update!(velocity, nu, shift)

        # ChebCoeffs starts at zero. Each homogeneous pressure solution has a
        # unit value at one wall and zero at the other; its derivative drives
        # a velocity response with zero values at both walls.
        # The backend takes boundary values in the order (+1, -1).
        ChebyshevHelmoltzSolvers.solve!(pressure, pplus, 1, 0)
        ChebyshevHelmoltzSolvers.diff!(work, pplus)
        parent(vplus) .= parent(work)
        ChebyshevHelmoltzSolvers.solve!(velocity, vplus, 0, 0)

        ChebyshevHelmoltzSolvers.solve!(pressure, pminus, 0, 1)
        ChebyshevHelmoltzSolvers.diff!(work, pminus)
        parent(vminus) .= parent(work)
        ChebyshevHelmoltzSolvers.solve!(velocity, vminus, 0, 0)

        # Rows select v'(+1), v'(-1); columns select the plus/minus response.
        # Multiplying this matrix by the two pressure amplitudes gives the
        # resulting change in the wall-normal velocity derivative at the walls.
        influence = [
            ChebyshevHelmoltzSolvers.endpoint_derivative(vplus, :right) ChebyshevHelmoltzSolvers.endpoint_derivative(vminus, :right)
            ChebyshevHelmoltzSolvers.endpoint_derivative(vplus, :left)  ChebyshevHelmoltzSolvers.endpoint_derivative(vminus, :left)
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
        fill!(parent(work), 0)
        work[P-1] = work[P] = 1
        ChebyshevHelmoltzSolvers.diff!(pzero, work)
        ChebyshevHelmoltzSolvers.solve!(pressure, pzero, 0, 0)
        ChebyshevHelmoltzSolvers.diff!(vzero, pzero)
        dPzeroNm1 = vzero[P-1]
        ChebyshevHelmoltzSolvers.solve!(velocity, vzero, 0, 0)

        solver = new{typeof(pressure), typeof(pplus)}(
            pressure, velocity, pplus, pminus, vplus, vminus, influence_inverse,
            pzero, vzero, zeros(2), shift, work, kx, kz, cache)
        _influence_correction!(solver, pzero, vzero)

        # Match Gibson's constructor: retain pzero' from BEFORE the influence
        # correction, but use the corrected vzero. At degrees P-1 and P,
        # vzero'' vanishes; at degree P, pzero' also vanishes.
        solver.sigma0[1] = shift*vzero[P-1] + dPzeroNm1
        solver.sigma0[2] = shift*vzero[P]
        return solver
    end
end

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
    b₊ = -ChebyshevHelmoltzSolvers.endpoint_derivative(v, :right)
    b₋ = -ChebyshevHelmoltzSolvers.endpoint_derivative(v, :left)
    A = solver.influence_inverse
    δ₊ = A[1, 1]*b₊ + A[1, 2]*b₋
    δ₋ = A[2, 1]*b₊ + A[2, 2]*b₋

    # A pressure response and its velocity response share one amplitude.
    parent(p) .+= δ₊ .* parent(solver.pressure_plus) .+
                  δ₋ .* parent(solver.pressure_minus)
    parent(v) .+= δ₊ .* parent(solver.velocity_plus) .+
                  δ₋ .* parent(solver.velocity_minus)
    return p, v
end

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

All input and output fields are `ChebCoeffs{ComplexF64}` with the solver's
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
                    Rz::Z) where {Z<:ChebyshevHelmoltzSolvers.ChebCoeffs{ComplexF64}}
    length(p) == length(solver.work) ||
        throw(DimensionMismatch("mode coefficients must match the solver's degree"))
    pc, vc, r, ry = solver.cache

    # Re(div R) = D Re(Ry) - kx Im(Rx) - kz Im(Rz);
    # Im(div R) = D Im(Ry) + kx Re(Rx) + kz Re(Rz).
    for (part, cross, sign) in ((real, imag, -1), (imag, real, 1))
        parent(ry) .= part.(parent(Ry))
        ChebyshevHelmoltzSolvers.diff!(r, ry)
        parent(r) .+= sign .* (solver.kx .* cross.(parent(Rx)) .+
                              solver.kz .* cross.(parent(Rz)))
        solve!(solver, pc, vc, r, ry)

        if part === real
            parent(p) .= parent(pc)
            parent(v) .= parent(vc)
        else
            parent(p) .+= im .* parent(pc)
            parent(v) .+= im .* parent(vc)
        end
    end

    # The horizontal momentum sources are i*k*p - R. Reuse the p/v
    # workspaces for their real/imaginary parts and the cached velocity factors.
    for (out, source, k) in ((u, Rx, solver.kx), (w, Rz, solver.kz))
        parent(pc) .= -k .* imag.(parent(p)) .- real.(parent(source))
        parent(vc) .=  k .* real.(parent(p)) .- imag.(parent(source))
        ChebyshevHelmoltzSolvers.solve!(solver.velocity, pc, 0, 0)
        ChebyshevHelmoltzSolvers.solve!(solver.velocity, vc, 0, 0)
        parent(out) .= complex.(parent(pc), parent(vc))
    end
    return u, v, w, p
end

"""
    solve!(solver::InfluenceModeSolver, p, v, r, Ry)

Overwrite `p` and `v` with the pressure and wall-normal velocity of one
nonzero Fourier mode, using Gibson's influence-matrix and tau corrections.
All four arguments are ordinary real `ChebCoeffs` expansions of degree
`P = Ny - 1`, indexed from `0` to `P`.

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
    parent(p) .= parent(r)
    ChebyshevHelmoltzSolvers.solve!(solver.pressure, p, 0, 0)

    # Form Dp - Ry and solve for v with homogeneous wall values.
    ChebyshevHelmoltzSolvers.diff!(solver.work, p)
    parent(solver.work) .-= parent(Ry)
    parent(v) .= parent(solver.work)
    ChebyshevHelmoltzSolvers.solve!(solver.velocity, v, 0, 0)

    _influence_correction!(solver, p, v)

    # Scalar tau solves leave degrees P-1 and P unconstrained. Evaluate the
    # residual solver.lambda*v + Dp - nu*D²v - Ry at those degrees.
    # D²v has degree at most P-2, so its contribution is exactly zero here.
    P = length(p) - 1
    ChebyshevHelmoltzSolvers.diff!(solver.work, p)
    sigmaNm1 = (solver.lambda*v[P-1] + solver.work[P-1] - Ry[P-1]) /
               (1 - solver.sigma0[1])
    sigmaN = (solver.lambda*v[P] + solver.work[P] - Ry[P]) /
             (1 - solver.sigma0[2])

    # sigma_m = sigma1_m/(1 - sigma0_m) includes the auxiliary response's
    # own tau contribution. Since P is even and differentiation swaps
    # parity, the pressure correction uses the opposite parity to velocity.
    @inbounds for n = 0:P
        p[n] += (iseven(n) ? sigmaNm1 : sigmaN) * solver.pressure_zero[n]
        v[n] += (iseven(n) ? sigmaN : sigmaNm1) * solver.velocity_zero[n]
    end
    return p, v
end

"""
    _bulkmean(a::ChebyshevHelmoltzSolvers.ChebCoeffs)

Return `integral(a(y), y=-1:1)/2` from ordinary Chebyshev coefficients.
Only even degrees contribute: the mean of `T_n` is `1/(1-n²)` for even `n`.
"""
function _bulkmean(a::ChebyshevHelmoltzSolvers.ChebCoeffs)
    value = a[0]
    for n = 2:2:length(a)-1
        value += a[n]/(1-n^2)
    end
    return value
end

"""
    MeanModeSolver(Ny, nu, lambda)

Cache the `(kx, kz) = (0, 0)` Stokes solve on `[-1, 1]`, with homogeneous
velocity wall values. `Ny ≥ 3` is the coefficient count, `nu > 0` the
viscosity and `lambda ≥ 0` the temporal shift; both parameters must be finite.

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
        isfinite(nu) && nu > 0 || throw(ArgumentError("nu must be finite and positive"))
        isfinite(lambda) && lambda ≥ 0 ||
            throw(ArgumentError("lambda must be finite and nonnegative"))

        velocity = ChebyshevHelmoltzSolvers.HelmoltzSolver(Ny-1, Float64)
        ChebyshevHelmoltzSolvers.update!(velocity, nu, lambda)
        response = ChebyshevHelmoltzSolvers.ChebCoeffs(Ny-1, Float64)
        work = similar(response)
        response[0] = 1
        ChebyshevHelmoltzSolvers.solve!(velocity, response, 0, 0)
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
                    bulkvelocity::Union{Nothing, NTuple{2, Real}}=nothing) where {Z<:ChebyshevHelmoltzSolvers.ChebCoeffs{ComplexF64}}
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
            ChebyshevHelmoltzSolvers.solve!(solver.velocity, solver.work, 0, 0)
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

"""
    FourierStokesSolver(grid, nu, lambda)

Cache the serial primitive-variable Stokes solve for all resolved Fourier
modes on `grid`. `nu` is viscosity and `lambda` the temporal shift, as in
[`InfluenceModeSolver`](@ref) and [`MeanModeSolver`](@ref).

Store a separate mean-mode solver and an influence solver for each active
nonzero mode. `modes[ix, iz]` follows the FFT storage order: nonnegative `kx`,
then positive and negative `kz` in their FFT slots. The mean and excluded
Nyquist slots contain `nothing`; they allocate no influence systems.
Physical wavenumbers include `2π/Lx` and `2π/Lz`.

Require odd `Ny ≥ 3`, positive periodic sizes and finite positive `Lx, Lz`.
Rebuild when the grid, viscosity or temporal shift changes. Factors and
workspaces are reused, so one instance must not be used concurrently.
"""
struct FourierStokesSolver{G, S, M}
     grid::G
    modes::S
     mean::M

    function FourierStokesSolver(  grid::Grid,
                                    nu::Real,
                                lambda::Real)
        Ny, Nx, Nz = physicalsize(grid, NotPadded())
        Lx, _, Lz = domainsize(grid)
        isodd(Ny) || throw(ArgumentError("Gibson's tau correction requires odd Ny"))
        Nx > 0 && Nz > 0 || throw(ArgumentError("Nx and Nz must be positive"))
        isfinite(Lx) && isfinite(Lz) && Lx > 0 && Lz > 0 ||
            throw(ArgumentError("Lx and Lz must be finite and positive"))

        mean = MeanModeSolver(Ny, nu, lambda)
        _, Nxh, _ = spectralsize(grid, NotPadded())
        modes = [if (ix == 1 && iz == 1) ||
                    (iseven(Nx) && ix == Nxh) ||
                    (iseven(Nz) && iz == (Nz >> 1)+1)
                     nothing
                 else
                     kx = (2π/Lx)*(ix-1)
                     kz = (2π/Lz)*(iz <= (Nz >> 1)+1 ? iz-1 : iz-1-Nz)
                     InfluenceModeSolver(Ny, kx, kz, nu, lambda)
                 end for ix = 1:Nxh, iz = 1:Nz]
        return new{typeof(grid), typeof(modes), typeof(mean)}(grid, modes, mean)
    end
end

"""Wrap Fourier slot `(ix, iz)` as Chebyshev coefficients without copying data."""
_chebcolumn(U::SpectralField, ix::Int, iz::Int) =
    ChebyshevHelmoltzSolvers.ChebCoeffs(view(parent(U), :, ix, iz))

"""
    solve!(solver::FourierStokesSolver, U, P, R;
           pressuregradient=nothing, bulkvelocity=nothing, baseflow=nothing)

Overwrite the perturbation velocity `U::VectorField` and pressure
`P::SpectralField` from the momentum source `R::VectorField`. All component
fields must store `ComplexF64` coefficients on the solver's grid, with the
resolved size `spectralsize(grid, NotPadded())`.

Apply the modal Stokes systems through views of contiguous Chebyshev columns.
Treat the mean separately and set all output Nyquist planes to zero. Sources
are preserved, including excluded modes. Outputs must have distinct storage
and must not overlap inputs or solver workspaces. Input spectra must satisfy
the conjugate symmetry required for real physical fields.

Return `(dPdx, dPdz)`. The mutually exclusive keywords have the same meaning
as in [`solve!(::MeanModeSolver, u, v, w, p, Rx, Ry, Rz)`](@ref), except that
`bulkvelocity=(Ubulk, Wbulk)` specifies TOTAL bulk velocities here. If
`baseflow` is supplied as ordinary Chebyshev coefficients, its streamwise mean
is subtracted to obtain the perturbation target; otherwise the target is
interpreted as a perturbation target. The grid itself has no base flow.

The stage assembly remains responsible for placing base-flow viscous terms
and any explicit forcing in `R`. This solve does not add them or advance time.
"""
function solve!(          solver::FourierStokesSolver,
                               U::VectorField{F},
                               P::F,
                               R::VectorField{F};
                pressuregradient::Union{Nothing, NTuple{2, Real}}=nothing,
                    bulkvelocity::Union{Nothing, NTuple{2, Real}}=nothing,
                         baseflow=nothing) where {F<:SpectralField{Float64}}
    fields = (U.components..., P, R.components...)
    expected = spectralsize(solver.grid, NotPadded())
    for field in fields
        grid(field) === solver.grid || throw(ArgumentError("fields must use the solver's grid"))
        size(field) == expected || throw(DimensionMismatch("fields must have the resolved spectral size"))
    end

    # Only the mean mode sees the uniform pressure gradient or bulk target.
    target = isnothing(bulkvelocity) ? nothing :
             (bulkvelocity[1] - (isnothing(baseflow) ? 0.0 :
                _bulkmean(ChebyshevHelmoltzSolvers.ChebCoeffs(baseflow))),
              bulkvelocity[2])
    gradients = solve!(solver.mean, map(field -> _chebcolumn(field, 1, 1), fields)...;
                       pressuregradient=pressuregradient, bulkvelocity=target)

    _, Nxh, Nz = expected
    for iz = 1:Nz, ix = 1:Nxh
        mode = solver.modes[ix, iz]
        isnothing(mode) && continue
        solve!(mode, map(field -> _chebcolumn(field, ix, iz), fields)...)
    end
    for field in (U.components..., P)
        zero_nyquist!(field)
    end
    return gradients
end
