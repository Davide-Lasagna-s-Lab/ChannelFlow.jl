export kinetic_energy, dissipation_rate, power_input
export laminar_kinetic_energy, laminar_dissipation_rate, laminar_power_input


#//////////////////////////////////////////////////////////////////////////////#
#///                        VOLUME INNER PRODUCT                            ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    dot(u::PhysicalField, v::PhysicalField)
    dot(U::SpectralField, V::SpectralField)
    dot(U::VectorField, V::VectorField)

Return the volume-averaged inner product `⟨u ⋅ v⟩`. Physical fields use
periodic averaging and Clenshaw--Curtis quadrature in y; spectral fields
use Fourier orthogonality and the exact unweighted Chebyshev product
integrals, without transforms. Inputs are preserved. Physical quadrature is
exact through degree Ny-1; the spectral method integrates the represented
polynomial products exactly. These routines allocate their work arrays.
"""
function LinearAlgebra.dot(u::PhysicalField, v::PhysicalField)
    Nx, Nz, Ny = size(u)
    # Match the exact unweighted integrals of T_n on [-1, 1]. Divide by
    # the wall-normal length and periodic point count for a volume average.
    C = [cospi(i*j/(Ny-1)) for i = 0:Ny-1, j = 0:Ny-1]
    moments = [iseven(j) ? 2/(1-j^2) : 0.0 for j = 0:Ny-1]
    weights = reshape(transpose(C) \ moments, 1, 1, Ny)
    return sum(weights .* parent(u) .* parent(v))/(2Nx*Nz)
end

function LinearAlgebra.dot(U::SpectralField{T}, V::SpectralField{T}) where {T}
    grid(U) == grid(V) || throw(ArgumentError("fields must share a grid"))
    size(U) == size(V) == spectralsize(grid(U), NotPadded()) ||
        throw(DimensionMismatch("expected resolved spectral fields"))
    Nxh, Nz, Ny = size(U)
    Nx, _, _ = physicalsize(grid(U), NotPadded())

    # M[m+1,n+1] = integral(T_m*T_n, -1, 1)/2. Ordinary Chebyshev
    # coefficients are not orthogonal for the unweighted physical integral.
    moment(n) = iseven(n) ? one(T)/(1-n^2) : zero(T)
    M = [(moment(m+n) + moment(abs(m-n)))/2 for m = 0:Ny-1, n = 0:Ny-1]
    work = zeros(Complex{T}, Ny)
    result = zero(T)
    for iz = 1:Nz, ix = 1:Nxh
        # Exclude the same Nyquist planes as the transforms. Positive k
        # represents both members of a conjugate pair; k=0 appears once.
        ((iseven(Nx) && ix == Nxh) ||
         (iseven(Nz) && iz == (Nz >> 1)+1)) && continue
        u = view(parent(U), ix, iz, :)
        v = view(parent(V), ix, iz, :)
        LinearAlgebra.mul!(work, M, v)
        result += (ix == 1 ? 1 : 2) * real(LinearAlgebra.dot(u, work))
    end
    return result
end

LinearAlgebra.dot(U::VectorField, V::VectorField) =
    sum(LinearAlgebra.dot(U[i], V[i]) for i = 1:3)

"""
    norm(U::VectorField)

Return the root-mean-square field magnitude `sqrt(⟨|U|²⟩)`, rather than
an unweighted norm of stored coefficients. Spectral input requires no transforms.
"""
LinearAlgebra.norm(u::VectorField{<:PhysicalField}) = sqrt(LinearAlgebra.dot(u, u))
LinearAlgebra.norm(U::VectorField{<:SpectralField}) = sqrt(LinearAlgebra.dot(U, U))

#//////////////////////////////////////////////////////////////////////////////#
#///                      KINETIC ENERGY AND DISSIPATION                     ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    kinetic_energy(U::VectorField)

Return volume-averaged kinetic energy `⟨|U|²⟩/2`. Accept physical or spectral
velocity. The supplied field is used as-is: add the base profile beforehand
if total rather than perturbation energy is desired.
"""
kinetic_energy(U::VectorField) = LinearAlgebra.norm(U)^2/2

"""
    dissipation_rate(U::VectorField{<:SpectralField}, nu)

Return `nu*⟨|curl(U)|²⟩`, allocating a spectral vorticity field. This equals
the volume-averaged viscous dissipation for incompressible channel velocity
with periodic x,z and impermeable, uniformly moving no-slip walls. It is an
integral identity, not a pointwise equality with strain-based dissipation.
Include the base profile in `U` to obtain total-flow dissipation.
"""
function dissipation_rate(U::VectorField{<:SpectralField}, nu::Real)
    omega = similar(U)
    curl!(omega, U)
    return nu * LinearAlgebra.norm(omega)^2
end


"""
    power_input(U::VectorField{<:SpectralField}, nu; pressuregradient=(0, 0))

Return power input per unit volume from moving walls and a uniform pressure
gradient `(dPdx, dPdz)`, on the channel interval `[-1, 1]`:
`nu/2 * [u⋅∂y(u)]₋₁⁺¹ - dPdx*⟨u⟩ - dPdz*⟨w⟩`.

Assume impermeable, spatially uniform no-slip wall velocities. Then only the
zero Fourier mode contributes to wall work, so no transforms are needed.
Include the base profile in `U` to obtain total-flow input, as for
[`kinetic_energy`](@ref) and [`dissipation_rate`](@ref).
For constant-flux simulations, supply the actual pressure gradient from the
solver. Additional body-force work and perturbation production by base shear
are not included. The input is preserved.
"""
function power_input(U::VectorField{<:SpectralField}, nu::Real;
                     pressuregradient::NTuple{2, Real}=(0, 0))
    input = 0.0
    for (i, gradient) in zip((1, 3), pressuregradient)
        mean = _meanprofile(U[i])

        # Uniform wall velocities multiply plane-averaged shear. The minus
        # sign at the lower wall is its outward-normal orientation.
        upper = sum(mean)
        lower = sum((-1)^n * mean[n+1] for n = 0:length(mean)-1)
        input += nu/2 * real(conj(upper)*diff(mean, :right) -
                             conj(lower)*diff(mean, :left))
        input -= gradient * real(_bulkmean(mean))
    end
    return input
end

_meanprofile(U::SpectralField) = view(parent(U), 1, 1, :)

#//////////////////////////////////////////////////////////////////////////////#
#///                         LAMINAR FLOW DIAGNOSTICS                        ///#
#//////////////////////////////////////////////////////////////////////////////#

# Embed the streamwise base profile in the zero Fourier mode so the same
# exact spectral integrals are used for laminar and instantaneous fields.
function _laminar_velocity(problem::ChannelFlowProblem)
    U = VectorField(SpectralField(problem.grid))
    U[1][1, 1, :] .= Array(problem.scheme.baseflow)
    return U
end

"""
    laminar_kinetic_energy(problem::ChannelFlowProblem)

Return the volume-averaged energy of the stored base profile, `⟨Ub²⟩/2`.
For standard Couette and Poiseuille flow this is `1/6` and `4/15`, respectively.
"""
laminar_kinetic_energy(problem::ChannelFlowProblem) =
    kinetic_energy(_laminar_velocity(problem))

"""
    laminar_dissipation_rate(problem::ChannelFlowProblem)

Return the base-profile dissipation `nu*⟨(∂y Ub)²⟩`.
For standard Couette and Poiseuille flow this is `nu` and `4nu/3`, respectively.
"""
laminar_dissipation_rate(problem::ChannelFlowProblem) =
    dissipation_rate(_laminar_velocity(problem), problem.scheme.nu)

"""
    laminar_power_input(problem::ChannelFlowProblem; pressuregradient=...)

Return moving-wall and pressure-gradient work for the stored base profile.
Use the problem's prescribed pressure gradient when present. For constant-flux
problems, default to `(nu*⟨Ub''⟩, 0)`, the gradient sustaining a quadratic
laminar profile; this assumes the prescribed flux matches that profile.
Override `pressuregradient` to evaluate a different driving gradient.
Additional body-force work is not included.

For standard, consistently driven Couette and Poiseuille flow the result is
`nu` and `4nu/3`, respectively, equal to laminar dissipation.
"""
function laminar_power_input(problem::ChannelFlowProblem;
                             pressuregradient=haskey(problem.constraint, :pressuregradient) ?
                                 problem.constraint.pressuregradient :
                                 (problem.scheme.nu * real(_bulkmean(Array(problem.scheme.basecurvature))), 0))
    return power_input(_laminar_velocity(problem), problem.scheme.nu;
                       pressuregradient=pressuregradient)
end
