export project!

"""
    project!(U::VectorField; bulkvelocity=nothing)

Project the spectral perturbation velocity `U` onto the divergence-free,
no-slip space, overwriting its three components and returning `U`.
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
the same tau discretisation as [`FourierStokesSolver`](@ref).

By default `G = 0`. Optionally impose TOTAL streamwise/spanwise bulk velocities
with `bulkvelocity=(Ubulk, Wbulk)`; the solver subtracts the base-flow mean
and determines the auxiliary uniform gradient `G`. A field already satisfying
the constraints is unchanged up to solve roundoff. Excluded Nyquist planes
are set to zero. The projection acts on the perturbation field and does not
add or remove a base profile.

Require resolved `ComplexF64` fields on one grid, odd `Ny ≥ 3`, independent
component storage and Fourier conjugate symmetry for a real velocity.
Allocate factors and workspaces on each call: this function is intended for
initialisation. The auxiliary multiplier `phi` is discarded; it is not the
physical or modified pressure needed to initialise CNRK2.
"""
function project!(           U::VectorField{F};
                  bulkvelocity::Union{Nothing, NTuple{2, Real}}=nothing) where {F<:SpectralField{Float64}}
    g = grid(U[1])
    for field in U.components
        grid(field) === g || throw(ArgumentError("velocity components must share a grid"))
        size(field) == spectralsize(g, NotPadded()) ||
            throw(DimensionMismatch("velocity must have the resolved spectral size"))
    end
    isnothing(bulkvelocity) || all(isfinite, bulkvelocity) ||
        throw(ArgumentError("bulk velocities must be finite"))

    solver = FourierStokesSolver(g, 1.0, 0.0)
    R, phi = similar(U), similar(U[1])
    # Form the complete source before solve! overwrites any velocity component.
    for i = 1:3
        laplacian!(R[i], U[i])
        R[i] .*= -1
    end
    solve!(solver, U, phi, R; bulkvelocity=bulkvelocity)
    return U
end
