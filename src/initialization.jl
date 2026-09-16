export zero_state, random_state, project!

#//////////////////////////////////////////////////////////////////////////////#
#///                         ZERO STATE ALLOCATION                          ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    zero_state(grid::Grid)

Allocate `State(U, P)` with zero spectral perturbation velocity and pressure.
A laminar base profile belongs to the problem, so it is not added here.
Zero pressure need not be a consistent initial pressure for time stepping.
"""
function zero_state(grid::Grid)
    P = SpectralField(zeros(ComplexF64, spectralsize(grid, NotPadded())), grid)
    return State(VectorField(P), P)
end

#//////////////////////////////////////////////////////////////////////////////#
#///                     RANDOM VELOCITY INITIALIZATION                     ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    random_state(grid::Grid, amplitude)

Create a random real-space perturbation velocity, transform it to spectral
storage, project it onto the divergence-free no-slip space, and return the
state `State(U, P)`. `amplitude` scales Gaussian samples before projection;
it is not a prescribed final RMS or kinetic energy. `P` is the projection
multiplier, not a dynamically consistent initial pressure.
Use Julia's default RNG; call `Random.seed!`
beforehand for reproducible initialization.
"""
function random_state(     grid::Grid,
                      amplitude::Real)
    amplitude >= 0 || throw(ArgumentError("amplitude must be non-negative"))
    physical = PhysicalField(zeros(Float64, physicalsize(grid, Padded())), grid)
    prototype = SpectralField(zeros(ComplexF64, spectralsize(grid, NotPadded())), grid)
    U = VectorField(prototype)
    fft = ForwardFFT!(physical)
    for component in U.components
        parent(physical) .= amplitude .* Random.randn(size(physical))
        fft(component, physical)
    end
    return project!(U)
end

#//////////////////////////////////////////////////////////////////////////////#
#///                   DIVERGENCE-FREE NO-SLIP PROJECTION                   ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    project!(U::VectorField; bulkvelocity=nothing)

Project the spectral perturbation velocity `U` onto the divergence-free,
no-slip space, overwriting its three components and returning the coupled
state `State(U, phi)`.
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

By default `G = 0`. Optionally impose perturbation bulk velocities with
`bulkvelocity=(Ubulk, Wbulk)` and determine the auxiliary uniform gradient
`G`. No base profile is supplied to this projection. A field already satisfying
the constraints is unchanged up to solve roundoff. Excluded Nyquist planes
are set to zero. The projection acts on the perturbation field and does not
add or remove a base profile.

Require resolved `ComplexF64` fields on one grid, odd `Ny ≥ 3`, independent
component storage and Fourier conjugate symmetry for a real velocity.
Allocate factors and workspaces on each call: this function is intended for
initialisation. The auxiliary multiplier `phi` is retained in `State`, but
is not the physical or modified pressure required to initialise CNRK2.
In particular, an already admissible velocity has zero projection multiplier,
irrespective of its dynamical pressure.
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
    return State(U, phi)
end
