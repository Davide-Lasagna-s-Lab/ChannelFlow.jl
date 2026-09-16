export Postprocessor, kinetic_energy, dissipation_rate, power_input,
       flow_diagnostics

#//////////////////////////////////////////////////////////////////////////////#
#///                       POSTPROCESSING WORKSPACE                         ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    Postprocessor(grid; fftwflags=FFTW.MEASURE,
                  fftwtimelimit=FFTW.NO_TIMELIMIT)

Allocate reusable workspaces for volume-integrated channel-flow diagnostics.
The physical buffers use the 3/2-padded periodic grid, while wall-normal
integration uses Clenshaw--Curtis weights on the Chebyshev--Lobatto points.
"""
struct Postprocessor{PV, PG, SV, SG, I, W}
       velocity::PV
        gradient::PG
    spectral_velocity::SV
    spectral_gradient::SG
            ifft::I
         weights::W
end

function Postprocessor(grid::Grid;
                       fftwflags::Integer=FFTW.MEASURE,
                       fftwtimelimit::Real=FFTW.NO_TIMELIMIT)
    Ny = length(grid.y)
    spectral = SpectralField(zeros(ComplexF64,
                                   spectralsize(grid, NotPadded())), grid)
    physical = PhysicalField(zeros(Float64,
                                   physicalsize(grid, Padded())), grid)
    spectral_velocity = VectorField(spectral)
    spectral_gradient = GradientField(spectral)
    velocity = VectorField(physical)
    gradient = GradientField(physical)
    ifft = InverseFFT!(spectral; flags=fftwflags,
                       timelimit=fftwtimelimit)

    # Interpolatory quadrature: C*w reproduces the exact integral of every
    # represented Chebyshev polynomial on [-1,1].
    C = [cospi(i*j/(Ny-1)) for i = 0:Ny-1, j = 0:Ny-1]
    moments = [iseven(j) ? 2/(1-j^2) : 0.0 for j = 0:Ny-1]
    weights = transpose(C) \ moments
    return Postprocessor(velocity, gradient, spectral_velocity,
                         spectral_gradient, ifft, weights)
end

#//////////////////////////////////////////////////////////////////////////////#
#///                    TOTAL-VELOCITY CACHE ASSEMBLY                      ///#
#//////////////////////////////////////////////////////////////////////////////#

function _total_velocity!(post::Postprocessor,
                          U::VectorField{S},
                          baseflow) where {S<:SpectralField}
    post.spectral_velocity .= U
    if !isnothing(baseflow)
        length(baseflow) == size(U[1], 1) ||
            throw(DimensionMismatch("baseflow must have one coefficient per wall-normal point"))
        @views post.spectral_velocity[1][:, 1, 1] .+= baseflow
    end
    post.ifft(post.velocity, post.spectral_velocity)
    return post.velocity
end

function _total_gradient!(post::Postprocessor)
    grad!(post.spectral_gradient, post.spectral_velocity)
    post.ifft(post.gradient, post.spectral_gradient)
    return post.gradient
end

function _volume_average(post::Postprocessor, values)
    _, Nx, Nz = size(values)
    integral = zero(eltype(values))
    @inbounds for iz = 1:Nz, ix = 1:Nx, iy = 1:length(post.weights)
        integral += post.weights[iy]*values[iy, ix, iz]
    end
    return integral/(2Nx*Nz)
end

function _volume_abs2_average(post::Postprocessor, values)
    _, Nx, Nz = size(values)
    integral = 0.0
    @inbounds for iz = 1:Nz, ix = 1:Nx, iy = 1:length(post.weights)
        integral += post.weights[iy]*abs2(values[iy, ix, iz])
    end
    return integral/(2Nx*Nz)
end

_kinetic_energy(post, velocity) =
    sum(_volume_abs2_average(post, parent(velocity[i])) for i = 1:3)/2

_dissipation_rate(post, gradient, nu) =
    nu*sum(_volume_abs2_average(post, parent(gradient[i, j]))
           for i = 1:3, j = 1:3)

function _power_input(post, velocity, gradient, nu, pressuregradient)
    _, Nx, Nz = size(velocity[1])
    wall = 0.0
    @inbounds for i = 1:3, iz = 1:Nz, ix = 1:Nx
        wall += velocity[i][1, ix, iz]*gradient[i, 2][1, ix, iz] -
                velocity[i][end, ix, iz]*gradient[i, 2][end, ix, iz]
    end
    wall *= nu/(2Nx*Nz)
    pressure = -pressuregradient[1]*_volume_average(post, parent(velocity[1])) -
               pressuregradient[2]*_volume_average(post, parent(velocity[3]))
    return wall + pressure
end

#//////////////////////////////////////////////////////////////////////////////#
#///                         INTEGRAL DIAGNOSTICS                           ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    kinetic_energy(post, U; baseflow=nothing)

Return the volume-averaged kinetic energy `⟨|u|²⟩/2`. `U` stores the
perturbation velocity; pass the base-flow Chebyshev coefficients to measure
the total velocity instead.
"""
function kinetic_energy(post::Postprocessor,
                        U::VectorField{<:SpectralField};
                        baseflow=nothing)
    u = _total_velocity!(post, U, baseflow)
    return _kinetic_energy(post, u)
end

"""
    dissipation_rate(post, U, nu; baseflow=nothing)

Return `nu*⟨sum(i,j) |∂u_i/∂x_j|²⟩` for the perturbation or total
velocity. This is the volume-integrated viscous dissipation convention used
by the channel-flow energy balance.
"""
function dissipation_rate(post::Postprocessor,
                          U::VectorField{<:SpectralField},
                          nu::Real;
                          baseflow=nothing)
    _total_velocity!(post, U, baseflow)
    gradient = _total_gradient!(post)
    return _dissipation_rate(post, gradient, nu)
end

"""
    power_input(post, U, nu; baseflow=nothing,
                pressuregradient=(0, 0))

Return total power input per unit volume. It includes work by moving walls and
work by the uniform streamwise/spanwise pressure gradient. Wall velocities
and wall shear are evaluated from the total velocity.
"""
function power_input(post::Postprocessor,
                     U::VectorField{<:SpectralField},
                     nu::Real;
                     baseflow=nothing,
                     pressuregradient::NTuple{2, Real}=(0, 0))
    velocity = _total_velocity!(post, U, baseflow)
    gradient = _total_gradient!(post)
    return _power_input(post, velocity, gradient, nu, pressuregradient)
end

"""
    flow_diagnostics(post, U, nu; baseflow=nothing,
                     pressuregradient=(0, 0))

Return kinetic energy, viscous dissipation rate and power input in one pass
through the total velocity and its gradient.
"""
function flow_diagnostics(post::Postprocessor,
                          U::VectorField{<:SpectralField},
                          nu::Real;
                          baseflow=nothing,
                          pressuregradient::NTuple{2, Real}=(0, 0))
    velocity = _total_velocity!(post, U, baseflow)
    gradient = _total_gradient!(post)
    return (; kinetic_energy=_kinetic_energy(post, velocity),
            dissipation_rate=_dissipation_rate(post, gradient, nu),
            power_input=_power_input(post, velocity, gradient, nu,
                                     pressuregradient))
end

# State overloads retain the same explicit base-flow convention.
kinetic_energy(post::Postprocessor, state::State; kwargs...) =
    kinetic_energy(post, velocity(state); kwargs...)
dissipation_rate(post::Postprocessor, state::State, nu::Real; kwargs...) =
    dissipation_rate(post, velocity(state), nu; kwargs...)
power_input(post::Postprocessor, state::State, nu::Real; kwargs...) =
    power_input(post, velocity(state), nu; kwargs...)
flow_diagnostics(post::Postprocessor, state::State, nu::Real; kwargs...) =
    flow_diagnostics(post, velocity(state), nu; kwargs...)
