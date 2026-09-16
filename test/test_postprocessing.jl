@testset "Integral postprocessing diagnostics" begin
    g = Grid(17, 8, 8, 2π, 2π)
    nu = 0.01
    state = zero_state(g)
    post = Postprocessor(g; fftwflags=FFTW.ESTIMATE)

    # Laminar Couette flow has <y²>/2 = 1/6. Its uniform shear gives equal
    # dissipation and moving-wall input, both equal to nu.
    couette = chebyshev_coefficients(g, identity)
    c = flow_diagnostics(post, state, nu; baseflow=couette)
    @test c.kinetic_energy ≈ 1/6 atol=2e-14
    @test c.dissipation_rate ≈ nu atol=2e-14
    @test c.power_input ≈ nu atol=2e-14
    @test kinetic_energy(post, state; baseflow=couette) ≈ c.kinetic_energy
    @test dissipation_rate(post, state, nu; baseflow=couette) ≈ c.dissipation_rate
    @test power_input(post, state, nu; baseflow=couette) ≈ c.power_input

    # For unit-centreline Poiseuille flow, <U²>/2=4/15 and
    # nu<|dU/dy|²>=4nu/3. Stationary walls do no work; the sustaining
    # pressure gradient -2nu supplies exactly the viscous dissipation.
    poiseuille = chebyshev_coefficients(g, y -> 1-y^2)
    p = flow_diagnostics(post, velocity(state), nu;
                         baseflow=poiseuille,
                         pressuregradient=(-2nu, 0.0))
    @test p.kinetic_energy ≈ 4/15 atol=2e-14
    @test p.dissipation_rate ≈ 4nu/3 atol=2e-14
    @test p.power_input ≈ 4nu/3 atol=2e-14

    # Omitting the base flow reports perturbation quantities; the zero state
    # must therefore have zero energy, dissipation and input.
    zero = flow_diagnostics(post, state, nu)
    @test zero == (kinetic_energy=0.0,
                   dissipation_rate=0.0,
                   power_input=0.0)
end
