@testset "Analytic Stokes projection" begin
    # Construct a velocity in the admissible subspace plus a known gradient.
    # Recovering the known velocity is stronger than checking divergence
    # alone, since returning zero would otherwise satisfy the constraints.
    for (Nx, Nz) in ((40, 42), (41, 43))
        g = Grid(13, Nx, Nz, 5.3, 7.1)
        a, b = 2π/5.3, 2π/7.1
        q(y) = (1-y^2)^2
        dq(y) = -4y+4y^3
        d2q(y) = -4+12y^2
        # exp(cos) excites every Fourier harmonic in both periodic directions.
        # These resolutions make the omitted tail negligible at the tolerances
        # below; this field is smooth but is not a finite Fourier polynomial.
        # Normalize the peak to one without changing harmonic support.
        h(x,z) = exp(cos(a*x)+cos(b*z)-2)
        hx(x,z) = -a*sin(a*x)*h(x,z)
        hz(x,z) = -b*sin(b*z)*h(x,z)
        hxx(x,z) = a^2*(sin(a*x)^2-cos(a*x))*h(x,z)
        hzz(x,z) = b^2*(sin(b*z)^2-cos(b*z))*h(x,z)
        # Uexact = (d_y psi, -d_x psi, 0), psi=q(y)*h(x,z).
        # Commuting mixed derivatives makes this streamfunction velocity
        # divergence-free. Both q and dq vanish at the walls, ensuring no slip
        # for the exact velocity and the added gradient.
        exact = ((x,y,z) -> dq(y)*h(x,z),
                 (x,y,z) -> -q(y)*hx(x,z), fzero)
        # Add grad(chi), chi=q(y)*h(x,z). The auxiliary multiplier
        # for -Delta Unew + grad(phi) = -Delta Uinput is phi=-Delta chi.
        correction = ((x,y,z) -> q(y)*hx(x,z),
                      (x,y,z) -> dq(y)*h(x,z),
                      (x,y,z) -> q(y)*hz(x,z))
        U = VectorField(ntuple(i -> spectral(g, (x,y,z) ->
                                exact[i](x,y,z)+correction[i](x,y,z)), 3))
        # Projection updates the supplied velocity in place and wraps it with
        # the auxiliary pressure in a State. Compare every reconstructed
        # component with the independently specified exact field.
        state = project!(U)
        @test state isa State
        @test velocity(state) === U
        for i = 1:3
            @test physical_values(U[i]) ≈ parent(sampled(g, exact[i])) atol=2e-10
        end
        # For this Stokes projection the multiplier is -Delta(chi), not chi.
        # Its analytic sign and amplitude check the pressure output
        # independently. It is a projection multiplier, not a test of the
        # physical initial pressure required by a dynamical problem.
        phi(x,y,z) = -d2q(y)*h(x,z)-q(y)*(hxx(x,z)+hzz(x,z))
        @test physical_values(pressure(state)) ≈ parent(sampled(g, phi)) atol=2e-9
        check_constraints(U)

        # An already admissible field must be unchanged by a second
        # projection. Its new auxiliary multiplier is zero; the previous
        # multiplier is not a physical pressure that must persist across
        # projection calls.
        saved = map(copy, U.components)
        second = project!(U)
        @test all(isapprox(parent(U[i]), parent(saved[i]); atol=2e-11) for i=1:3)
        @test norm(parent(pressure(second)), Inf) < 2e-9
    end

    g = Grid(9, 5, 5, 2π, 2π)
    # Starting from zero with specified bulk velocities tests the constrained
    # mean mode. These are perturbation means because projection receives no
    # base flow. Independent quadrature and wall/divergence checks verify the
    # imposed values and admissibility together.
    state = project!(velocity(zero_state(g)); bulkvelocity=(0.2, -0.1))
    U = velocity(state)
    @test real(bulkmean(ChebCoeffs(view(parent(U[1]), :, 1, 1)))) ≈ 0.2 atol=2e-12
    @test real(bulkmean(ChebCoeffs(view(parent(U[3]), :, 1, 1)))) ≈ -0.1 atol=2e-12
    check_constraints(U)
    # Reject nonfinite constraints, components attached to different domains,
    # and padded spectral storage. Matching array dimensions alone cannot
    # establish a common grid.
    @test_throws ArgumentError project!(U; bulkvelocity=(NaN, 0))
    other = Grid(9, 5, 5, 3π, 2π)
    bad = VectorField((U[1], spectral(other, fzero), U[3]))
    @test_throws ArgumentError project!(bad)
    padded = SpectralField(zeros(ComplexF64, spectralsize(g, Padded())), g)
    @test_throws DimensionMismatch project!(VectorField(padded))
end

@testset "Reproducible random initialization" begin
    g = Grid(9, 5, 5, 2π, 2π)
    # Reseeding the default generator with the same seed must reproduce both velocity and
    # auxiliary pressure exactly. Nonzero output rules out a trivial zero
    # initializer; the shared checks verify projection onto the constrained
    # subspace.
    Random.seed!(17)
    a = random_state(g, 0.1)
    Random.seed!(17)
    b = random_state(g, 0.1)
    @test all(parent(velocity(a)[i]) == parent(velocity(b)[i]) for i=1:3)
    @test parent(pressure(a)) == parent(pressure(b))
    @test any(norm(parent(component)) > 0 for component in velocity(a).components)
    check_constraints(velocity(a))
    # The spectrum represents real data, including the kx=0 conjugate pairs.
    # A real-space round trip must preserve the full retained spectrum,
    # including conjugate pairs on the kx=0 plane. This catches complex
    # coefficients that cannot represent a real velocity field.
    for U in velocity(a).components
        @test parent(spectral(g, physical_values(U))) ≈ parent(U) atol=2e-11
    end
    # A negative requested amplitude is outside the initialization interface
    # contract.
    @test_throws ArgumentError random_state(g, -1)
end
