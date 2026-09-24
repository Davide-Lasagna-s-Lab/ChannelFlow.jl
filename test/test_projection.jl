@testset "Analytic Stokes projection" begin
    # Construct a velocity in the admissible subspace plus a known gradient.
    # Recovering the known velocity is stronger than checking divergence
    # alone, since returning zero would otherwise satisfy the constraints.
    for (Nx, Nz) in ((40, 42), (41, 43))
        g = Grid(Nx, 13, Nz, 5.3, 7.1)
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
        problem = ChannelFlowProblem(g, y -> 0.0, 0.01, 0.01;
                                     form=CF.ConvectiveForm(),
                                     fftwflags=FFTW.ESTIMATE)
        # Projection updates and returns the supplied velocity in place.
        # Reconstruct pressure separately. Compare every reconstructed
        # component with the independently specified exact field.
        @test project!(U, problem) === U
        P = pressure(U, problem)
        for i = 1:3
            @test physical_values(U[i]) ≈ parent(sampled(g, exact[i])) atol=2e-10
        end
        @test all(isfinite, parent(P))
        check_constraints(U)

        # An already admissible field must be unchanged by a second projection,
        # and its reconstructed instantaneous pressure must also be unchanged.
        saved = map(copy, U.components)
        saved_pressure = copy(P)
        project!(U, problem)
        second_pressure = pressure(U, problem)
        @test all(isapprox(parent(U[i]), parent(saved[i]); atol=2e-11) for i=1:3)
        @test parent(second_pressure) ≈ parent(saved_pressure) atol=2e-9
    end

    g = Grid(5, 9, 5, 2π, 2π)
    # Starting from zero with specified bulk velocities tests the constrained
    # mean mode. These are perturbation means because projection receives no
    # base flow. Independent quadrature and wall/divergence checks verify the
    # imposed values and admissibility together.
    problem = ChannelFlowProblem(g, y -> 0.0, 0.01, 0.01;
                                 bulkvelocity=(0.2, -0.1),
                                 fftwflags=FFTW.ESTIMATE)
    U = project!(velocity(zero_state(g)), problem)
    @test real(bulkmean(view(parent(U[1]), 1, 1, :))) ≈ 0.2 atol=2e-12
    @test real(bulkmean(view(parent(U[3]), 1, 1, :))) ≈ -0.1 atol=2e-12
    check_constraints(U)
    # Nonfinite bulk targets must be rejected at problem construction.
    # Projection assumes resolved fields on the problem's grid; it no longer
    # provides the removed component-grid validation interface.
    @test_throws ArgumentError ChannelFlowProblem(g, y -> 0.0, 0.01, 0.01;
                                                  bulkvelocity=(NaN, 0.0),
                                                  fftwflags=FFTW.ESTIMATE)
end

@testset "Pressure associated with the nonlinear form" begin
    g = Grid(5, 9, 5, 2π, 2π)
    problem = CouetteFlow(g, 0.01, 0.01; fftwflags=FFTW.ESTIMATE)
    U = project!(velocity(zero_state(g)), problem)
    P = pressure(U, problem)

    # For laminar Couette flow the rotational acceleration is grad(y^2/2),
    # hence the stored modified pressure is y^2/2 up to its arbitrary gauge.
    expected = parent(sampled(g, (x, y, z) -> y^2/2 - 1/6))
    @test physical_values(P) ≈ expected atol=2e-10
end

@testset "Reproducible random initialization" begin
    g = Grid(5, 9, 5, 2π, 2π)
    problem = ChannelFlowProblem(g, y -> 0.0, 0.01, 0.01;
                                 fftwflags=FFTW.ESTIMATE)
    # Reseeding the default generator with the same seed must reproduce both
    # velocity and pressure exactly. Nonzero output rules out a trivial zero
    # initializer; the shared checks verify projection onto the constrained
    # subspace.
    Random.seed!(17)
    a = random_state(problem, 0.1)
    Random.seed!(17)
    b = random_state(problem, 0.1)
    @test all(parent(velocity(a)[i]) == parent(velocity(b)[i]) for i=1:3)
    @test parent(stagepressure(a)) == parent(stagepressure(b))
    @test any(norm(parent(component)) > 0 for component in velocity(a).components)
    check_constraints(velocity(a))
    # The spectrum represents real data, including the k=0 conjugate pairs.
    # A real-space round trip must preserve the full retained spectrum,
    # including conjugate pairs on the k=0 plane. This catches complex
    # coefficients that cannot represent a real velocity field.
    for U in velocity(a).components
        @test parent(spectral(g, physical_values(U))) ≈ parent(U) atol=2e-11
    end
    # A negative requested amplitude is outside the initialization interface
    # contract.
    @test_throws ArgumentError random_state(problem, -1)
end
