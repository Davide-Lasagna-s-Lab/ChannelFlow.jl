@testset "Analytic scalar Helmholtz boundary-value problem" begin
    # Manufacture a cubic solution of nu*u double-prime - lambda*u = forcing
    # using its analytic second derivative. Multiple sizes, zero/nonzero
    # shift, and signed amplitudes exercise factor updates, boundary scaling
    # and both coefficient parities without approximation error.
    for Ny in (8, 9, 17)
        solver = ChebyshevHelmoltzSolvers.HelmoltzSolver(Ny-1, Float64)
        exact(y) = 0.7+0.3y-0.2y^2+0.1y^3
        # Unequal nonzero endpoint values detect swapped wall conventions.
        for (nu, lambda) in ((0.1, 0.0), (0.03, 2.5)), amplitude in (1.0, -0.4)
            ChebyshevHelmoltzSolvers.update!(solver, nu, lambda)
            forcing(y) = amplitude*(nu*(-0.4+0.6y)-lambda*exact(y))
            rhs = real.(parent(coefficients(forcing, Ny)))
            expected = real.(parent(coefficients(y -> amplitude*exact(y), Ny)))
            source = copy(rhs)
            saved = copy(source)
            result = ChebyshevHelmoltzSolvers.solve!(solver, rhs, source,
                         amplitude*exact(1), amplitude*exact(-1))
            @test source == saved
            # The scalar solve preserves source and returns rhs. Compare every
            # coefficient to the analytic polynomial, then evaluate both
            # endpoints independently to verify boundary ordering and values.
            @test result === rhs
            @test rhs ≈ expected atol=2e-12
            @test wallvalue(rhs, :right) ≈ amplitude*exact(1) atol=2e-12
            @test wallvalue(rhs, :left) ≈ amplitude*exact(-1) atol=2e-12
        end
    end
end

# Check the discrete constrained Stokes equations in coefficient space. Unlike
# the analytic solution comparisons, these residual checks use the library
# derivative and therefore test mutual consistency of the assembled operators
# and solver.
function check_stokes(solver, u, v, w, p, Rx, Ry, Rz, nu)
    P = length(p) - 1
    dp = derivative(p)
    divergence = im*solver.k .* u .+ parent(derivative(v)) .+
                 im*solver.l .* w
    # Incompressibility applies to all coefficients, including the highest
    # degrees; it is not relaxed by the momentum tau treatment.
    @test norm(divergence, Inf) < 2e-10

    for (field, source, gradp) in ((u, Rx, im*solver.k .* p),
                                   (v, Ry, dp),
                                   (w, Rz, im*solver.l .* p))
        residual = solver.lambda .* field .-
                   nu .* parent(derivative(derivative(field))) .+
                   gradp .- source
        # With maximum degree P, the momentum equations retain degrees 0
        # through P-2 (array entries 1:P-1). The two remaining equations
        # impose wall conditions, so demanding zero momentum residual in those
        # last two entries would test a different discretization.
        # The highest two residual coefficients are the momentum tau terms.
        @test norm(residual[1:P-1], Inf) < 2e-10
        @test abs(wallvalue(field, :left)) < 2e-11
        @test abs(wallvalue(field, :right)) < 2e-11
    end
    # No slip together with incompressibility requires the normal derivative
    # of v to vanish at both walls. Checking it explicitly detects incomplete
    # influence-matrix boundary correction.
    @test abs(diff(v, :left)) < 2e-11
    @test abs(diff(v, :right)) < 2e-11
end

@testset "Complex primitive-variable mode" begin
    # This solver requires the supported odd number of coefficients and a
    # nonzero Fourier mode. The zero mode has a different pressure gauge and
    # bulk constraint and must use MeanModeSolver.
    @test_throws ArgumentError InfluenceModeSolver(10, 1, 1, 0.1, 2)
    @test_throws ArgumentError InfluenceModeSolver(9, 0, 0, 0.1, 2)

    for Ny in (9, 17, 33), (k, l) in ((1.25, 0.0), (0.0, -2.0),
                                        (1.25, -1.75), (1.25, 1.75))
        @testset "Ny=$Ny, k=($k,$l)" begin
            nu, lambda = 0.03, 2.5
            solver = InfluenceModeSolver(Ny, k, l, nu, lambda)
            kappa2 = k^2 + l^2
            shift = lambda + nu*kappa2
            # Choose v with a double zero at both walls. Setting
            # (u,w)=i*(k,l)*dv/kappa2+(l,-k)*q cancels dv in the
            # divergence; q supplies an independent transverse component. Both
            # tangential velocities vanish at the walls. Complex amplitudes
            # exercise real and imaginary solves.
            vfun(y) = (0.7+0.2im)*(1-y^2)^2*(1+0.2y)
            dvfun(y) = (0.7+0.2im)*(-4y+4y^3+0.2*(1-6y^2+5y^4))
            d2vfun(y) = (0.7+0.2im)*(-4+12y^2+0.2*(-12y+20y^3))
            d3vfun(y) = (0.7+0.2im)*(24y+0.2*(-12+60y^2))
            qfun(y) = (0.15-0.1im)*(1-y^2)*(1+0.3y)
            d2qfun(y) = (0.15-0.1im)*(-2-1.8y)
            ufun(y) = im*k/kappa2*dvfun(y) + l*qfun(y)
            wfun(y) = im*l/kappa2*dvfun(y) - k*qfun(y)
            pfun(y) = (0.4-0.3im)*(1+y+0.2y^3)
            dpfun(y) = (0.4-0.3im)*(1+0.6y^2)

            # Build the forcing from explicitly written derivatives of the
            # manufactured solution: (lambda+nu*kappa2)*velocity - nu*velocity
            # double-prime + pressure gradient. Thus a shared derivative
            # implementation cannot make an incorrect solution pass this
            # comparison.
            Rx = coefficients(y -> shift*ufun(y) -
                                   nu*(im*k/kappa2*d3vfun(y)+l*d2qfun(y)) +
                                   im*k*pfun(y), Ny)
            Ry = coefficients(y -> shift*vfun(y)-nu*d2vfun(y)+dpfun(y), Ny)
            Rz = coefficients(y -> shift*wfun(y) -
                                   nu*(im*l/kappa2*d3vfun(y)-k*d2qfun(y)) +
                                   im*l*pfun(y), Ny)
            source = map(x -> copy(x), (Rx, Ry, Rz))
            responses = map(x -> copy(x),
                            (solver.pressure_plus, solver.pressure_minus,
                             solver.velocity_plus, solver.velocity_minus,
                             solver.pressure_zero, solver.velocity_zero))
            u, v, w, p = ntuple(_ -> zeros(ComplexF64, Ny), 4)

            # Repeated solves must overwrite the outputs without changing
            # either the source or the precomputed influence/tau responses.
            for _ = 1:2
                @test solve!(solver, u, v, w, p, Rx, Ry, Rz) == (u, v, w, p)
                for (field, exact) in zip((u, v, w, p), (ufun, vfun, wfun, pfun))
                    @test norm(field-parent(coefficients(exact, Ny)), Inf) < 2e-10
                end
                check_stokes(solver, u, v, w, p, Rx, Ry, Rz, nu)
                @test map(identity, (Rx, Ry, Rz)) == source
                @test map(identity, (solver.pressure_plus, solver.pressure_minus,
                                   solver.velocity_plus, solver.velocity_minus,
                                   solver.pressure_zero, solver.velocity_zero)) == responses
            end
        end
    end

    # Nonzero high-degree forcing exercises the tau correction, rather than
    # only exact low-degree polynomials. Views match Fourier-column storage.
    Ny = 17
    solver = InfluenceModeSolver(Ny, 1.25, -1.75, 0.03, 2.5)
    data = zeros(ComplexF64, Ny, 7)
    u, v, w, p, Rx, Ry, Rz = ntuple(i -> view(data, :, i), 7)
    for (j, rhs) in enumerate((Rx, Ry, Rz)), n = 0:Ny-1
        rhs[n+1] = complex(sin((j+1)*(n+1)), cos((j+2)*(n+1)))/(n+1)^2
    end
    source = copy(data[:, 5:7])
    solve!(solver, u, v, w, p, Rx, Ry, Rz)
    check_stokes(solver, u, v, w, p, Rx, Ry, Rz, 0.03)
    @test data[:, 5:7] == source

    # An incompatible coefficient count must fail before a modal solve can
    # index or overwrite the wrong storage.
    wrong = ntuple(_ -> zeros(ComplexF64, 9), 7)
    @test_throws DimensionMismatch solve!(solver, wrong...)
end

@testset "Mean Fourier mode" begin
    # The mean solve needs enough coefficients for two wall conditions,
    # positive finite viscosity and a finite nonnegative shift. Exercise each
    # invalid constructor input separately.
    @test_throws ArgumentError MeanModeSolver(2, 0.03, 2.5)
    @test_throws ArgumentError MeanModeSolver(9, 0, 2.5)
    @test_throws ArgumentError MeanModeSolver(9, -0.03, 2.5)
    @test_throws ArgumentError MeanModeSolver(9, 0, 0)
    @test_throws ArgumentError MeanModeSolver(9, 0.03, -1)
    @test_throws ArgumentError MeanModeSolver(9, Inf, 2.5)
    @test_throws ArgumentError MeanModeSolver(9, 0.03, NaN)

    for Ny in (9, 16, 33), lambda in (0.0, 2.5)
        @testset "Ny=$Ny, lambda=$lambda" begin
            nu = 0.03
            solver = MeanModeSolver(Ny, nu, lambda)
            # These wall-vanishing tangential polynomials have analytic means
            # 0.48 and -0.2. The pressure has zero bulk mean because the odd
            # terms integrate to zero and mean(y^2)=1/3. Normal velocity is
            # zero for the incompressible mean mode.
            gradients, means = (-0.7, 0.3), (0.48, -0.2)
            ufun(y) = (1-y^2)*(0.7+0.2y+0.1y^2)
            wfun(y) = (1-y^2)*(-0.3+0.15y)
            pfun(y) = y+0.4*(y^2-1/3)+0.2y^3
            Rx = coefficients(y -> lambda*ufun(y)+nu*(1.2+1.2y+1.2y^2)+gradients[1], Ny)
            Ry = coefficients(y -> 1+0.8y+0.6y^2, Ny)
            Rz = coefficients(y -> lambda*wfun(y)-nu*(0.6-0.9y)+gradients[2], Ny)
            source = map(x -> copy(x), (Rx, Ry, Rz))
            response = copy(solver.response)
            u, v, w, p = ntuple(_ -> zeros(ComplexF64, Ny), 4)

            # Prescribing either the known pressure gradients or the known
            # means must recover the same fields and gradients. Repetition
            # also checks that the precomputed homogeneous response and right-
            # hand sides remain reusable.
            for _ = 1:2, fixedbulk in (false, true)
                actual = fixedbulk ? solve!(solver, u, v, w, p, Rx, Ry, Rz; bulkvelocity=means) :
                                     solve!(solver, u, v, w, p, Rx, Ry, Rz; pressuregradient=gradients)
                @test all(abs.(actual .- gradients) .< 2e-11)
                for (field, exact) in zip((u, w, p), (ufun, wfun, pfun))
                    @test norm(field-parent(coefficients(exact, Ny)), Inf) < 2e-10
                end
                # Verify the special zero-mode constraints: v=0, zero-mean
                # pressure gauge, the two bulk velocities and dp/dy=Ry.
                # Quadrature provides an independent check of the mean
                # normalization.
                @test all(iszero, v)
                @test abs(bulkmean(p)) < 2e-12
                @test bulkmean(u) ≈ means[1] atol=2e-11
                @test bulkmean(w) ≈ means[2] atol=2e-11
                @test norm(parent(derivative(p))-Ry, Inf) < 2e-10
                # The uniform pressure gradient contributes only to Chebyshev
                # degree zero. Check retained momentum equations and both
                # walls separately; the last two momentum coefficients are tau
                # residuals, not equations required to vanish.
                for (field, rhs, gradient) in ((u, Rx, actual[1]), (w, Rz, actual[2]))
                    residual = lambda .* field .-
                               nu .* parent(derivative(derivative(field))) .- rhs
                    residual[1] += gradient
                    @test norm(residual[1:Ny-2], Inf) < 2e-10
                    @test abs(wallvalue(field, :right)) < 2e-11
                    @test abs(wallvalue(field, :left)) < 2e-11
                end
                @test map(identity, (Rx, Ry, Rz)) == source
                @test solver.response == response
            end
        end
    end

    # Fourier-column views and high-degree, complex forcing exercise both
    # real-factor passes and the pressure truncation at the last coefficient.
    Ny, nu, lambda = 17, 0.03, 2.5
    solver = MeanModeSolver(Ny, nu, lambda)
    data = zeros(ComplexF64, Ny, 7)
    u, v, w, p, Rx, Ry, Rz = ntuple(i -> view(data, :, i), 7)
    # Zero forcing must produce zero fields and zero returned gradients even
    # with view-backed coefficient arrays. Then nonzero complex forcing
    # exercises the same storage and both real-factor solve passes.
    @test solve!(solver, u, v, w, p, Rx, Ry, Rz) == (0.0, 0.0)
    @test all(iszero, data)
    for (i, rhs) in enumerate((Rx, Ry, Rz)), n = 0:Ny-1
        rhs[n+1] = complex(sin(i*(n+1)), cos((i+1)*(n+1)))/(n+1)^2
    end
    source = copy(data[:, 5:7])
    gradients = solve!(solver, u, v, w, p, Rx, Ry, Rz; bulkvelocity=(0.2, -0.1))
    # The bulk constraint sets the real mean velocities; pressure fixes the
    # full complex zero-mean gauge. This complex forcing is a solver algebra
    # test, not a physically real zero Fourier mode.
    @test real(bulkmean(u)) ≈ 0.2 atol=2e-12
    @test real(bulkmean(w)) ≈ -0.1 atol=2e-12
    @test abs(bulkmean(p)) < 2e-12
    @test all(iszero, v)
    # A degree Ny-1 pressure has a derivative of degree at most Ny-2. The last
    # forcing coefficient therefore cannot be integrated within this space:
    # the final residual must equal minus that coefficient, while every lower
    # residual vanishes.
    normal = parent(derivative(p))-Ry
    @test norm(normal[1:Ny-1], Inf) < 2e-11
    @test normal[end] ≈ -Ry[Ny] atol=2e-12
    # As above, add the returned constant gradient only to degree zero and
    # test the retained momentum equations plus no-slip walls. Source columns
    # must survive these view-backed solves unchanged.
    for (field, rhs, gradient) in ((u, Rx, gradients[1]), (w, Rz, gradients[2]))
        residual = lambda .* field .-
                   nu .* parent(derivative(derivative(field))) .- rhs
        residual[1] += gradient
        @test norm(residual[1:Ny-2], Inf) < 2e-10
        @test abs(wallvalue(field, :right)) < 2e-11
        @test abs(wallvalue(field, :left)) < 2e-11
    end
    @test data[:, 5:7] == source
    # Pressure gradient and bulk velocity are alternative specifications, not
    # simultaneous constraints. A separate wrong-length call checks
    # coefficient storage validation.
    @test_throws ArgumentError solve!(solver, u, v, w, p, Rx, Ry, Rz;
                                      pressuregradient=(0, 0), bulkvelocity=(0, 0))
    wrong = ntuple(_ -> zeros(ComplexF64, 9), 7)
    @test_throws DimensionMismatch solve!(solver, wrong...)

    # The structured factorisation requires at least four coefficients.
    @test_throws ArgumentError MeanModeSolver(3, nu, 0)
    small = ntuple(_ -> zeros(ComplexF64, 5), 7)
    gradients = solve!(MeanModeSolver(5, nu, 0), small...; bulkvelocity=(2/3, 0))
    @test all(abs.(gradients .- (-2nu, 0)) .< 2e-12)
    @test small[1] ≈ [0.5, 0, -0.5, 0, 0]

end


@testset "Influence-corrected tau coefficients at low degree" begin
    # High-degree forcing at small Ny exposes the mismatch between an old
    # pressure derivative and the influence-corrected auxiliary velocity.
    # Populate every degree with slowly decaying complex forcing so the tau
    # correction cannot be hidden by a low-degree right-hand side. Require the
    # same divergence, wall and retained-equation tolerances for both shifts;
    # this is a regression for recomputing the corrected auxiliary pressure
    # derivative.
    for Ny in (5, 9), lambda in (0.0, 2.5)
        solver = InfluenceModeSolver(Ny, 2.0, -2.0, 1.0, lambda)
        u, v, w, p, Rx, Ry, Rz = ntuple(_ -> zeros(ComplexF64, Ny), 7)
        for (j, rhs) in enumerate((Rx, Ry, Rz)), n in 0:Ny-1
            rhs[n+1] = complex(sin(j+n), cos(2j+n))/(n+1)
        end
        solve!(solver, u, v, w, p, Rx, Ry, Rz)
        check_stokes(solver, u, v, w, p, Rx, Ry, Rz, 1.0)
    end
end
