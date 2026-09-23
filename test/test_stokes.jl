@testset "Fourier Stokes assembly" begin
    # Test the Fourier wrapper, not only isolated mode solves: independently
    # assemble exact coefficients and forcing for every active mode.
    # Nonstandard domain lengths test physical wavenumber scaling. The two
    # base profiles test nonzero and zero base bulk means.
    Ny, nu, lambda = 9, 0.03, 2.5
    Lx, Lz = 5.3, 7.1
    gradients = (-0.4, 0.15)
    profiles = ((y -> 1-y^2, 2/3), (y -> y, 0.0))

    # Cover both Fourier parities, negative kz, single periodic points and
    # a grid whose only active mode is the mean (all other slots are Nyquist).
    for (Nx, Nz) in ((6, 6), (5, 5), (6, 5), (5, 6), (1, 5), (5, 1), (1, 1), (2, 2)),
        (profile, base_mean) in profiles
        @testset "Nx=$Nx, Nz=$Nz, base mean=$base_mean" begin
            grid = Grid(Nx, Ny, Nz, Lx, Lz)
            solver = StokesSolver(grid, nu, lambda)
            prototype = SpectralField(zeros(ComplexF64, spectralsize(grid, NotPadded())), grid)
            U, R, exact = ntuple(_ -> VectorField(prototype), 3)
            P, exactP, work, grad = ntuple(_ -> similar(prototype), 4)
            for field in (U.components..., R.components..., exact.components..., P, exactP)
                fill!(parent(field), 0)
            end

            # Build analytic, divergence-free modes independently of the
            # solver's index-to-wavenumber mapping. Even-grid Nyquist kz is
            # in the negative block here; that slot must be excluded anyway.
            zmodes = vcat(0:fld(Nz-1, 2), -fld(Nz, 2):-1)
            for (iz, mz) in enumerate(zmodes), (ix, mx) in enumerate(0:fld(Nx, 2))
                # Put conspicuous nonzero data in excluded Nyquist source
                # planes. The output must still be zero there, while the
                # source remains untouched; silently solving these planes or
                # leaving stale output will fail.
                if (iseven(Nx) && mx == Nx÷2) || (iseven(Nz) && mz == -Nz÷2)
                    for field in R.components
                        @views parent(field)[ix, iz, :] .= 7+3im
                    end
                    continue
                end
                kx, kz = 2π*mx/Lx, 2π*mz/Lz
                meanmode = mx == mz == 0
                # Treat the mean separately to avoid division by zero and set
                # a zero-mean pressure gauge. Other modes use a wall-vanishing
                # divergence-free polynomial with an additional transverse
                # component. This is a coefficient-space assembly test;
                # arbitrary modal data need not satisfy real-field conjugate
                # symmetry.
                if meanmode
                    uf, vf, wf = y -> 0.2*(1-y^2), y -> 0.0, y -> -0.1*(1-y^2)
                    d2uf, d2vf, d2wf = y -> -0.4, y -> 0.0, y -> 0.2
                    pf, dpf = y -> 0.3y+0.2*(y^2-1/3), y -> 0.3+0.4y
                else
                    k2 = kx^2+kz^2
                    a, b = 0.1/(1+mx^2+mz^2), 0.05im
                    vf = y -> a*(1-y^2)^2
                    dvf = y -> a*(-4y+4y^3)
                    d2vf = y -> a*(-4+12y^2)
                    d3vf = y -> 24a*y
                    uf = y -> im*kx/k2*dvf(y)+kz*b*(1-y^2)
                    wf = y -> im*kz/k2*dvf(y)-kx*b*(1-y^2)
                    d2uf = y -> im*kx/k2*d3vf(y)-2kz*b
                    d2wf = y -> im*kz/k2*d3vf(y)+2kx*b
                    pf, dpf = y -> 0.3+0.2y+0.1y^2, y -> 0.2+0.2y
                end
                # Use analytic second derivatives to manufacture the Stokes
                # right-hand side. A constant imposed gradient enters only the
                # mean mode; nonzero modes instead use i*k times the pressure
                # coefficients.
                shift = lambda+nu*(kx^2+kz^2)
                rxf = y -> shift*uf(y)-nu*d2uf(y)+(meanmode ? gradients[1] : im*kx*pf(y))
                ryf = y -> shift*vf(y)-nu*d2vf(y)+dpf(y)
                rzf = y -> shift*wf(y)-nu*d2wf(y)+(meanmode ? gradients[2] : im*kz*pf(y))
                for (field, f) in zip((exact.components..., exactP, R.components...),
                                      (uf, vf, wf, pf, rxf, ryf, rzf))
                    @views parent(field)[ix, iz, :] .= parent(coefficients(f, Ny))
                end
            end
            source = map(field -> copy(parent(field)), R.components)
            # Exercise both ways of specifying the mean constraint. The
            # prescribed total streamwise bulk velocity includes base_mean,
            # whereas the stored velocity is a perturbation. Prefilling
            # outputs with nonzero data detects incomplete overwrite,
            # including excluded planes.
            for fixedbulk in (false, true)
                for field in (U.components..., P)
                    fill!(parent(field), 9+2im)
                end
                actual = fixedbulk ? solve!(solver, U, P, R; bulkvelocity=(base_mean+0.2*2/3, -0.1*2/3), baseflow=parent(ChebyshevHelmoltzSolvers.chebcoeffs(profile.(grid.y)))) :
                                     solve!(solver, U, P, R; pressuregradient=gradients)
                # Recover the known pressure gradients, all velocity
                # coefficients and the pressure field, while preserving every
                # source coefficient. These direct analytic comparisons are
                # the primary correctness check.
                @test all(abs.(actual .- gradients) .< 2e-11)
                for i = 1:3
                    @test norm(parent(U[i])-parent(exact[i]), Inf) < 2e-10
                end
                @test norm(parent(P)-parent(exactP), Inf) < 2e-10
                @test map(parent, R.components) == source

                # Check the assembled fields with the field operators used
                # by DNS, not by calling the modal solver again.
                ddx1!(work, U[1])
                ddx2!(work, U[2], true)
                ddx3!(work, U[3], true)
                @test norm(parent(work), Inf) < 2e-10
                for (i, derivative!) in enumerate((ddx1!, ddx2!, ddx3!))
                    ChannelFlow.laplacian!(work, U[i])
                    derivative!(grad, P)
                    parent(work) .= lambda .* parent(U[i]) .- nu .* parent(work) .+
                                    parent(grad) .- parent(R[i])
                    i == 1 && (work[1, 1, 1] += actual[1])
                    i == 3 && (work[1, 1, 1] += actual[2])
                    # The source intentionally contains junk on excluded
                    # planes, so omit those residuals. Likewise only degrees 0
                    # through Ny-3 enforce momentum equations; the top two
                    # coefficients carry tau residuals for the wall
                    # conditions.
                    ChannelFlow.zero_nyquist!(work)
                    @test norm(view(parent(work), :, :, 1:Ny-2), Inf) < 2e-10
                end
                # Quadrature of the mean Fourier column verifies the
                # total/perturbation bulk conversion and the pressure gauge
                # independently of the solver internal mean formula.
                @test real(bulkmean(view(parent(U[1]), 1, 1, :))) + base_mean ≈
                      base_mean+0.2*2/3 atol=2e-11
                @test real(bulkmean(view(parent(U[3]), 1, 1, :))) ≈ -0.1*2/3 atol=2e-11
                @test abs(bulkmean(view(parent(P), 1, 1, :))) < 2e-12
            end
        end
    end

    # Check wrapper validation independently of the algebra: incompatible
    # domain, padded pressure, a single malformed source component,
    # conflicting constraints and unsupported grid sizes or lengths must be
    # rejected.
    grid = Grid(5, Ny, 5, Lx, Lz)
    solver = StokesSolver(grid, nu, lambda)
    P = SpectralField(zeros(ComplexF64, spectralsize(grid, NotPadded())), grid)
    U, R = VectorField(P), VectorField(P)
    other = Grid(5, Ny, 5, 2Lx, Lz)
    badgrid = SpectralField(copy(parent(P)), other)
    padded = SpectralField(zeros(ComplexF64, spectralsize(grid, Padded())), grid)
    @test_throws ArgumentError solve!(solver, U, badgrid, R)
    @test_throws DimensionMismatch solve!(solver, U, padded, R)
    badR = VectorField((R[1], R[2], padded))
    @test_throws DimensionMismatch solve!(solver, U, P, badR)
    @test_throws ArgumentError solve!(solver, U, P, R;
                                      pressuregradient=(0, 0), bulkvelocity=(0, 0))
    @test_throws ArgumentError StokesSolver(Grid(5, 10, 5, Lx, Lz), nu, lambda)
    @test_throws ArgumentError StokesSolver(Grid(0, Ny, 5, Lx, Lz), nu, lambda)
    @test_throws ArgumentError StokesSolver(Grid(5, Ny, 5, 0, Lz), nu, lambda)
end
