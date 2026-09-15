function dnsstate(grid)
    P = SpectralField(zeros(ComplexF64, spectralsize(grid, NotPadded())), grid)
    U = VectorField(P)
    for field in U.components
        fill!(parent(field), 0)
    end
    return U, P
end

function zero_rhs!(t, U, N)
    for field in N.components
        fill!(parent(field), 0)
    end
    return N
end

@testset "Channelflow CNRK2" begin
    nu, dt = 0.03, 0.05
    for (profile, base_mean, gradient) in ((y -> y, 0.0, 0.0), (y -> 1-y^2, 2/3, -2nu)),
        form in (CanonicalFlows.ConvectiveForm(), RotatingForm())
        @testset "Laminar base, mean=$base_mean, form=$(typeof(form))" begin
            grid = Grid(9, 4, 4, 2π, 2π, profile)
            U, P = dnsstate(grid)
            if form isa RotatingForm
                pressure = coefficients(y -> profile(y)^2/2, 9)
                pressure[0] -= bulkmean(pressure)
                @views parent(P)[:, 1, 1] .= parent(pressure)
            end
            initialP = copy(parent(P))
            scheme = CNRK2(U, nu, dt)
            nonlinear = NonLinearTerm(PhysicalField(grid), U[1]; form=form, fftwflags=FFTW.ESTIMATE)
            for fixedbulk in (false, true)
                t = 0.0
                for field in scheme.Q.components
                    fill!(parent(field), NaN)  # stage one must overwrite old history
                end
                for _ = 1:3
                    t, actual = fixedbulk ? step!(scheme, nonlinear, U, P, t; bulkvelocity=(base_mean, 0)) :
                                            step!(scheme, nonlinear, U, P, t; pressuregradient=(gradient, 0))
                    @test all(norm(parent(field), Inf) < 2e-11 for field in U.components)
                    @test norm(parent(P)-initialP, Inf) < 2e-10
                    @test all(abs.(actual .- (gradient, 0)) .< 2e-11)
                end
                @test t ≈ 3dt
            end

            # Uniform body forces enter both the explicit recurrence and
            # the old pressure-gradient balance for a fixed bulk constraint.
            force! = function (t, U, F)
                zero_rhs!(t, U, F)
                F[1][1, 1, 1] = 0.7
                F[3][1, 1, 1] = -0.2
            end
            _, actual = step!(scheme, nonlinear, U, P, 0.0;
                              forcing=force!, bulkvelocity=(base_mean, 0))
            @test all(norm(parent(field), Inf) < 2e-11 for field in U.components)
            @test all(abs.(actual .- (gradient+0.7, -0.2)) .< 2e-11)
        end
    end

    @testset "Retained stage pressure" begin
        grid = Grid(9, 5, 3, 2π, 3π, y -> 0.0)
        U, P = dnsstate(grid)
        @views parent(P)[:, 2, 1] .= parent(coefficients(y -> 0.3+0.2y+0.1y^2, 9))
        initialP = copy(parent(P))
        force = VectorField(P)
        ddx1!(force[1], P)
        ddx2!(force[2], P)
        ddx3!(force[3], P)
        force! = (t, U, F) -> (F .= force)
        scheme = CNRK2(U, nu, dt)
        for n = 0:2
            step!(scheme, zero_rhs!, U, P, n*dt; forcing=force!)
            @test all(norm(parent(field), Inf) < 2e-11 for field in U.components)
            @test norm(parent(P)-initialP, Inf) < 2e-10
        end
    end

    @testset "Advection-diffusion order and forcing times" begin
        # w = Re[exp(sigma*t + i*x)] cos(pi*y/2), advected by constant Ub.
        # Self-advection vanishes, so this is an exact Navier--Stokes solution.
        grid = Grid(25, 5, 3, 2π, 3π, y -> 0.7)
        nu, finaltime = 0.15, 0.4
        sigma = -nu*((π/2)^2+1)-0.7im
        shape = parent(coefficients(y -> cospi(y/2)/2, 25))
        errors = Float64[]
        for dt in (0.1, 0.05, 0.025)
            U, P = dnsstate(grid)
            @views parent(U[3])[:, 2, 1] .= shape
            scheme = CNRK2(U, nu, dt)
            nonlinear = NonLinearTerm(PhysicalField(grid), U[1]; fftwflags=FFTW.ESTIMATE)
            for n = 0:round(Int, finaltime/dt)-1
                step!(scheme, nonlinear, U, P, n*dt)
            end
            push!(errors, norm(parent(U[3])[:, 2, 1] - exp(sigma*finaltime).*shape, Inf))
            @test norm(parent(P), Inf) < 2e-10
            @test norm(parent(U[1]), Inf) < 2e-10
            @test norm(parent(U[2]), Inf) < 2e-10
        end
        @test 3.5 < errors[1]/errors[2] < 4.5
        @test 3.5 < errors[2]/errors[3] < 4.5

        # A time-dependent forcing makes w=(1+t+t^2)*shape exact. Record
        # callback times to distinguish RK stage sampling from frozen forcing.
        # Use finer steps here: this solution has a small leading CN error,
        # so the higher-order RK contribution dominates on the coarse grid.
        errors = Float64[]
        for dt in (0.025, 0.0125, 0.00625)
            U, P = dnsstate(grid)
            @views parent(U[3])[:, 2, 1] .= shape
            scheme = CNRK2(U, nu, dt)
            nonlinear = NonLinearTerm(PhysicalField(grid), U[1]; fftwflags=FFTW.ESTIMATE)
            times = Float64[]
            force! = function (t, U, F)
                push!(times, t)
                zero_rhs!(t, U, F)
                @views parent(F[3])[:, 2, 1] .= (1+2t-sigma*(1+t+t^2)) .* shape
            end
            steps = round(Int, finaltime/dt)
            for n = 0:steps-1
                step!(scheme, nonlinear, U, P, n*dt; forcing=force!)
            end
            expectedtimes = [n*dt+c*dt for n = 0:steps-1 for c in (0, 1/3, 3/4)]
            @test times ≈ expectedtimes
            push!(errors, norm(parent(U[3])[:, 2, 1] - (1+finaltime+finaltime^2).*shape, Inf))
        end
        @test 3.5 < errors[1]/errors[2] < 4.5
        @test 3.5 < errors[2]/errors[3] < 4.5
    end

    U, P = dnsstate(Grid(9, 3, 3, 2π, 2π, y -> 0.0))
    @test_throws ArgumentError CNRK2(U, nu, 0)
    @test_throws ArgumentError CNRK2(U, nu, Inf)
    scheme = CNRK2(U, nu, dt)
    @test_throws ArgumentError step!(scheme, zero_rhs!, U, P, 0;
                                     pressuregradient=(0, 0), bulkvelocity=(0, 0))
end
