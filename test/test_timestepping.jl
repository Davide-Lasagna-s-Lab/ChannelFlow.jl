@testset "Time-stepper and Flows interface" begin
    g = Grid(5, 9, 5, 2π, 2π)
    base = zeros(9)
    nu, dt = 0.1, 0.02
    # Reject zero/nonfinite time steps, an incompatible base-flow length and
    # simultaneous bulk-velocity/pressure-gradient constraints before
    # integration begins.
    @test_throws ArgumentError CNRK2(g, base, nu, 0)
    @test_throws ArgumentError CNRK2(g, base, nu, Inf)
    @test_throws DimensionMismatch CNRK2(g, zeros(8), nu, dt)
    @test_throws ArgumentError ChannelFlowProblem(g, y -> 0.0, nu, dt;
        pressuregradient=(0,0), bulkvelocity=(0,0), fftwflags=FFTW.ESTIMATE)

    # A zero forcing callback records its invocation times without introducing
    # nontrivial dynamics. This makes the test an integration-interface and
    # scheduling check, not a physical solution or accuracy benchmark.
    times = Float64[]
    force! = function (t, U, F)
        push!(times, t)
        F .= 0
    end
    problem = ChannelFlowProblem(g, y -> 0.0, nu, dt;
                                 forcing=force!, fftwflags=FFTW.ESTIMATE)
    @test problem.nlterm isa NonLinearTerm{Float64, RotatingForm}
    state = zero_state(g)
    U, P = velocity(state), stagepressure(state)
    flow = Flows.flow(problem)
    # Two nominal steps and a shortened final step exercise adapter dispatch
    # and stage sampling; no physical time-evolution benchmark is used here.
    finaltime = 2.5dt
    @test flow(state, (0.0, finaltime)) === state
    # CNRK2 samples forcing at offsets 0, 1/3 and 3/4 of each step. The last
    # half-step must use its own shorter duration in these offsets, while
    # preserving the nominal dt for subsequent calls.
    expected = [t+c*h for (t,h) in ((0.0,dt),(dt,dt),(2dt,dt/2))
                       for c in (0.0,1/3,3/4)]
    @test times ≈ expected
    # Integration must retain the state buffers, restore the nominal time step
    # and produce finite values. Buffer identity checks the in-place contract
    # rather than merely equality of zero data.
    @test velocity(state) === U
    @test stagepressure(state) === P
    @test problem.scheme.dt == dt
    @test all(all(isfinite, field) for field in (U.components..., P))
    # The low-level step interface must enforce the same exclusive
    # constraints. A state belonging to another grid object must also be
    # rejected by the configured flow adapter, even when its dimensions match.
    @test_throws ArgumentError step!(problem.scheme, problem.nlterm, U, P, 0;
                                    pressuregradient=(0,0), bulkvelocity=(0,0))
    other = zero_state(Grid(5, 9, 5, 2π, 2π))
    @test_throws ArgumentError flow(other, (0.0, dt))
end
