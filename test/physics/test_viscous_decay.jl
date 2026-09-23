#//////////////////////////////////////////////////////////////////////////////#
#///                         EXACT VISCOUS DECAY                            ///#
#//////////////////////////////////////////////////////////////////////////////#

@testset "Exact viscous decay" begin
    # With only streamwise velocity and no x dependence, all convective terms
    # vanish, including interactions with the base profile. The full nonlinear
    # DNS must therefore reproduce diffusion of this Laplacian eigenfunction:
    # u'=A*sin(n*pi*(y+1)/2)*cos(z)*exp(-mu*t), v'=w'=0.
    # A single Fourier mode is intentional: exp(cos(z)) would have a different
    # decay rate for each harmonic and no single exponential energy law.
    Re, A, T = 100.0, 0.1, 4.0
    nu = 1/Re
    g = Grid(5, 33, 8, 2π, 2π)
    Nx, Nz, Ny = physicalsize(g, Padded())

    # Integrate physical-space energy with interpolatory Chebyshev quadrature.
    # The weights reproduce integral(T_j,-1,1), independently of the solver's
    # bulk constraint. Periodic averages use the uniform padded mesh.
    C = [cospi(i*j/(Ny-1)) for i=0:Ny-1, j=0:Ny-1]
    moments = [iseven(j) ? 2/(1-j^2) : 0.0 for j=0:Ny-1]
    weights = transpose(C) \ moments
    energy(u) = sum(weights .* vec(sum(abs2, u; dims=(1,2))))/(4Nx*Nz)

    for (name, profile, gradient) in (("Poiseuille", y -> 1-y^2, -2nu),
                                      ("Couette", y -> y, 0.0)), n in (1,3)
        @testset "$name, n=$n" begin
            alpha = n*π/2
            mu = nu*(alpha^2+1)
            initial(x,y,z) = A*sin(alpha*(y+1))*cos(z)
            initial_values = parent(sampled(g, initial))
            E0 = energy(initial_values)
            # Mean(sin^2)=mean(cos^2)=1/2, including the 1/2 energy factor.
            @test E0 ≈ A^2/8 rtol=1e-12
            errors = Float64[]
            for dt in (0.5, 0.25, 0.125)
                problem = ChannelFlowProblem(g, profile, nu, dt;
                    form=CF.ConvectiveForm(), pressuregradient=(gradient,0.0),
                    fftwflags=FFTW.ESTIMATE)
                state = zero_state(g)
                U = velocity(state)
                U[1] .= spectral(g, initial)
                # Pressure is exactly zero for the convective formulation;
                # the sustaining uniform gradient is configured separately.
                # No projection is needed: the analytic data are solenoidal.
                flow = Flows.flow(problem)
                for t in (T/2,T)
                    flow(state, (t-T/2,t))
                    u = physical_values(U[1])
                    expected = initial_values .* exp(-mu*t)
                    error = maximum(abs, u-expected)/maximum(abs, expected)
                    @test error < 1e-3
                    @test energy(u)/E0 ≈ exp(-2mu*t) rtol=2e-3
                    # Infer amplitude decay from energy, retaining the factor
                    # two between the amplitude and energy exponents.
                    rate = -log(energy(u)/E0)/(2t)
                    @test rate ≈ mu rtol=2e-3
                    @test maximum(abs, parent(U[2])) < 1e-11
                    @test maximum(abs, parent(U[3])) < 1e-11
                    @test maximum(abs, parent(stagepressure(state))) < 1e-10
                    check_constraints(U)

                    # Signed perturbation shear nu*du'/dy, at both walls.
                    # This is the xy stress component, not outward traction;
                    # the base shear is excluded on both sides of comparison.
                    shear = similar(U[1])
                    ddx2!(shear,U[1])
                    numerical = nu .* physical_values(shear)
                    exact = parent(sampled(g, (x,y,z) ->
                        nu*A*alpha*cos(alpha*(y+1))*cos(z)*exp(-mu*t)))
                    @test maximum(abs, numerical[[1,end],:,:]-exact[[1,end],:,:]) <
                          1e-3*nu*A*alpha*exp(-mu*t)
                    # beta=1 has zero periodic mean, so no perturbation flux
                    # or unintended change of the sustained base is allowed.
                    @test maximum(abs, parent(U[1])[1, 1, :]) < 1e-10
                    t == T && push!(errors,error)
                end
            end
            # Spatial error is negligible at Ny=33 for n<=3. Halving dt must
            # reduce the final field error by approximately four for CNRK2.
            @test 3.5 < errors[1]/errors[2] < 4.5
            @test 3.5 < errors[2]/errors[3] < 4.5
        end
    end
end
