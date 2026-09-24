using ChannelFlow, CUDA, Adapt, FFTW, LinearAlgebra, Random, Test
const CF=ChannelFlow
CUDA.allowscalar(false)
CUDA.versioninfo()

@testset "CUDA full-step parity" begin
    for form in (CF.RotatingForm(), CF.ConvectiveForm(), CF.DivergenceForm()), N in (8, 16)
        g=Grid(N, 17, N, 2π, 2π)
        cpu=CouetteFlow(g, 1/400, 0.01; form = form, fftwflags = FFTW.ESTIMATE)
        Random.seed!(12)
        state=random_state(cpu, 0.05)
        gpu=adapt(CuArray, cpu)
        device=adapt(CuArray, state)
        # Transfers are explicit and excluded from steady time stepping.
        for j = 1:5
            step!(
                cpu.scheme,
                cpu.nlterm,
                velocity(state),
                stagepressure(state),
                (j-1)*0.01;
                cpu.constraint...,
            )
            step!(
                gpu.scheme,
                gpu.nlterm,
                velocity(device),
                stagepressure(device),
                (j-1)*0.01;
                gpu.constraint...,
            )
        end
        CUDA.synchronize()
        for (a, b) in zip(
            (velocity(state).components..., stagepressure(state)),
            (velocity(device).components..., stagepressure(device)),
        )
            @test Array(parent(b)) ≈ parent(a) rtol=2e-10 atol=2e-11
        end
        divergence=similar(velocity(device)[1])
        CF.div!(divergence, velocity(device))
        @test maximum(abs, parent(divergence)) < 2e-9
        # Check the alternative dense GPU Chebyshev backend independently.
        prototype=velocity(device)[1]
        reference=similar(prototype)
        dense=similar(prototype)
        mul!(reference, CF.plan_cheb(prototype, :cufft), prototype)
        mul!(dense, CF.plan_cheb(prototype, :gemm), prototype)
        @test Array(parent(dense)) ≈ Array(parent(reference)) atol=2e-11
    end
end

@testset "CUDA operators, pressure and constrained integration" begin
    # Nontrivial periodic data exercises every retained Fourier mode. CPU
    # analytic tests establish the mathematics; these comparisons verify that
    # device dispatch preserves it, including alias-safe differentiation.
    g=Grid(8, 17, 8, 2π, 2π)
    p=CouetteFlow(g, 1/400, 0.01; fftwflags = FFTW.ESTIMATE)
    u=PhysicalField(g, (x, y, z)->(1-y^2)^2*exp(cos(x)+cos(z)))
    a=FFT(u)
    da=adapt(CuArray, a)
    for operator in (CF.ddx1!, CF.ddx2!, CF.ddx3!, CF.laplacian!)
        expected=similar(a)
        actual=similar(da)
        operator(expected, a)
        operator(actual, da)
        @test Array(parent(actual)) ≈ parent(expected) atol=2e-10
        alias=copy(da)
        operator(alias, alias)
        @test Array(parent(alias)) ≈ parent(expected) atol=2e-10
    end
    @test Array(parent(IFFT(da))) ≈ parent(IFFT(a)) atol=2e-12
    @test Array(parent(FFT(adapt(CuArray, u)))) ≈ parent(a) atol=2e-12
    @test dot(da, da) ≈ dot(a, a) rtol=2e-12

    for makeproblem in (CouetteFlow, PoiseuilleFlow), fixedflux in (false, true)
        # Poiseuille also checks the base-flow curvature and driving signs.
        # Prescribed flux exercises the separate zero-mode response.
        target=makeproblem===CouetteFlow ? 0.0 : 2/3
        cpu=fixedflux ?
            makeproblem(g, 1/400, 0.01; bulkvelocity = (target, 0.0), fftwflags = FFTW.ESTIMATE) :
            makeproblem(g, 1/400, 0.01; fftwflags = FFTW.ESTIMATE)
        gpu=adapt(CuArray, cpu)
        Random.seed!(82)
        state=random_state(cpu, 0.02)
        U=velocity(state)
        dU=adapt(CuArray, U)
        @test Array(parent(pressure(dU, gpu))) ≈ parent(pressure(U, cpu)) atol=2e-10
        # Applying the projection twice should preserve an admissible field.
        projected=project!(copy(dU), gpu)
        for i = 1:3
            @test Array(parent(projected[i])) ≈ parent(U[i]) atol=2e-10
        end
        device=adapt(CuArray, state)
        # This interval deliberately ends between nominal steps, exercising
        # the Flows adapter and temporary device factors for the final step.
        CF.Flows.flow(cpu)(state, (0.0, 0.025))
        CF.Flows.flow(gpu)(device, (0.0, 0.025))
        for (a, b) in zip(
            (velocity(state).components..., stagepressure(state)),
            (velocity(device).components..., stagepressure(device)),
        )
            @test Array(parent(b)) ≈ parent(a) rtol=2e-9 atol=2e-10
        end
    end
end

@testset "CUDA exact viscous decay" begin
    # Streamwise-only velocity, independent of x, has no self-advection.
    # With zero base profile its analytic amplitude is exp(-nu*k²*t).
    # This is an independent physical check, not merely CPU/GPU agreement.
    g=Grid(8, 33, 8, 2π, 2π)
    nu=0.1
    dt=0.002
    stop=0.1
    cpu=ChannelFlowProblem(g, y->0.0, nu, dt; form = CF.ConvectiveForm(), fftwflags = FFTW.ESTIMATE)
    profile=FFT(PhysicalField(g, (x, y, z)->cos(π*y/2)*cos(z)))
    U=VectorField(copy(profile), zero_state(g).velocity[2], zero_state(g).velocity[3])
    p=adapt(CuArray, cpu)
    s=adapt(CuArray, State(U, pressure(U, cpu)))
    CF.Flows.flow(p)(s, (0.0, stop))
    exact=parent(profile) .* exp(-nu*(π^2/4+1)*stop)
    @test Array(parent(velocity(s)[1])) ≈ exact rtol=2e-8 atol=2e-10
end

@testset "CUDA diagnostics" begin
    p = CouetteFlow(Grid(8, 17, 8, 2π, 2π), 1/400, 0.01; fftwflags = FFTW.ESTIMATE)
    gpu = adapt(CuArray, p)
    @test laminar_power_input(gpu) ≈ 1/400 atol=1e-12
    U = CF._laminar_velocity(p)
    dU = adapt(CuArray, U)
    @test power_input(dU, 1/400) ≈ power_input(U, 1/400) atol=1e-12
    @test dissipation_rate(dU, 1/400) ≈ dissipation_rate(U, 1/400) atol=1e-12
end
