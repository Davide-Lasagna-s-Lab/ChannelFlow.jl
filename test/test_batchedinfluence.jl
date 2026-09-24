@testset "Batched influence solver versus independent modal solves" begin
    # Excite every polynomial degree in complex sources, including the two
    # tau residuals. A batch must agree with scalar reference solves and
    # satisfy continuity, retained momentum equations and both walls.
    for Ny in (5, 9, 17, 33), lambda in (0.0, 3.0)
        k = [1.0, 0.0, 1.25, 2.0]
        l = [0.0, -2.0, 1.75, -0.5]
        B, nu = length(k), 0.03
        h = CF.BatchedInfluenceSolver(Ny,k,l,nu,lambda)
        R = ntuple(_ -> randn(ComplexF64,B,Ny),3)
        saved = map(copy,R)
        outputs = ntuple(_ -> zeros(ComplexF64,B,Ny),4)
        for repeat = 1:2
            solve!(h,outputs...,R...)
            @test R == saved
            for s = 1:B
                reference = InfluenceModeSolver(Ny,k[s],l[s],nu,lambda)
                exact = ntuple(_ -> zeros(ComplexF64,Ny),4)
                sources = map(a -> view(a,s,:),R)
                solve!(reference,exact...,map(copy,sources)...)
                for (out, target) in zip(outputs,exact)
                    @test out[s,:] ≈ target rtol=2e-11 atol=2e-11
                end
                check_stokes(reference,map(a -> view(a,s,:),outputs)...,sources...,nu)
            end
        end
    end
end
