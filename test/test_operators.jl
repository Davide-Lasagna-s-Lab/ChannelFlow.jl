@testset "Analytic derivatives and tensor operators" begin
    # A cubic in y times exp(cos(ax)+cos(bz)) exercises every periodic
    # harmonic. Explicit derivatives remain independent of the operators.
    # The larger grids suppress Fourier truncation below the test tolerances;
    # parity and unequal domain lengths still probe indexing and scaling.
    for Ny in (9, 17), (Nx, Nz) in ((40, 42), (41, 43))
        g = Grid(Nx, Ny, Nz, 5.3, 7.1)
        a, b = 2π/5.3, 2π/7.1
        p(y) = 1 + 2y + 3y^2 + 4y^3
        dp(y) = 2 + 6y + 12y^2
        d2p(y) = 6 + 24y
        # Normalize the peak to one without changing harmonic support.
        h(x,z) = exp(cos(a*x)+cos(b*z)-2)
        f(x,y,z) = p(y)*h(x,z)
        fx(x,y,z) = -a*p(y)*sin(a*x)*h(x,z)
        fy(x,y,z) = dp(y)*h(x,z)
        fz(x,y,z) = -b*p(y)*sin(b*z)*h(x,z)
        lap(x,y,z) = (d2p(y)+p(y)*(a^2*(sin(a*x)^2-cos(a*x))+
                                                 b^2*(sin(b*z)^2-cos(b*z))))*h(x,z)
        U = spectral(g, f)
        saved = copy(parent(U))
        for (op, expected) in ((ddx1!, fx), (ddx2!, fy), (ddx3!, fz))
            # Prefill with NaN to detect incomplete overwrites. Check the
            # returned destination, analytic derivative values, and
            # preservation of the input. Tolerances allow transform and
            # differentiation roundoff, not truncation error.
            out = similar(U)
            fill!(out, NaN)
            @test op(out, U) === out
            @test physical_values(out) ≈ parent(sampled(g, expected)) atol=2e-10
            @test parent(U) == saved
            # Accumulation must add one more derivative, doubling the result.
            # The alias case then verifies that overwriting the input itself
            # does not destroy coefficients still needed by the calculation.
            op(out, U, true)
            @test physical_values(out) ≈ 2 .* parent(sampled(g, expected)) atol=4e-10
            alias = copy(U)
            op(alias, alias)
            @test physical_values(alias) ≈ parent(sampled(g, expected)) atol=2e-10
        end
        # The Laplacian combines p double-prime times h with p times the
        # periodic Laplacian of h. Compare the analytic expression, then
        # verify that aliased execution
        # matches the distinct-buffer result. Second derivatives amplify
        # roundoff more strongly.
        out = similar(U)
        @test CF.laplacian!(out, U) === out
        @test physical_values(out) ≈ parent(sampled(g, lap)) atol=2e-9
        alias = copy(U)
        CF.laplacian!(alias, alias)
        @test parent(alias) ≈ parent(out) atol=2e-10

        # For three copies of f, every gradient row is (fx,fy,fz). This checks
        # component/direction ordering. Vector divergence is fx+fy+fz; tensor
        # divergence of the gradient must return the scalar Laplacian in each
        # component.
        V = VectorField((copy(U), copy(U), copy(U)))
        G = CF.GradientField(U)
        @test CF.grad!(G, V) === G
        for i = 1:3, (j, exact) in enumerate((fx, fy, fz))
            @test physical_values(G[i, j]) ≈ parent(sampled(g, exact)) atol=2e-10
        end
        # With all three velocity components equal to f, the analytic curl is
        # (fy-fz, fz-fx, fx-fy). This checks signs, component ordering and the
        # fused combination of Chebyshev and Fourier derivatives.
        Ω = similar(V)
        @test curl!(Ω, V) === Ω
        exact_curl = ((x,y,z) -> fy(x,y,z)-fz(x,y,z),
                      (x,y,z) -> fz(x,y,z)-fx(x,y,z),
                      (x,y,z) -> fx(x,y,z)-fy(x,y,z))
        @test all(isapprox(physical_values(Ω[i]),
                           parent(sampled(g, exact_curl[i])); atol=4e-10)
                  for i = 1:3)
        @test CF.div!(out, V) === out
        @test physical_values(out) ≈ parent(sampled(g, (x,y,z) -> fx(x,y,z)+fy(x,y,z)+fz(x,y,z))) atol=3e-10
        @test CF.div!(V, G) === V
        @test all(isapprox(physical_values(V[i]), parent(sampled(g, lap)); atol=2e-9) for i = 1:3)
    end

    # Constant components (1,2,3) isolate tensor algebra from transforms and
    # derivatives. The outer product entry is i*j, and contraction gives
    # i*(1^2+2^2+3^2)=14*i. These integer-valued operations admit exact
    # comparisons.
    g = Grid(5, 9, 5, 2π, 2π)
    u = VectorField(sampled(g, fzero))
    for i = 1:3
        u[i] .= i
    end
    tensor = CF.GradientField(u[1])
    @test CF.outer!(tensor, u, u) === tensor
    @test all(all(==(i*j), tensor[i,j]) for i = 1:3, j = 1:3)
    out = similar(u)
    @test CF.dot!(out, u, tensor) === out
    @test all(all(==(14i), out[i]) for i = 1:3)
end
