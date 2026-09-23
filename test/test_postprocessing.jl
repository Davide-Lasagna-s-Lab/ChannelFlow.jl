@testset "Energy and vorticity dissipation" begin
    g = Grid(8, 17, 8, 2π, 2π)
    nu = 0.01

    # Supply the complete analytic velocity explicitly. Couette has energy
    # <y²>/2 = 1/6 and constant vorticity, hence dissipation nu.
    U = VectorField(spectral(g, (x,y,z) -> y),
                    spectral(g, fzero), spectral(g, fzero))
    saved = copy(U)
    @test kinetic_energy(U) ≈ 1/6 atol=2e-14
    @test dissipation_rate(U, nu) ≈ nu atol=2e-14
    @test LinearAlgebra.dot(U, U) ≈ 1/3 atol=2e-14
    @test LinearAlgebra.norm(U)^2 ≈ 1/3 atol=2e-14
    @test all(parent(U[i]) == parent(saved[i]) for i = 1:3)

    # The mixed inner product of odd Couette and even Poiseuille profiles is
    # zero. This checks a two-field integral, not just the squared norm.
    V = VectorField(spectral(g, (x,y,z) -> 1-y^2),
                    spectral(g, fzero), spectral(g, fzero))
    @test LinearAlgebra.dot(U, V) ≈ 0 atol=2e-14
    @test kinetic_energy(V) ≈ 4/15 atol=2e-14
    @test kinetic_energy(IFFT(V)) ≈ 4/15 atol=2e-14
    @test dissipation_rate(V, nu) ≈ 4nu/3 atol=2e-14

    # The supplied zero perturbation remains zero; no base profile is added
    # implicitly by either diagnostic.
    Z = velocity(zero_state(g))
    @test kinetic_energy(Z) == 0
    @test dissipation_rate(Z, nu) == 0
end
