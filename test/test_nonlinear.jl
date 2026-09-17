@testset "Analytic nonlinear forms and base-flow input" begin
    # Use perturbation velocity (y^2,1,0) and base flow (y,0,0). The total
    # velocity is divergence-free, so convective and divergence forms agree.
    # This is an algebraic test field; it deliberately need not satisfy
    # channel wall conditions.
    g = Grid(5, 9, 5, 2π, 2π)
    U = VectorField((spectral(g, (x,y,z) -> y^2),
                     spectral(g, (x,y,z) -> 1.0), spectral(g, fzero)))
    original = map(field -> copy(parent(field)), U.components)
    base = ChebyshevHelmoltzSolvers.chebcoeffs(g.y)
    for form in (CF.ConvectiveForm(), CF.DivergenceForm(),
                 RotatingForm())
        op = NonLinearTerm(PhysicalField(g), U[1], base;
                           form=form, fftwflags=FFTW.ESTIMATE)
        N = similar(U)
        # The momentum contribution is minus advection: -(2y+1) in x. The
        # base-flow derivative supplies the +1. The rotating form uses u cross
        # curl(u), adding (y^2+y)*(2y+1) in y; it differs by a kinetic-energy
        # gradient, so comparing it to the convective vector would be
        # incorrect.
        expected = ((x,y,z) -> -(2y+1),
                    form isa RotatingForm ? (x,y,z) -> (y^2+y)*(2y+1) : fzero,
                    fzero)
        # Overwrite, accumulate, then overwrite again: the expected
        # multipliers are 1,2,1. Repeated calls also exercise
        # cache reuse. Every polynomial product fits the wall-
        # normal resolution.
        for add in (false, true, false)
            op(0.0, U, N, add)
            for i=1:3
                @test physical_values(N[i]) ≈ (add ? 2 : 1) .* parent(sampled(g, expected[i])) atol=2e-11
            end
        end
        # The nonlinear evaluation may use scratch arrays but must leave the
        # supplied perturbation coefficients unchanged.
        @test map(parent, U.components) == original
    end
end
