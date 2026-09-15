using CanonicalFlows
using ChebyshevHelmoltzSolvers: ChebCoeffs, diff!, endpoint_derivative
using FFTW
using LinearAlgebra
using Test

function coefficients(f, Ny)
    P = Ny - 1
    a = FFTW.r2r(ComplexF64[f(cospi(n/P)) for n = 0:P], FFTW.REDFT00)/P
    a[1] /= 2
    a[end] /= 2
    return ChebCoeffs(a)
end

derivative(a) = diff!(similar(a), a)
wallvalue(a, side) = sum(a[n]*(side == :right ? 1 : (-1)^n) for n = 0:length(a)-1)

@testset "VectorField broadcast" begin
    grid = Grid(9, 4, 4, 2π, 2π, y -> 0.0)
    prototype = SpectralField(zeros(ComplexF64, spectralsize(grid, NotPadded())), grid)
    A, B = VectorField(prototype), VectorField(prototype)
    for i = 1:3
        B[i] .= i
    end
    A .= B
    @test all(parent(A[i]) == parent(B[i]) for i = 1:3)
    A .-= B
    @test all(iszero, (norm(parent(A[i])) for i = 1:3))
end

function check_stokes(solver, u, v, w, p, Rx, Ry, Rz, nu)
    P = length(p) - 1
    dp = derivative(p)
    divergence = im*solver.kx .* parent(u) .+ parent(derivative(v)) .+
                 im*solver.kz .* parent(w)
    @test norm(divergence, Inf) < 2e-10

    for (field, source, gradp) in ((u, Rx, im*solver.kx .* parent(p)),
                                   (v, Ry, parent(dp)),
                                   (w, Rz, im*solver.kz .* parent(p)))
        residual = solver.lambda .* parent(field) .-
                   nu .* parent(derivative(derivative(field))) .+
                   gradp .- parent(source)
        # The highest two residual coefficients are the momentum tau terms.
        @test norm(residual[1:P-1], Inf) < 2e-10
        @test abs(wallvalue(field, :left)) < 2e-11
        @test abs(wallvalue(field, :right)) < 2e-11
    end
    @test abs(endpoint_derivative(v, :left)) < 2e-11
    @test abs(endpoint_derivative(v, :right)) < 2e-11
end

@testset "Complex primitive-variable mode" begin
    @test_throws ArgumentError InfluenceModeSolver(10, 1, 1, 0.1, 2)
    @test_throws ArgumentError InfluenceModeSolver(9, 0, 0, 0.1, 2)

    for Ny in (9, 17, 33), (kx, kz) in ((1.25, 0.0), (0.0, -2.0),
                                        (1.25, -1.75), (1.25, 1.75))
        @testset "Ny=$Ny, k=($kx,$kz)" begin
            nu, lambda = 0.03, 2.5
            solver = InfluenceModeSolver(Ny, kx, kz, nu, lambda)
            kappa2 = kx^2 + kz^2
            shift = lambda + nu*kappa2
            vfun(y) = (0.7+0.2im)*(1-y^2)^2*(1+0.2y)
            dvfun(y) = (0.7+0.2im)*(-4y+4y^3+0.2*(1-6y^2+5y^4))
            d2vfun(y) = (0.7+0.2im)*(-4+12y^2+0.2*(-12y+20y^3))
            d3vfun(y) = (0.7+0.2im)*(24y+0.2*(-12+60y^2))
            qfun(y) = (0.15-0.1im)*(1-y^2)*(1+0.3y)
            d2qfun(y) = (0.15-0.1im)*(-2-1.8y)
            ufun(y) = im*kx/kappa2*dvfun(y) + kz*qfun(y)
            wfun(y) = im*kz/kappa2*dvfun(y) - kx*qfun(y)
            pfun(y) = (0.4-0.3im)*(1+y+0.2y^3)
            dpfun(y) = (0.4-0.3im)*(1+0.6y^2)

            Rx = coefficients(y -> shift*ufun(y) -
                                   nu*(im*kx/kappa2*d3vfun(y)+kz*d2qfun(y)) +
                                   im*kx*pfun(y), Ny)
            Ry = coefficients(y -> shift*vfun(y)-nu*d2vfun(y)+dpfun(y), Ny)
            Rz = coefficients(y -> shift*wfun(y) -
                                   nu*(im*kz/kappa2*d3vfun(y)-kx*d2qfun(y)) +
                                   im*kz*pfun(y), Ny)
            source = map(x -> copy(parent(x)), (Rx, Ry, Rz))
            responses = map(x -> copy(parent(x)),
                            (solver.pressure_plus, solver.pressure_minus,
                             solver.velocity_plus, solver.velocity_minus,
                             solver.pressure_zero, solver.velocity_zero))
            u, v, w, p = ntuple(_ -> ChebCoeffs(Ny-1, ComplexF64), 4)

            # Repeated solves must overwrite the outputs without changing
            # either the source or the precomputed influence/tau responses.
            for _ = 1:2
                @test solve!(solver, u, v, w, p, Rx, Ry, Rz) == (u, v, w, p)
                for (field, exact) in zip((u, v, w, p), (ufun, vfun, wfun, pfun))
                    @test norm(parent(field)-parent(coefficients(exact, Ny)), Inf) < 2e-10
                end
                check_stokes(solver, u, v, w, p, Rx, Ry, Rz, nu)
                @test map(parent, (Rx, Ry, Rz)) == source
                @test map(parent, (solver.pressure_plus, solver.pressure_minus,
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
    u, v, w, p, Rx, Ry, Rz = ntuple(i -> ChebCoeffs(view(data, :, i)), 7)
    for (j, rhs) in enumerate((Rx, Ry, Rz)), n = 0:Ny-1
        rhs[n] = complex(sin((j+1)*(n+1)), cos((j+2)*(n+1)))/(n+1)^2
    end
    source = copy(data[:, 5:7])
    solve!(solver, u, v, w, p, Rx, Ry, Rz)
    check_stokes(solver, u, v, w, p, Rx, Ry, Rz, 0.03)
    @test data[:, 5:7] == source

    wrong = ntuple(_ -> ChebCoeffs(8, ComplexF64), 7)
    @test_throws DimensionMismatch solve!(solver, wrong...)
end
