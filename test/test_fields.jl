@testset "Grid, fields and state interface" begin
    # Check the public size convention explicitly: y is first, padding
    # enlarges only periodic axes, and the real FFT reduces x to floor(Nx/2)+1
    # coefficients. Padding rounds 3N/2 upward without forcing odd sizes;
    # the wall-normal interval always has length 2.
    g = Grid(6, 9, 8, 5.0, 7.0)
    @test physicalsize(g, NotPadded()) == (9, 6, 8)
    @test physicalsize(g, Padded()) == (9, 9, 12)
    @test spectralsize(g, NotPadded()) == (9, 4, 8)
    @test spectralsize(g, Padded()) == (9, 5, 12)
    @test CF.domainsize(g) == (5.0, 2.0, 7.0)
    # Check node placement independently: Chebyshev-Lobatto points descend
    # from +1 to -1, while periodic nodes start at zero and omit the
    # duplicated endpoint.
    y, x, z = points(g)
    @test vec(y) ≈ cospi.((0:8) ./ 8)
    @test vec(x) ≈ (0:5) .* (5/6)
    @test vec(z) ≈ (0:7) .* (7/8)
    # Use y^2=(T_2+T_0)/2 to obtain the coefficients by hand. This checks the
    # constant coefficient normalization as well as polynomial conversion.
    @test parent(ChebyshevHelmoltzSolvers.chebyshev_coefficients(1 .+ 2 .* g.y .+ 3 .* g.y.^2)) ≈
          [2.5, 2, 1.5, zeros(6)...] atol=1e-13

    # Different weights on x, y and z expose argument-order mistakes in the
    # functional constructor. The zero constructor and shape rejection
    # establish allocation defaults and grid/storage consistency.
    f(x, y, z) = x + 2y + 3z
    u = PhysicalField(g, f)
    @test parent(u) ≈ f.(x, y, z)
    @test all(iszero, PhysicalField(g))
    @test_throws DimensionMismatch PhysicalField(zeros(2, 2, 2), g)
    # Both physical and spectral fields must preserve grid identity when
    # copied or used as allocation templates, but allocate independent data.
    # similar() values are intentionally not inspected before assignment
    # because its contents need not be initialized.
    for field in (u, spectral(g, fzero))
        other = copy(field)
        work = similar(field)
        @test CF.grid(other) === CF.grid(work) === g
        @test parent(other) !== parent(field)
        @test parent(work) !== parent(field)
        @test size(work) == size(field)
        @test parent(other) == parent(field)
        # Broadcast must operate on stored values; scalar indexing must
        # address the same storage. Index zero is invalid for a field array,
        # unlike degree indexing in ChebCoeffs.
        work .= 2 .* field .+ 1
        @test parent(work) ≈ 2 .* parent(field) .+ 1
        work[1, 2, 3] = 4
        @test work[1, 2, 3] == 4
        @test_throws BoundsError work[0, 1, 1]
    end

    # A zero state owns three distinct velocity arrays and a separate pressure
    # array. Mutating one component detects accidental shared allocation.
    # State copy/similar must likewise allocate fresh component and pressure
    # storage.
    s = zero_state(g)
    U, P = velocity(s), stagepressure(s)
    @test s isa State
    @test all(iszero, P)
    @test all(component -> all(iszero, component), U.components)
    U[1][1] = 2
    @test U[2][1] == U[3][1] == P[1] == 0
    for other in (copy(s), similar(s))
        @test CF.grid(stagepressure(other)) === g
        @test parent(stagepressure(other)) !== parent(P)
        for i = 1:3
            @test parent(velocity(other)[i]) !== parent(U[i])
        end
    end
    # The vector broadcast identity 2*copy(U)-U=U checks componentwise
    # assignment without a manual loop in user code. Copying a state must
    # preserve pressure values too.
    A, B = similar(U), copy(U)
    A .= 2 .* B .- U
    @test all(parent(A[i]) == parent(U[i]) for i = 1:3)
    @test parent(stagepressure(copy(s))) == parent(P)
    # Two-index tensor access and nested row access must refer to the same
    # component. Off-diagonal and diagonal arrays must remain independent when
    # one entry is changed.
    tensor = CF.GradientField(P)
    tensor[1, 2][1] = 7
    @test tensor[1][2][1] == 7
    @test tensor[2, 1][1] == tensor[1, 1][1] == 0
end

@testset "Field broadcast axes" begin
    # A wall-normal profile has singleton periodic axes. Assignment must
    # expand it across the destination, rather than index it as a full field.
    g = Grid(6, 9, 8, 5.0, 7.0)
    profile_grid = Grid(1, 9, 1, 5.0, 7.0)
    bad_grid = Grid(4, 9, 5, 5.0, 7.0)
    for constructor in (PhysicalField, SpectralField)
        physical = constructor === PhysicalField
        shape(grid) = physical ? physicalsize(grid, NotPadded()) : spectralsize(grid, NotPadded())
        T = physical ? Float64 : ComplexF64
        dest = constructor(zeros(T, shape(g)), g)
        source = constructor(reshape(T.(1:9), shape(profile_grid)), profile_grid)
        dest .= 2 .* source .+ 1
        expected = zeros(T, size(dest))
        expected .= 2 .* parent(source) .+ 1
        @test parent(dest) == expected
        # Self-reference must retain ordinary fused, in-place semantics.
        dest .= 3 .* dest .- source
        @test parent(dest) == 3 .* expected .- parent(source)
        # Reject incompatible axes before entering an unchecked element loop.
        bad = constructor(zeros(T, shape(bad_grid)), bad_grid)
        @test_throws DimensionMismatch dest .= bad
    end
end
