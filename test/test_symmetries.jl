@testset "Discrete channel symmetries" begin
    # Test even and odd z grids: signed-wavenumber parity matters for odd Nz.
    for Nz in (8, 9)
        g = Grid(8, 9, Nz, 2π, 2π)
        fs = ((x,y,z) -> (1-y^2)*(sin(x)+y*cos(2z)),
              (x,y,z) -> y*(1-y^2)*cos(x+z),
              (x,y,z) -> (1-y^2)*sin(2x-z))
        U = VectorField(map(f -> spectral(g, f), fs)...)
        for S in (ShiftReflect(), ShiftRotate())
            # Compare with independently sampled coordinate transformations,
            # including vector component signs rather than only involution.
            signs = S isa ShiftReflect ? (1,1,-1) : (-1,-1,1)
            expected = VectorField(ntuple(3) do i
                spectral(g, (x,y,z) -> S isa ShiftReflect ?
                    signs[i]*fs[i](x+π,y,-z) : signs[i]*fs[i](-x+π,-y,z+π))
            end...)
            V = S(U)
            for i in 1:3
                @test parent(V[i]) ≈ parent(expected[i]) atol=2e-14
                @test parent(S(V)[i]) ≈ parent(U[i]) atol=2e-14
            end
            # In-place transformation must not overwrite coefficients that a
            # later reflected pair still needs to read.
            W = copy(U)
            S(W, W)
            @test all(parent(W[i]) ≈ parent(V[i]) for i in 1:3)
        end

        # Both projectors commute. A second projection leaves the state fixed,
        # and the pressure uses the scalar action without velocity signs.
        P = spectral(g, (x,y,z) -> y*cos(x+z))
        state = State(copy(U), copy(P))
        apply!(state, (ShiftReflect(), ShiftRotate()))
        saved = copy(state)
        apply!(state, (ShiftReflect(), ShiftRotate()))
        for S in (ShiftReflect(), ShiftRotate())
            @test all(isapprox(parent(S(velocity(state))[i]), parent(velocity(state)[i]); atol=2e-14) for i in 1:3)
            @test parent(S(stagepressure(state))) ≈ parent(stagepressure(state)) atol=2e-14
        end
        @test all(isapprox(parent(velocity(state)[i]), parent(velocity(saved)[i]); atol=2e-14) for i in 1:3)
        # Compare against the explicit average, including every stored plane.
        # Arbitrary coefficients exercise the zero and Nyquist self-pairs too.
        for S in (ShiftReflect(), ShiftRotate())
            A = copy(U)
            for Ui in A
                parent(Ui) .= randn(ComplexF64, size(Ui))
            end
            expected = copy(A)
            transformed = S(A)
            expected .= (expected .+ transformed) ./ 2
            apply!(A, S)
            @test all(isapprox(parent(A[i]), parent(expected[i]); atol=2e-14) for i in 1:3)
            # Warm the method before measuring: projection needs no heap storage.
            apply!(A, S)
            @test (@allocated apply!(A, S)) == 0
        end
    end
end
