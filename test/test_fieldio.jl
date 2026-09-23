@testset "Field and state files" begin
    # Exercise physical/resolved/padded storage and all supported containers.
    # Exact equality is required: saving must not transform or truncate data.
    g = Grid(6, 7, 8, 5.0, 3.0)
    mktempdir() do dir
        for scalar in (PhysicalField(g), PhysicalField(g, Padded()),
                       SpectralField(g),
                       SpectralField(zeros(ComplexF64, spectralsize(g, Padded())), g))
            randn!(parent(scalar))
            for object in (scalar, VectorField(copy(scalar), copy(scalar), copy(scalar)),
                           CF.GradientField(scalar))
                path = joinpath(dir, "field.bin")
                @test savefield(path, object) == path
                restored = loadfield(path)
                @test typeof(restored) == typeof(object)
                fields(x::Union{PhysicalField, SpectralField}) = (x,)
                fields(x::VectorField) = x.components
                fields(x::CF.GradientField) = Tuple(x[i,j] for i=1:3 for j=1:3)
                for (a, b) in zip(fields(object), fields(restored))
                    @test parent(a) == parent(b)
                    @test parent(a) !== parent(b)
                    @test CF.grid(b).physicalsize == g.physicalsize
                    @test CF.grid(b).domainsize == g.domainsize
                end
                @test all(CF.grid(b) === CF.grid(first(fields(restored))) for b in fields(restored))
            end
        end
        # Pressure and velocity share the same reconstructed grid, essential
        # for creating a new problem from the loaded state.
        state = zero_state(g)
        for u in velocity(state)
            randn!(parent(u))
        end
        randn!(parent(stagepressure(state)))
        savefield(joinpath(dir, "state.bin"), state)
        restored = loadfield(joinpath(dir, "state.bin"))
        @test all(parent(a) == parent(b) for (a,b) in zip(velocity(state), velocity(restored)))
        @test parent(stagepressure(state)) == parent(stagepressure(restored))
        @test CF.grid(stagepressure(restored)) === CF.grid(velocity(restored)[1])
        @test CF.grid(velocity(restored)[1]) !== g
    end
end

@testset "Legacy coefficient-first field files" begin
    # The old format had no envelope. Its spectral arrays can be deserialized
    # with the existing struct, then explicitly converted once at load time.
    g = Grid(6,9,8,5.0,3.0)
    data = randn(ComplexF64,9,4,8)
    legacy = SpectralField(data,g)
    mktempdir() do dir
        path = joinpath(dir,"old.bin")
        CF.Serialization.serialize(path,State(VectorField(legacy,copy(legacy),copy(legacy)),copy(legacy)))
        state = loadfield(path)
        @test parent(velocity(state)[1]) == permutedims(data,(2,3,1))
        @test size(stagepressure(state)) == spectralsize(g,NotPadded())
        @test CF.grid(stagepressure(state)) === CF.grid(velocity(state)[1])
    end
end
