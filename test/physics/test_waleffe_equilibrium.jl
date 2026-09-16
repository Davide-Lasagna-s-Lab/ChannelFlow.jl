#//////////////////////////////////////////////////////////////////////////////#
#///                      WALEFFE EQUILIBRIUM IMPORT                        ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    waleffe_equilibrium(branch, Ny, N)

Read the archived R=400, alpha=1.14, gamma=2.5 Couette equilibrium.
Validate published mean/rms profiles, then convert total velocity to the
DNS perturbation convention. Requires `tar` with gzip support; no download
occurs during tests. Original archives and provenance are in `data/`.
"""
function waleffe_equilibrium(branch, Ny, N)
    name = "RRC.a1.14.g2.5.R400.$branch"
    archive = joinpath(@__DIR__, "data", "$name.tar.gz")
    contents(member) = read(`tar -xOf $archive $name.$member`, String)
    # The stats file has 34 Gauss points and columns y,U,urms,vrms,wrms.
    lines = filter(line -> !startswith(strip(line), "%"),
                   split(contents("stats"), '\n'))
    stats = transpose(reshape(parse.(Float64, split(join(lines," "))),5,34))
    y = cospi.(((0:33).+0.5)/34)
    C = [cospi((i+0.5)*j/34) for i=0:33, j=0:33]
    @test maximum(abs, stats[:,1]-y) < 5e-9
    g = Grid(N, Ny, N,2π/1.14,2π/2.5)
    state = zero_state(g)
    for (i, component) in enumerate(("u","v","w"))
        # Fortran order is (x,z,Chebyshev degree), with x varying fastest.
        # There is no DCT here: the y entries are already ordinary Chebyshev
        # coefficients, not samples on either a Gauss or a Lobatto mesh.
        data = parse.(Float64,split(contents("$component.asc")))
        @test length(data) == 32*32*34
        raw = permutedims(reshape(data,32,32,34),(3,1,2))
        values = C*reshape(raw,34,:)
        means = sum(values; dims=2)/1024
        rms = sqrt.(sum(abs2,values .- means; dims=2)/1024)
        i == 1 && (@test maximum(abs,vec(means)-stats[:,2]) < 5e-8)
        @test maximum(abs,vec(rms)-stats[:,i+2]) < 5e-8

        # Fourier-transform only x,z, normalize by the source mesh size,
        # and zero-extend the retained modes onto the requested DNS grid.
        # Ny=35 is the smallest odd size that retains all 34 coefficients.
        spectrum = FFTW.rfft(raw,(2,3))/1024
        @test maximum(abs,spectrum[:,17,:]) < 1e-12
        @test maximum(abs,spectrum[:,:,17]) < 1e-12
        for kz=-15:15, kx=0:15
            parent(velocity(state)[i])[1:34,kx+1,mod(kz,N)+1] .=
                spectrum[:,kx+1,mod(kz,32)+1]
        end
    end
    # The file contains total Couette velocity. Subtract y=T_1 only from
    # the streamwise mean Fourier mode; neither transverse field is shifted.
    parent(velocity(state)[1])[2,1,1] -= 1
    return state
end

#//////////////////////////////////////////////////////////////////////////////#
#///                       VOLUME AND WALL DIAGNOSTICS                      ///#
#//////////////////////////////////////////////////////////////////////////////#

"""Volume energy, total viscous dissipation and moving-wall power input."""
function equilibrium_diagnostics(U)
    g = CF.grid(U[1])
    Ny,Nx,Nz = physicalsize(g,Padded())
    # Unweighted Chebyshev quadrature, normalized by volume 2*Lx*Lz.
    C = [cospi(i*j/(Ny-1)) for i=0:Ny-1, j=0:Ny-1]
    weights = transpose(C) \ [iseven(j) ? 2/(1-j^2) : 0.0 for j=0:Ny-1]
    average(a) = sum(weights .* vec(sum(a; dims=(2,3))))/(2Nx*Nz)
    energy = sum(average(abs2.(physical_values(u))) for u in U.components)/2
    dissipation, input = 0.0, 0.0
    work = similar(U[1])
    for i=1:3, (j, derivative!) in enumerate((ddx1!,ddx2!,ddx3!))
        derivative!(work,U[i])
        d = physical_values(work)
        # The dissipation and wall power concern total velocity; add the
        # derivative of the base Couette profile only to du/dy.
        if i == 1 && j == 2
            d .+= 1
            input = (sum(d[1,:,:])+sum(d[end,:,:]))/(800Nx*Nz)
        end
        dissipation += average(abs2.(d))/400
    end
    return (; energy, dissipation, input)
end

@testset "Waleffe R400 Couette equilibria" begin
    for branch in ("LB","UB")
        @testset "$branch" begin
            for (Ny,N) in ((35,32),(49,48))
                @testset "Ny=$Ny, Nx=Nz=$N" begin
                    initial = waleffe_equilibrium(branch,Ny,N)
                    U = velocity(initial)
                    g = CF.grid(U[1])
                    check_constraints(U)
                    baseline = equilibrium_diagnostics(U)
                    # Nonzero perturbation energy rules out accidental loading
                    # of just the laminar profile. Wall power must balance
                    # total dissipation approximately for a steady Couette ECS.
                    @test baseline.energy > 1e-3
                    @test baseline.dissipation > 1/400
                    @test abs(baseline.input/baseline.dissipation-1) < 1e-3

                    # No pressure is supplied in the archive. Recover it from
                    # a stationary Stokes solve with the imported nonlinear
                    # source. Keep the ORIGINAL velocity, never replace it with
                    # the solve's output or refine it toward a new equilibrium.
                    problem = ChannelFlowProblem(g,identity,1/400,0.025;
                                                  fftwflags=FFTW.ESTIMATE)
                    source, response = similar(U),similar(U)
                    problem.nlterm(0.0,U,source,false)
                    solve!(StokesSolver(g,1/400,0.0),response,
                           stagepressure(initial),source)
                    response .-= U
                    defect = sqrt(equilibrium_diagnostics(response).energy/baseline.energy)
                    # The archive has finite resolution. Bounds distinguish its
                    # two branches; they are not claims of machine-precision
                    # equilibrium or of continuum accuracy for this solver.
                    bound = branch == "LB" ? 1e-4 : 2e-3
                    @test defect < bound
                    finals = []
                    for dt in (0.025,0.0125)
                        channel = ChannelFlowProblem(g,identity,1/400,dt;
                                                      fftwflags=FFTW.ESTIMATE)
                        state = copy(initial)
                        flow = Flows.flow(channel)
                        for t in (0.25,0.5)
                            flow(state,(t-0.25,t))
                            difference = copy(velocity(state))
                            difference .-= U
                            drift = sqrt(equilibrium_diagnostics(difference).energy/baseline.energy)
                            current = equilibrium_diagnostics(velocity(state))
                            @test drift < bound
                            @test abs(current.energy/baseline.energy-1) < bound
                            @test abs(current.dissipation/baseline.dissipation-1) < bound
                            @test abs(current.input/baseline.input-1) < bound
                            check_constraints(velocity(state))
                            @info "Waleffe equilibrium" branch Ny N dt t defect drift
                        end
                        push!(finals,velocity(state))
                    end
                    # Time refinement compares two evolutions of the SAME
                    # archived state. Its finite spatial residual need not drop
                    # by four when dt is halved, unlike pure temporal error.
                    difference = copy(finals[1])
                    difference .-= finals[2]
                    temporal = sqrt(equilibrium_diagnostics(difference).energy/baseline.energy)
                    @test temporal < 1e-5
                end
            end
        end
    end
end
