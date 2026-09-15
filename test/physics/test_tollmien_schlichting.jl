#//////////////////////////////////////////////////////////////////////////////#
#///                   INDEPENDENT ORR-SOMMERFELD MODE                      ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    poiseuille_ts_mode(Ny)

Compute the alpha=1, beta=0, Re=8000 unstable mode with a small dense
Chebyshev-Galerkin eigenproblem. This test reference does not call the DNS
operators, Helmholtz solvers or influence matrices. Return the temporal
exponent, velocity and pressure coefficient vectors, and eigenpair residual.
"""
function poiseuille_ts_mode(Ny)
    # A clamped basis imposes v(±1)=v'(±1)=0 identically. These conditions
    # imply no slip for u=i*v' at alpha=1. Columns contain Chebyshev
    # coefficients of T_k-2(k+2)/(k+3)*T_(k+2)+(k+1)/(k+3)*T_(k+4).
    S = zeros(Ny, Ny-4)
    for k=0:Ny-5
        S[k+1,k+1] = 1
        S[k+3,k+1] = -2*(k+2)/(k+3)
        S[k+5,k+1] = (k+1)/(k+3)
    end
    # Independently assemble differentiation from T_n'=n*U_(n-1).
    # The constant coefficient has half the weight of the other entries.
    D = zeros(Ny, Ny)
    for n=1:Ny-1, k=n-1:-2:0
        D[k+1,n+1] = k == 0 ? n : 2n
    end
    y = cospi.(((0:2Ny-1).+0.5)/(2Ny))
    C = [cos(n*acos(x)) for x in y, n=0:Ny-1]
    V, L = C*S, C*(D*D-I)*S
    # For exp(i*x+s*t), Orr-Sommerfeld reads
    # s*L*v = L^2*v/Re - i*Ub*L*v + i*Ub''*v, Ub=1-y^2.
    # Gauss-Chebyshev weights are constant and cancel from the pencil.
    A = transpose(V)*(C*(D*D-I)^2*S/8000 - im.*(1 .- y.^2).*L - 2im.*V)
    B = transpose(V)*L
    modes = eigen(complex.(A), complex.(B))
    target = 0.002664410371 - 0.2470750602im
    index = argmin(abs.(modes.values .- target))
    s, c = modes.values[index], modes.vectors[:,index]
    residual = norm(A*c-s*B*c)/(norm(A*c)+norm(s*B*c))
    v = S*c
    u = im*D*v
    scale = maximum(sqrt.(abs2.(C*u)+abs2.(C*v)))
    u ./= scale
    v ./= scale
    # Multiplication by y in coefficient space: y*T_0=T_1 and
    # y*T_n=(T_(n-1)+T_(n+1))/2. The discarded top tail is resolved by
    # checking the eigenvalue at two wall-normal resolutions below.
    Y = zeros(Ny, Ny)
    Y[2,1] = 1
    for n=1:Ny-1
        Y[n,n+1] = 0.5
        n+2 <= Ny && (Y[n+2,n+1] = 0.5)
    end
    # Recover physical perturbation pressure from streamwise momentum:
    # i*p=-i*Ub*u-v*Ub'+nu*L*u-s*u. It is not the auxiliary pressure
    # returned by a divergence-free projection; we do not project this mode.
    p = (-im*(I-Y*Y)*u + 2Y*v + (D*D-I)*u/8000 - s*u)/im
    return (; s, u, v, p, residual)
end

#//////////////////////////////////////////////////////////////////////////////#
#///                 TOLLMIEN-SCHLICHTING WAVE EVOLUTION                     ///#
#//////////////////////////////////////////////////////////////////////////////#

@testset "Poiseuille Tollmien-Schlichting wave" begin
    # Reference: Mortensen (2017), section 5, equations (63)-(65), table 3:
    # https://arxiv.org/pdf/1701.03787
    # His wall-normal x is our y. With exp(i*(x-omega*t)), s=-i*omega:
    # amplitude grows as exp(real(s)*t), energy as exp(2*real(s)*t).
    # A single harmonic is essential for this eigenmode benchmark.
    reference = 0.002664410371 - 0.2470750602im
    coarse = poiseuille_ts_mode(65)
    mode = poiseuille_ts_mode(81)
    @test abs(mode.s-reference) < 5e-10
    @test abs(coarse.s-mode.s) < 1e-10
    @test mode.residual < 1e-10

    g = Grid(81, 8, 1, 2π, 2π)
    nu, amplitude, T = 1/8000, 1e-7, 50.0
    # Positive kx=1 stores half the complex amplitude of a real wave. The
    # negative harmonic is implicit in the real FFT. All other modes start
    # at zero, but the nonlinear DNS is free to generate them during evolution.
    initial = zero_state(g)
    parent(velocity(initial)[1])[:,2,1] .= amplitude/2 .* mode.u
    parent(velocity(initial)[2])[:,2,1] .= amplitude/2 .* mode.v
    parent(pressure(initial))[:,2,1] .= amplitude/2 .* mode.p
    check_constraints(velocity(initial))

    # Unweighted volume energy from independent Gauss-Legendre quadrature
    # in y and Fourier Parseval weights in x,z. Include every velocity mode,
    # not just the unstable harmonic, so nonlinear contamination is visible.
    nodes, vectors = eigen(SymTridiagonal(zeros(81),
                          [n/sqrt(4n^2-1) for n=1:80]))
    C = [cos(n*acos(y)) for y in nodes, n=0:80]
    weights = 2 .* vectors[1,:].^2
    function energy(U)
        total = 0.0
        for f in U.components, ix=1:size(f,2), iz=1:size(f,3)
            multiplicity = ix == 1 || ix == size(f,2) ? 1 : 2
            total += multiplicity*sum(weights .* abs2.(C*parent(f)[:,ix,iz]))/4
        end
        return total
    end
    E0 = energy(velocity(initial))
    errors = Float64[]
    # Use steps large enough for temporal error to dominate the roundoff and
    # finite-amplitude floor of this very small perturbation.
    for dt in (0.4, 0.2, 0.1)
        @testset "dt=$dt" begin
            problem = ChannelFlowProblem(g, y -> 1-y^2, nu, dt;
                pressuregradient=(-2nu,0.0), fftwflags=FFTW.ESTIMATE)
            state = copy(initial)
            flow = Flows.flow(problem)
            previous, phase = 1.0+0im, 0.0
            for t in 10.0:10.0:T
                flow(state, (t-10,t))
                U = velocity(state)
                # Track complex modal amplitude by projection on the initial
                # v profile. Each 10-unit interval advances less than pi, so
                # incremental phase unwrapping cannot miss a revolution.
                v0 = parent(velocity(initial)[2])[:,2,1]
                modal = dot(v0,parent(U[2])[:,2,1])/dot(v0,v0)
                phase += angle(modal/previous)
                previous = modal
                @test log(abs(modal))/t ≈ real(reference) rtol=0.02
                @test -phase/t ≈ -imag(reference) rtol=2e-3
                @test energy(U)/E0 ≈ exp(2real(reference)*t) rtol=5e-3
                @test maximum(abs,parent(U[3])) < 1e-13
                check_constraints(U)
            end
            # Compare the full coefficient field, including generated modes,
            # with the linear eigenfunction. Nonlinear terms are O(amplitude^2)
            # and therefore small but not identically zero in this benchmark.
            exact = velocity(copy(initial))
            exact .*= exp(reference*T)
            error = sqrt(sum(sum(abs2,parent(velocity(state)[i])-parent(exact[i]))
                             for i=1:3)/sum(sum(abs2,parent(exact[i])) for i=1:3))
            push!(errors,error)
            @test error < 5e-3
            @info "TS benchmark" dt relative_field_error=error energy_gain=energy(velocity(state))/E0
        end
    end
    # Require at least the expected second-order reduction on halving dt.
    # Do not impose an upper bound: cancellation between explicit/implicit
    # errors can yield faster convergence for this particular eigenmode.
    # This is a convergence check for this case, not a general order proof.
    @test errors[1]/errors[2] > 3.5
    @test errors[2]/errors[3] > 3.5
    @test errors[end] < 1e-5
end
