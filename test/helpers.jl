# Construct reference Chebyshev coefficients directly with a DCT-I on Lobatto
# nodes. This bypasses the package Fourier transform and its mode indexing.
# Dividing by P and halving the endpoint coefficients gives f(y) =
# sum(a[n]*T_n(y)), including an unhalved constant term.
function coefficients(f, Ny)
    P = Ny - 1
    a = FFTW.r2r(ComplexF64[f(cospi(n/P)) for n = 0:P], FFTW.REDFT00)/P
    a[1] /= 2
    a[end] /= 2
    return ChebCoeffs(a)
end

# Allocate a separate derivative for residual checks. This helper uses the
# library derivative, so residual checks complement, rather than replace,
# comparisons against explicitly differentiated analytic polynomials.
derivative(a) = diff!(similar(a), a)
# Evaluate the polynomial at the walls without transforming to physical space:
# T_n(1)=1 and T_n(-1)=(-1)^n. ChebCoeffs uses degree-based, zero-origin
# indexing.
wallvalue(a, side) = sum(a[n]*(side == :right ? 1 : (-1)^n) for n = 0:length(a)-1)

# Independent Gauss--Legendre quadrature of the polynomial, rather than
# the coefficient formula used by the solver's bulk constraint.
function bulkmean(a)
    # The Jacobi matrix gives Gauss-Legendre nodes and weights. The square of
    # the first eigenvector component is half the quadrature weight, directly
    # giving the bulk mean (1/2)*integral(f,-1,1). N nodes integrate these
    # degree N-1 polynomials exactly in exact arithmetic.
    N = length(a)
    nodes, vectors = eigen(SymTridiagonal(zeros(N), [n/sqrt(4n^2-1) for n = 1:N-1]))
    return sum(vectors[1, j]^2 * sum(a[n]*cos(n*acos(nodes[j])) for n = 0:N-1)
               for j = 1:N)
end


fzero(x, y, z) = 0.0

# Sample formulas independently of points() and the Fourier mode indexing.
# Build storage in (y,x,z) order while passing analytic functions (x,y,z).
# Periodic coordinates exclude the repeated endpoint; wall-normal coordinates
# include both walls. Using explicit coordinate formulas avoids sharing a
# points() indexing error with its tests.
function sampled(g, f)
    Ny, Nx, Nz = physicalsize(g, Padded())
    Lx, _, Lz = CF.domainsize(g)
    data = Float64[f(Lx*(ix-1)/Nx, cospi((iy-1)/(Ny-1)), Lz*(iz-1)/Nz)
                   for iy=1:Ny, ix=1:Nx, iz=1:Nz]
    return PhysicalField(data, g)
end

spectral(g, f) = spectral(g, parent(sampled(g, f)))

# Convert padded physical samples to the retained spectrum. These helpers use
# the actual FFT implementation, whose normalization and coefficient locations
# are independently checked in test_ffts.jl. ESTIMATE avoids timing-based FFT
# planning in tests.
function spectral(g, data::Array{Float64,3})
    u = PhysicalField(data, g)
    U = SpectralField(zeros(ComplexF64, spectralsize(g, NotPadded())), g)
    ForwardFFT!(u; flags=FFTW.ESTIMATE)(U, u)
    return U
end

# Reconstruct on the padded mesh for pointwise comparisons with analytic
# functions. Allocate a fresh destination so the reference comparisons do not
# depend on a previously populated work buffer.
function physical_values(U)
    g = CF.grid(U)
    u = PhysicalField(zeros(Float64, physicalsize(g, Padded())), g)
    InverseFFT!(U; flags=FFTW.ESTIMATE)(u, U)
    return parent(u)
end

# Check every retained coefficient of divergence and both wall values of all
# three velocity components. Wall sums are independent of inverse transforms.
# Differentiation amplifies coefficient roundoff, hence its tolerance is
# looser than the endpoint tolerance.
function check_constraints(U)
    out = similar(U[1])
    CF.div!(out, U)
    @test norm(parent(out), Inf) < 2e-9
    for field in U.components
        data = parent(field)
        signs = reshape((-1.0) .^ (0:size(data,1)-1), :, 1, 1)
        @test maximum(abs, sum(data; dims=1)) < 2e-10
        @test maximum(abs, sum(data .* signs; dims=1)) < 2e-10
    end
end
