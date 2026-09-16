@testset "Analytic Fourier-Chebyshev transforms" begin
    # Exercise all even/odd axis combinations. Unequal lengths and resolutions
    # expose swapped axes; the polynomial and Fourier modes are exactly
    # representable, so only floating-point error is expected.
    for (Nx, Nz) in ((6, 8), (7, 9), (6, 9), (7, 8))
        g = Grid(Nx, 9, Nz, 5.3, 7.1)
        a, b = 2π/5.3, 2π/7.1
        f(x, y, z) = 2 + 0.3y + (2y^2-1)*cos(a*x)*sin(b*z)
        u = sampled(g, f)
        original = copy(parent(u))
        U = spectral(g, fzero)
        fft = ForwardFFT!(u; flags=FFTW.ESTIMATE)
        @test fft(U, u) === U
        # Exact coefficient amplitudes fix normalisation, axis order and signs.
        # The y factors are T_0, T_1 and T_2. A cosine contributes 1/2 at
        # kx=+1; sine contributes -i/2 at kz=+1 and +i/2 at kz=-1. Their
        # products are -i/4 and +i/4 at row 3, x slot 2, and z slots 2 and
        # end. This analytic oracle prevents matching forward/inverse scaling
        # errors from hiding in a round trip.
        exact = zeros(ComplexF64, size(U))
        exact[1, 1, 1], exact[2, 1, 1] = 2, 0.3
        exact[3, 2, 2], exact[3, 2, end] = -0.25im, 0.25im
        @test parent(U) ≈ exact atol=2e-13
        # The forward transform must preserve its physical input. The inverse
        # must return the provided destination, recover the original samples
        # and preserve the resolved spectral input despite using internal work
        # arrays.
        @test parent(u) == original
        saved = copy(parent(U))
        ifft = InverseFFT!(U; flags=FFTW.ESTIMATE)
        @test ifft(u, U) === u
        @test parent(u) ≈ original atol=2e-13
        @test parent(U) == saved
        # Reuse the same plans with a different signal to detect stale padding.
        fill!(u, 3)
        fft(U, u)
        exact .= 0
        exact[1, 1, 1] = 3
        @test parent(U) ≈ exact atol=2e-13
        # These plans operate on padded physical storage. An ordinary unpadded
        # PhysicalField must be rejected, rather than silently transformed
        # with incompatible dimensions.
        @test_throws DimensionMismatch fft(U, PhysicalField(g))
        @test_throws DimensionMismatch ifft(PhysicalField(g), U)
    end

    g = Grid(6, 9, 8, 2π, 2π)
    # For Nx=6 and Nz=8 these are exactly the two resolved Nyquist
    # frequencies. The solver convention removes both planes, so no
    # coefficient should survive; this checks filtering separately from the
    # low-mode round trip.
    U = spectral(g, (x, y, z) -> cos(3x) + cos(4z))
    @test norm(parent(U), Inf) < 2e-13  # resolved Nyquist planes are excluded
end

@testset "Quadratic product with 3/2 padding" begin
    # Sample the analytic quadratic product on the padded mesh before
    # truncation. Since cos(3x)^2=(1+cos(6x))/2, only the mean 1/2 is
    # retained. Without padding, mode 6 would alias into a low mode on these
    # grids. Testing x and z separately covers both the reduced and full
    # Fourier axes.
    for N in (7, 8)
        g = Grid(N, 9, N, 2π, 2π)
        # cos(3x)^2 has only a mean and mode 6. Truncation must discard
        # mode 6 without folding it back onto a retained lower mode.
        for f in ((x,y,z) -> cos(3x)^2, (x,y,z) -> cos(3z)^2)
            U = spectral(g, f)
            exact = zeros(ComplexF64, size(U))
            exact[1,1,1] = 0.5
            @test parent(U) ≈ exact atol=2e-13
        end
    end
end

@testset "Highest Chebyshev coefficient with Fourier padding" begin
    # Exercise the degree used by the DNS benchmark, including DCT endpoint
    # weights. The highest Chebyshev polynomial and a retained Fourier mode
    # have known coefficients; truncating Fourier columns before the DCT
    # must preserve both, for even and odd padded periodic lengths.
    for N in (7, 8)
        g = Grid(N, 35, N, 2π, 2π)
        f(x, y, z) = cos(34acos(clamp(y, -1, 1)))*cos(2x)*cos(2z)
        u = sampled(g, f)
        U = spectral(g, fzero)
        fft = ForwardFFT!(u; flags=FFTW.ESTIMATE)
        ifft = InverseFFT!(U; flags=FFTW.ESTIMATE)
        fft(U, u)
        expected = zeros(ComplexF64, size(U))
        expected[35, 3, 3] = expected[35, 3, end-1] = 0.25
        @test parent(U) ≈ expected atol=2e-12
        original = copy(parent(u))
        ifft(u, U)
        @test parent(u) ≈ original atol=2e-12
    end
end
