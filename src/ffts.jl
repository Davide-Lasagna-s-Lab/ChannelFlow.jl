import FFTW

export ForwardFFT!, InverseFFT!

#//////////////////////////////////////////////////////////////////////////////#
#///                   FOURIER-CHEBYSHEV TRANSFORM LAYOUT                   ///#
#//////////////////////////////////////////////////////////////////////////////#

# Array storage is `(y,x,z)`. FFTW applies the real transform along the first
# entry and complex transforms along the remaining entries. Using `(2,3)`
# therefore gives an rfft in x and a full FFT in z. A separate DCT-I in the
# first dimension converts Lobatto values to/from Chebyshev coefficients.
const FFT_DIMS = (2, 3)

#//////////////////////////////////////////////////////////////////////////////#
#///                           FORWARD TRANSFORM                            ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    ForwardFFT!(u; flags=FFTW.EXHAUSTIVE, timelimit=FFTW.NO_TIMELIMIT)

Plan the Fourier--Chebyshev transform from physical storage `(y, x, z)` to
spectral storage `(n, kx, kz)`. Fourier amplitudes are normalised by the padded
periodic size; Chebyshev coefficients use the ordinary `sum(a_n*T_n)` series.
"""
struct ForwardFFT!{P, C, A, T}
    plan::P          # plan from the padded physical grid to Fourier space
    chebyplan::C     # in-place DCT-I along the wall-normal dimension
    padded::A        # complete padded spectrum produced by the plan
    normalization::T # inverse periodic size and Chebyshev degree
end

function ForwardFFT!(u::PhysicalField{T};
                     flags::Integer=FFTW.EXHAUSTIVE,
                     timelimit::Real=FFTW.NO_TIMELIMIT) where {T}
    # Only periodic dimensions are padded. All Ny Chebyshev coefficients
    # are retained; there is no wall-normal dealiasing in this implementation.
    Ny, Nxp, Nzp = physicalsize(grid(u), Padded())
    padded = SpectralField(
        zeros(Complex{T}, spectralsize(grid(u), Padded())), grid(u))

    # Execution later uses any real array with this type and layout.
    plan = FFTW.plan_rfft(zeros(T, Ny, Nxp, Nzp), FFT_DIMS;
                          flags=flags, timelimit=timelimit)
    # FFTW applies this real-to-real transform to the real and imaginary
    # parts of the complex buffer independently, with no extra workspace.
    chebyplan = FFTW.plan_r2r!(parent(padded), FFTW.REDFT00, (1,);
                              flags=flags, timelimit=timelimit)
    return ForwardFFT!(plan, chebyplan, padded, inv(T(Nxp * Nzp * (Ny-1))))
end

"""Transform the padded physical array `u` and retain the resolved modes in `U`."""
function (fft::ForwardFFT!)(U::SpectralField, u::PhysicalField)
    size(u) == physicalsize(grid(fft.padded), Padded()) ||
        throw(DimensionMismatch("forward transform requires padded physical input"))
    size(U) == spectralsize(grid(fft.padded), NotPadded()) ||
        throw(DimensionMismatch("forward transform requires resolved spectral output"))
    # The nonlinear product is sampled on the padded grid. Transform it into
    # the internal padded spectrum before discarding unresolved modes.
    FFTW.unsafe_execute!(fft.plan, parent(u), parent(fft.padded))
    FFTW.unsafe_execute!(fft.chebyplan, parent(fft.padded), parent(fft.padded))

    # ChebyCoeff::chebyfft in Channelflow divides DCT-I by P = Ny-1 and
    # halves its endpoint coefficients. Combine 1/P with Fourier scaling.
    fft.padded .*= fft.normalization
    @views parent(fft.padded)[1, :, :] ./= 2
    @views parent(fft.padded)[end, :, :] ./= 2
    copy_from_padded!(U, fft.padded)

    # Nyquist modes do not have an unambiguous positive/negative partner and
    # are excluded from derivatives and nonlinear products.
    zero_nyquist!(U)
    return U
end

"""Transform each component of physical vector field `u` into `U`."""
function (fft::ForwardFFT!)(U::VectorField, u::VectorField)
    @inbounds for i = 1:3
        fft(U[i], u[i])
    end
    return U
end

"""Transform every component of physical gradient field `grad` into `GRAD`."""
function (fft::ForwardFFT!)(GRAD::GradientField,
                            grad::GradientField)
    @inbounds for i = 1:3
        fft(GRAD[i], grad[i])
    end
    return GRAD
end

#//////////////////////////////////////////////////////////////////////////////#
#///                           INVERSE TRANSFORM                            ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    InverseFFT!(U; flags=FFTW.EXHAUSTIVE, timelimit=FFTW.NO_TIMELIMIT)

Plan the inverse transform from resolved spectral storage `(n, kx, kz)` to the
3/2-padded physical storage `(y, xp, zp)`. The resolved input is preserved
because the coefficients are first embedded in the internal padded buffer;
FFTW's `brfft` is then allowed to overwrite that buffer.
"""
struct InverseFFT!{P, C, A}
    plan::P      # plan from the padded spectrum to the padded physical grid
    chebyplan::C # in-place DCT-I along the wall-normal dimension
    padded::A    # zero-padded spectrum used as destructive brfft input
end

function InverseFFT!(U::SpectralField{T};
                     flags::Integer=FFTW.EXHAUSTIVE,
                     timelimit::Real=FFTW.NO_TIMELIMIT) where {T}
    # The inverse plan must use exactly the same padded layout as the forward
    # plan. `Nxp` is passed explicitly because an rfft half-spectrum alone
    # cannot distinguish an even physical length from the adjacent odd one.
    _, Nxp, _ = physicalsize(grid(U), Padded())
    padded = SpectralField(
        zeros(eltype(U), spectralsize(grid(U), Padded())), grid(U))
    plan = FFTW.plan_brfft(parent(padded), Nxp, FFT_DIMS;
                           flags=flags, timelimit=timelimit)
    chebyplan = FFTW.plan_r2r!(parent(padded), FFTW.REDFT00, (1,);
                              flags=flags, timelimit=timelimit)
    return InverseFFT!(plan, chebyplan, padded)
end

"""Transform `U` into the padded physical array `u` and return `u`."""
function (ifft::InverseFFT!)(u::PhysicalField, U::SpectralField)
    size(u) == physicalsize(grid(ifft.padded), Padded()) ||
        throw(DimensionMismatch("inverse transform requires padded physical output"))
    size(U) == spectralsize(grid(ifft.padded), NotPadded()) ||
        throw(DimensionMismatch("inverse transform requires resolved spectral input"))
    # Clear all unresolved modes, then insert the compact resolved spectrum.
    # Copying also protects U from the destructive brfft implementation.
    fill!(ifft.padded, zero(eltype(ifft.padded)))
    copy_to_padded!(ifft.padded, U)
    zero_nyquist!(ifft.padded)

    # ChebyCoeff::ichebyfft: undo the endpoint weights, apply raw DCT-I,
    # then divide by two. The Fourier inverse remains unnormalised.
    @views parent(ifft.padded)[1, :, :] .*= 2
    @views parent(ifft.padded)[end, :, :] .*= 2
    FFTW.unsafe_execute!(ifft.chebyplan, parent(ifft.padded), parent(ifft.padded))
    ifft.padded ./= 2
    FFTW.unsafe_execute!(ifft.plan, parent(ifft.padded), parent(u))
    return u
end

"""Transform each component of spectral vector field `U` into `u`."""
function (ifft::InverseFFT!)(u::VectorField, U::VectorField)
    @inbounds for i = 1:3
        ifft(u[i], U[i])
    end
    return u
end

"""Transform every component of spectral gradient field `GRAD` into `grad`."""
function (ifft::InverseFFT!)(grad::GradientField,
                             GRAD::GradientField)
    @inbounds for i = 1:3
        ifft(grad[i], GRAD[i])
    end
    return grad
end

#//////////////////////////////////////////////////////////////////////////////#
#///                     FOURIER PADDING AND TRUNCATION                     ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    copy_to_padded!(dest, src)

Embed a compact resolved spectrum in a zeroed padded spectrum. The stored
nonnegative `kx` modes remain a prefix of dimension two. Nonnegative `kz`
modes stay at the start of dimension three; negative modes move to its end.
"""
function copy_to_padded!(dest::SpectralField{T}, src::SpectralField{T}) where {T}
    _, Nxh, Nz = size(src)
    positive, negative, padded_negative = _complex_mode_ranges(Nz, size(dest, 3))

    @views parent(dest)[:, 1:Nxh, positive] .= parent(src)[:, :, positive]
    @views parent(dest)[:, 1:Nxh, padded_negative] .= parent(src)[:, :, negative]
    return dest
end

"""
    copy_from_padded!(dest, src)

Truncate a padded spectrum to the modes represented by `dest`. This is the
inverse index mapping of [`copy_to_padded!`](@ref).
"""
function copy_from_padded!(dest::SpectralField{T}, src::SpectralField{T}) where {T}
    _, Nxh, Nz = size(dest)
    positive, negative, padded_negative = _complex_mode_ranges(Nz, size(src, 3))

    @views parent(dest)[:, :, positive] .= parent(src)[:, 1:Nxh, positive]
    @views parent(dest)[:, :, negative] .= parent(src)[:, 1:Nxh, padded_negative]
    return dest
end

# Full complex FFT storage keeps nonnegative modes at the beginning of the
# axis and negative modes at its end. Padding only moves the negative block.
function _complex_mode_ranges(N::Integer, Np::Integer)
    positive = 1:((N >> 1) + 1)
    negative = (last(positive) + 1):N
    padded_negative = (Np - length(negative) + 1):Np
    return positive, negative, padded_negative
end

#//////////////////////////////////////////////////////////////////////////////#
#///                         NYQUIST MODE FILTERING                         ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    zero_nyquist!(U)

Set the Fourier Nyquist planes to zero when the corresponding resolved grid
size is even. The indices are shared by compact and padded spectra because both
Nyquist modes belong to the leading nonnegative-frequency blocks.
"""
function zero_nyquist!(U::SpectralField)
    _, Nx, Nz = physicalsize(grid(U), NotPadded())
    data = parent(U)
    iseven(Nx) && (@views data[:, (Nx >> 1) + 1, :] .= 0)
    iseven(Nz) && (@views data[:, :, (Nz >> 1) + 1] .= 0)
    return U
end
