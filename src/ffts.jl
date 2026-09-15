import FFTW

export ForwardFFT!, InverseFFT!

# Array storage is `(y,x,z)`. FFTW applies the real transform along the first
# entry and complex transforms along the remaining entries. Using `(2,3)`
# therefore gives an rfft in x and a full FFT in z, while leaving y untouched.
const FFT_DIMS = (2, 3)

"""
    ForwardFFT!(u; flags=FFTW.EXHAUSTIVE, timelimit=FFTW.NO_TIMELIMIT)

Plan the Fourier transform from physical storage `(y, x, z)` to spectral
storage `(y, kx, kz)`. The `x` direction is stored as a real-transform
half-spectrum. Fourier coefficients are normalised by the padded size.
"""
struct ForwardFFT!{P, A, T}
    plan::P          # plan from the padded physical grid to Fourier space
    padded::A        # complete padded spectrum produced by the plan
    normalization::T # inverse number of padded x-z points
end

function ForwardFFT!(u::PhysicalField{T};
                     flags::Integer=FFTW.EXHAUSTIVE,
                     timelimit::Real=FFTW.NO_TIMELIMIT) where {T}
    # Planning uses the 3/2-padded periodic dimensions. The wall-normal
    # direction is neither transformed nor padded.
    Ny, Nxp, Nzp = physicalsize(grid(u), Padded())
    padded = SpectralField(
        zeros(Complex{T}, spectralsize(grid(u), Padded())), grid(u))

    # Execution later uses any real array with this type and layout.
    plan = FFTW.plan_rfft(zeros(T, Ny, Nxp, Nzp), FFT_DIMS;
                          flags=flags, timelimit=timelimit)
    return ForwardFFT!(plan, padded, inv(T(Nxp * Nzp)))
end

"""Transform the padded physical array `u` and retain the resolved modes in `U`."""
function (fft::ForwardFFT!)(U::SpectralField, u::PhysicalField)
    # The nonlinear product is sampled on the padded grid. Transform it into
    # the internal padded spectrum before discarding unresolved modes.
    FFTW.unsafe_execute!(fft.plan, parent(u), parent(fft.padded))

    # FFTW leaves forward transforms unnormalised. With this convention the
    # stored coefficients are Fourier-series amplitudes and brfft needs no
    # additional scaling.
    fft.padded .*= fft.normalization
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

"""
    InverseFFT!(U; flags=FFTW.EXHAUSTIVE, timelimit=FFTW.NO_TIMELIMIT)

Plan the inverse transform from resolved spectral storage `(y, kx, kz)` to the
3/2-padded physical storage `(y, xp, zp)`. The resolved input is preserved
because the coefficients are first embedded in the internal padded buffer;
FFTW's `brfft` is then allowed to overwrite that buffer.
"""
struct InverseFFT!{P, A}
    plan::P   # plan from the padded spectrum to the padded physical grid
    padded::A # zero-padded spectrum used as destructive brfft input
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
    return InverseFFT!(plan, padded)
end

"""Transform `U` into the padded physical array `u` and return `u`."""
function (ifft::InverseFFT!)(u::PhysicalField, U::SpectralField)
    # Clear all unresolved modes, then insert the compact resolved spectrum.
    # Copying also protects U from the destructive brfft implementation.
    fill!(ifft.padded, zero(eltype(ifft.padded)))
    copy_to_padded!(ifft.padded, U)
    zero_nyquist!(ifft.padded)
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
