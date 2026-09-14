import FFTW

export ForwardFFT!, InverseFFT!

# Array storage is `(y,x,z)`. FFTW applies the real transform along the first
# entry and complex transforms along the remaining entries. Using `(3,2)`
# therefore gives an rfft in z and a full FFT in x, while leaving y untouched.
const FFT_DIMS = (3, 2)

"""
    ForwardFFT!(u; flags=FFTW.EXHAUSTIVE, timelimit=FFTW.NO_TIMELIMIT)

Plan the Fourier transform from physical storage `(y, x, z)` to spectral
storage `(y, kx, kz)`. The `z` direction is stored as a real-transform
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
    Ny, Nxp, Nzp = paddedsize(grid(u))
    padded = SpectralField(
        zeros(Complex{T}, spectralsize((Ny, Nxp, Nzp), FFT_DIMS)), grid(u))

    # Execution later uses any real array with this type and layout.
    plan = FFTW.plan_rfft(zeros(T, Ny, Nxp, Nzp), FFT_DIMS;
                          flags=flags, timelimit=timelimit)
    return ForwardFFT!(plan, padded, inv(T(Nxp * Nzp)))
end

"""Transform the padded physical array `u` and retain the resolved modes in `U`."""
function (fft::ForwardFFT!)(U::SpectralField, u::AbstractArray{T, 3}) where {T<:AbstractFloat}
    # The nonlinear product is sampled on the padded grid. Transform it into
    # the internal padded spectrum before discarding unresolved modes.
    FFTW.unsafe_execute!(fft.plan, u, parent(fft.padded))

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
function (fft::ForwardFFT!)(U::VectorField, u::NTuple{3, <:AbstractArray})
    @inbounds for i = 1:3
        fft(U[i], u[i])
    end
    return U
end

"""Transform every component of physical gradient field `grad` into `GRAD`."""
function (fft::ForwardFFT!)(GRAD::GradientField,
                            grad::NTuple{3, <:NTuple{3, <:AbstractArray}})
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
    # plan. `Nzp` is passed explicitly because an rfft half-spectrum alone
    # cannot distinguish an even physical length from the adjacent odd one.
    Ny, Nxp, Nzp = paddedsize(grid(U))
    padded = SpectralField(
        zeros(eltype(U), spectralsize((Ny, Nxp, Nzp), FFT_DIMS)), grid(U))
    plan = FFTW.plan_brfft(parent(padded), Nzp, FFT_DIMS;
                           flags=flags, timelimit=timelimit)
    return InverseFFT!(plan, padded)
end

"""Transform `U` into the padded physical array `u` and return `u`."""
function (ifft::InverseFFT!)(u::AbstractArray{T, 3}, U::SpectralField) where {T<:AbstractFloat}
    # Clear all unresolved modes, then insert the compact resolved spectrum.
    # Copying also protects U from the destructive brfft implementation.
    fill!(ifft.padded, zero(eltype(ifft.padded)))
    copy_to_padded!(ifft.padded, U)
    zero_nyquist!(ifft.padded)
    FFTW.unsafe_execute!(ifft.plan, parent(ifft.padded), u)
    return u
end

"""Transform each component of spectral vector field `U` into `u`."""
function (ifft::InverseFFT!)(u::NTuple{3, <:AbstractArray}, U::VectorField)
    @inbounds for i = 1:3
        ifft(u[i], U[i])
    end
    return u
end

"""Transform every component of spectral gradient field `GRAD` into `grad`."""
function (ifft::InverseFFT!)(grad::NTuple{3, <:NTuple{3, <:AbstractArray}},
                             GRAD::GradientField)
    @inbounds for i = 1:3
        ifft(grad[i], GRAD[i])
    end
    return grad
end

"""
    copy_to_padded!(dest, src)

Embed a compact resolved spectrum in a zeroed padded spectrum. Nonnegative
`kx` modes remain at the start of the second dimension; negative modes move to
its end. The stored nonnegative `kz` block remains a prefix of dimension three.
"""
function copy_to_padded!(dest::SpectralField{T},
                         src::SpectralField{T}) where {T}
    _, Nx, Nzh = size(src)
    positive, negative, padded_negative = _xmode_ranges(Nx, size(dest, 2))

    @views parent(dest)[:, positive, 1:Nzh] .= parent(src)[:, positive, :]
    @views parent(dest)[:, padded_negative, 1:Nzh] .= parent(src)[:, negative, :]
    return dest
end

"""
    copy_from_padded!(dest, src)

Truncate a padded spectrum to the modes represented by `dest`. This is the
inverse index mapping of [`copy_to_padded!`](@ref).
"""
function copy_from_padded!(dest::SpectralField{T},
                           src::SpectralField{T}) where {T}
    _, Nx, Nzh = size(dest)
    positive, negative, padded_negative = _xmode_ranges(Nx, size(src, 2))

    @views parent(dest)[:, positive, :] .= parent(src)[:, positive, 1:Nzh]
    @views parent(dest)[:, negative, :] .= parent(src)[:, padded_negative, 1:Nzh]
    return dest
end

# Full complex FFT storage keeps nonnegative x modes at the beginning of the
# axis and negative modes at its end. Padding only moves the negative block.
function _xmode_ranges(Nx::Integer, Nxp::Integer)
    positive = 1:((Nx >> 1) + 1)
    negative = (last(positive) + 1):Nx
    padded_negative = (Nxp - length(negative) + 1):Nxp
    return positive, negative, padded_negative
end

"""
    zero_nyquist!(U)

Set the `x` and `z` Nyquist planes to zero when the corresponding resolved grid
size is even. The indices are shared by compact and padded spectra because both
Nyquist modes belong to the leading nonnegative-frequency blocks.
"""
function zero_nyquist!(U::SpectralField)
    _, Nx, Nz = gridsize(grid(U))
    data = parent(U)
    iseven(Nx) && (@views data[:, (Nx >> 1) + 1, :] .= 0)
    iseven(Nz) && (@views data[:, :, (Nz >> 1) + 1] .= 0)
    return U
end
