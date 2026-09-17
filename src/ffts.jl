export ForwardFFT!, InverseFFT!, ForwardFFT, InverseFFT, FFT, IFFT

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
             plan::P  # padded physical grid to Fourier space
        chebyplan::C  # dense or FFTW wall-normal transform
           padded::A  # complete padded spectrum produced by the plan
         resolved::A  # Fourier-truncated input to the Chebyshev transform
    normalization::T  # inverse periodic size and Chebyshev degree
end

function ForwardFFT!(        u::PhysicalField{T};
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
    # The wall-normal backend works only on retained Fourier columns.
    # It writes the final coefficients into the caller's output field.
    resolved = SpectralField(zeros(Complex{T}, spectralsize(grid(u), NotPadded())), grid(u))
    chebyplan = plan_cheb(resolved; flags=flags, timelimit=timelimit)
    return ForwardFFT!(plan, chebyplan, padded, resolved,
                       inv(T(Nxp * Nzp * (Ny-1))))
end

"""
    ForwardFFT(grid::Grid, [T=Float64]; kwargs...)

Construct a forward transform plan directly from `grid`. Call the returned
plan as `fft(U, u)`, with resolved spectral output and padded physical input.
`T` is the real element type; forward planning keywords to [`ForwardFFT!`](@ref).
"""
function ForwardFFT(grid::Grid, ::Type{T}=Float64; kwargs...) where {T<:AbstractFloat}
    u = PhysicalField(zeros(T, physicalsize(grid, Padded())), grid)
    return ForwardFFT!(u; kwargs...)
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
    # Fourier truncation commutes with the wall-normal transform. Discard
    # unresolved columns first, so the DCT only processes retained modes.
    copy_from_padded!(fft.resolved, fft.padded)
    LinearAlgebra.mul!(U, fft.chebyplan, fft.resolved)

    # Apply Fourier/Chebyshev normalisation, endpoint weights and Nyquist
    # filtering in one traversal of the resolved buffer.
    normalize_forward!(U, fft.normalization)
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
by evaluating the wall-normal transform into a separate resolved buffer,
then embedding those values into the padded Fourier spectrum. FFTW's `brfft`
may overwrite that padded buffer. This order skips DCTs of zero columns.
"""
struct InverseFFT!{P, C, A}
         plan::P  # padded spectrum to the padded physical grid
    chebyplan::C  # dense or FFTW wall-normal transform
       padded::A  # zero-padded spectrum used as destructive brfft input
     resolved::A  # retained columns for the wall-normal transform
end

function InverseFFT!(        U::SpectralField{T};
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
    resolved = SpectralField(zeros(eltype(U), spectralsize(grid(U), NotPadded())), grid(U))
    chebyplan = plan_icheb(resolved;
                                    flags=flags, timelimit=timelimit)
    return InverseFFT!(plan, chebyplan, padded, resolved)
end

"""
    InverseFFT(grid::Grid, [T=Float64]; kwargs...)

Construct an inverse transform plan directly from `grid`. Call the returned
plan as `ifft(u, U)`, with padded physical output and resolved spectral input.
`T` is the real element type; forward planning keywords to [`InverseFFT!`](@ref).
"""
function InverseFFT(grid::Grid, ::Type{T}=Float64; kwargs...) where {T<:AbstractFloat}
    U = SpectralField(zeros(Complex{T}, spectralsize(grid, NotPadded())), grid)
    return InverseFFT!(U; kwargs...)
end

"""Transform `U` into the padded physical array `u` and return `u`."""
function (ifft::InverseFFT!)(u::PhysicalField, U::SpectralField)
    size(u) == physicalsize(grid(ifft.padded), Padded()) ||
        throw(DimensionMismatch("inverse transform requires padded physical output"))
    size(U) == spectralsize(grid(ifft.padded), NotPadded()) ||
        throw(DimensionMismatch("inverse transform requires resolved spectral input"))
    # Transform only retained Fourier columns. Padding commutes with this
    # DCT; transforming zero columns on the padded grid wastes most of its work.
    LinearAlgebra.mul!(ifft.resolved, ifft.chebyplan, U)

    # Preserve U and provide a disposable, zero-padded buffer to brfft.
    fill!(ifft.padded, zero(eltype(ifft.padded)))
    copy_to_padded!(ifft.padded, ifft.resolved, 0.5)
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

#//////////////////////////////////////////////////////////////////////////////#
#///                         ALLOCATING TRANSFORMS                          ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    FFT(u::PhysicalField; flags=FFTW.ESTIMATE, kwargs...)
    FFT(u::VectorField{<:PhysicalField}; flags=FFTW.ESTIMATE, kwargs...)

Allocate and return the resolved Fourier-Chebyshev spectrum of padded physical
input `u`, preserving its values and grid. Follow the same padding and Nyquist
conventions as [`ForwardFFT!`](@ref). Vector components share one transform plan.
Forward planning keywords to `ForwardFFT!`; use cached plans for repeated calls.
"""
function FFT(u::PhysicalField{T}; flags=FFTW.ESTIMATE, kwargs...) where {T}
    U = SpectralField(grid(u), T)
    return ForwardFFT!(u; flags=flags, kwargs...)(U, u)
end

function FFT(u::VectorField{F}; flags=FFTW.ESTIMATE, kwargs...) where {T, F<:PhysicalField{T}}
    g = grid(u[1])
    U = VectorField(SpectralField(g, T))
    return ForwardFFT!(u[1]; flags=flags, kwargs...)(U, u)
end

"""
    IFFT(U::SpectralField; flags=FFTW.ESTIMATE, kwargs...)
    IFFT(U::VectorField{<:SpectralField}; flags=FFTW.ESTIMATE, kwargs...)

Allocate and return the padded physical field reconstructed from resolved
spectrum `U`, preserving its values and grid. Vector components share one plan.
Forward planning keywords to [`InverseFFT!`](@ref). The output is suitable for
`FFT`; excluded Nyquist modes are not reconstructed.
"""
function IFFT(U::SpectralField{T}; flags=FFTW.ESTIMATE, kwargs...) where {T}
    u = PhysicalField(zeros(T, physicalsize(grid(U), Padded())), grid(U))
    return InverseFFT!(U; flags=flags, kwargs...)(u, U)
end

function IFFT(U::VectorField{F}; flags=FFTW.ESTIMATE, kwargs...) where {T, F<:SpectralField{T}}
    g = grid(U[1])
    u = VectorField(PhysicalField(zeros(T, physicalsize(g, Padded())), g))
    return InverseFFT!(U[1]; flags=flags, kwargs...)(u, U)
end

#//////////////////////////////////////////////////////////////////////////////#
#///                     FOURIER PADDING AND TRUNCATION                     ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    copy_to_padded!(dest, src, scale=1)

Embed a compact resolved spectrum in a zeroed padded spectrum. The stored
nonnegative `kx` modes remain a prefix of dimension two. Nonnegative `kz`
modes stay at the start of dimension three; negative modes move to its end.
Multiply copied coefficients by `scale`; entries outside the copied blocks
are left untouched, so the caller must zero `dest` before padding.
"""
function copy_to_padded!(dest::SpectralField{T},
                          src::SpectralField{T},
                         scale=one(T)) where {T}
    _, Nxh, Nz = size(src)
    positive, negative, padded_negative = _complex_mode_ranges(Nz, size(dest, 3))

    @views parent(dest)[:, 1:Nxh, positive] .= scale .* parent(src)[:, :, positive]
    @views parent(dest)[:, 1:Nxh, padded_negative] .= scale .* parent(src)[:, :, negative]
    return dest
end

"""Apply transform scaling, Chebyshev endpoint weights and Nyquist filtering."""
function normalize_forward!(U::SpectralField, normalization)
    Ny, Nxh, Nz = size(U)
    _, Nx, _ = physicalsize(grid(U), NotPadded())
    xnyquist = iseven(Nx) ? (Nx >> 1) + 1 : 0
    znyquist = iseven(Nz) ? (Nz >> 1) + 1 : 0

    @inbounds for iz = 1:Nz, ix = 1:Nxh
        if ix == xnyquist || iz == znyquist
            for iy = 1:Ny
                U[iy, ix, iz] = 0
            end
        else
            for iy = 1:Ny
                endpoint = (iy == 1 || iy == Ny) ? 0.5 : 1.0
                U[iy, ix, iz] *= normalization*endpoint
            end
        end
    end
    return U
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
