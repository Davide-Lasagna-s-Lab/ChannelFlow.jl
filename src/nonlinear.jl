export NonLinearTerm, RotatingForm

#//////////////////////////////////////////////////////////////////////////////#
#///                          NONLINEAR FORM TAGS                           ///#
#//////////////////////////////////////////////////////////////////////////////#

# Abstract type representing the form used to calculate the nonlinear term.
abstract type NonlinearityForm end

"""Negative advection: `-(u ⋅ ∇)u`."""
struct ConvectiveForm  <: NonlinearityForm end

"""Negative flux divergence: `-∇ ⋅ (u ⊗ u)`."""
struct DivergenceForm  <: NonlinearityForm end

"""Rotational acceleration `u × curl(u)` with modified pressure (default)."""
struct RotatingForm    <: NonlinearityForm end


#//////////////////////////////////////////////////////////////////////////////#
#///                    NONLINEAR OPERATOR CONSTRUCTION                     ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    NonLinearTerm(u, U, baseflow;
                  chebbackend=:fftw, fftwflags=FFTW.EXHAUSTIVE,
                  fftwtimelimit=FFTW.NO_TIMELIMIT,
                  form=RotatingForm())

Construct a pseudo-spectral convection operator using scalar physical/spectral
fields `u` and `U` as allocation and transform prototypes.

The Chebyshev coefficients of the stationary streamwise profile `Ub(y)` are
passed explicitly as `baseflow`. The evaluation
`Eq(t, Upert, rhs)` accepts spectral vector fields and computes the selected
nonlinear form using `utotal = upert + Ub(y) e_x`. The rotational form returns
`FFT(utotal × curl(utotal))`; the other forms return negative advection.

Call `Eq(t, Upert, rhs)` to overwrite `rhs`, or pass `true` as the fourth argument to
accumulate. Input velocity is preserved. `t` supports the Flows interface;
the stationary base flow has no explicit time dependence. The instance owns
mutable workspaces and must not be called concurrently.

For this parallel reference convective self-advection vanishes. The rotational
form instead contributes a pressure gradient that is absorbed into its modified
pressure variable. The sustaining viscous and pressure-gradient balance,
additional forcing and pressure projection are handled outside this operator.
"""
struct NonLinearTerm{T, FORM<:NonlinearityForm, CACHE, IFFT, FFT, B<:AbstractVector{T}}
       cache::CACHE # cache specific to the selected nonlinear form
        ifft::IFFT  # concrete callable inverse transform
         fft::FFT   # concrete callable forward transform
    baseflow::B     # Chebyshev coefficients of the stationary profile

    function NonLinearTerm(             u::PhysicalField{T},
                                       U::SpectralField{T},
                                baseflow::AbstractVector{T};
                             chebbackend::Symbol=:fftw,
                               fftwflags::Integer=FFTW.EXHAUSTIVE,
                           fftwtimelimit::Real=FFTW.NO_TIMELIMIT,
                                    form::NonlinearityForm=RotatingForm()) where {T}
        cache = _gencache(form, u, U)
        ifft = InverseFFT!(U; chebbackend=chebbackend, flags=fftwflags, timelimit=fftwtimelimit)
        fft  = ForwardFFT!(u; chebbackend=chebbackend, flags=fftwflags, timelimit=fftwtimelimit)
        return new{T,
                   typeof(form),
                   typeof(cache),
                   typeof(ifft),
                   typeof(fft),
                   typeof(baseflow)}(cache, ifft, fft, baseflow)
    end
end

#//////////////////////////////////////////////////////////////////////////////#
#///                          FORM-SPECIFIC CACHES                          ///#
#//////////////////////////////////////////////////////////////////////////////#

function _gencache( ::ConvectiveForm,
                   u::PhysicalField{T},
                   U::SpectralField) where {T}
    padded = PhysicalField(zeros(T, physicalsize(grid(u), Padded())), grid(u))
    return (VectorField(padded),
            VectorField(padded),
            GradientField(padded),
            VectorField(U),
            GradientField(U))
end

function _gencache( ::DivergenceForm,
                   u::PhysicalField{T},
                   U::SpectralField) where {T}
    padded = PhysicalField(zeros(T, physicalsize(grid(u), Padded())), grid(u))
    return (VectorField(padded),
            GradientField(padded),
            VectorField(U),
            GradientField(U))
end

function _gencache( ::RotatingForm,
                   u::PhysicalField{T},
                   U::SpectralField) where {T}
    padded = PhysicalField(zeros(T, physicalsize(grid(u), Padded())), grid(u))
    return (VectorField(padded),
            VectorField(padded),
            VectorField(padded),
            VectorField(U),
            VectorField(U))
end

#//////////////////////////////////////////////////////////////////////////////#
#///                            CONVECTIVE FORM                             ///#
#//////////////////////////////////////////////////////////////////////////////#

function (Eq::NonLinearTerm{T, ConvectiveForm})(   t::Real,
                                                   U::VectorField{S},
                                                dUdt::VectorField{S},
                                                 add::Bool=false) where {T, S<:SpectralField{T}}
    u, n, grad, TMP, GRAD = Eq.cache

    # TMP initially holds the TOTAL spectral velocity. Add the reference's
    # Chebyshev coefficients to the zero Fourier mode here,
    # before either gradient evaluation or inverse transformation. Adding it
    # only to the physical advecting velocity would omit the v*Ub' shear term.
    TMP .= U
    @views TMP[1][1, 1, :] .+= Eq.baseflow

    grad!(GRAD, TMP)

    # Evaluate velocity and all derivatives on the padded physical grid.
    Eq.ifft(u, TMP)
    Eq.ifft(grad, GRAD)

    dot!(n, u, grad)

    # dot! produces positive advection. Transform it, then apply the minus
    # sign required by the momentum RHS. TMP can now be reused: total velocity
    # is already consumed.
    Eq.fft(TMP, n)

    return _store_rhs!(dUdt, TMP, add, -1)
end

#//////////////////////////////////////////////////////////////////////////////#
#///                            DIVERGENCE FORM                             ///#
#//////////////////////////////////////////////////////////////////////////////#

function (Eq::NonLinearTerm{T, DivergenceForm})(   t::Real,
                                                   U::VectorField{S},
                                                dUdt::VectorField{S},
                                                 add::Bool=false) where {T, S<:SpectralField{T}}
    u, uu, N, UU = Eq.cache

    # Form the total velocity, including the base profile in the zero mode.
    N .= U
    @views N[1][1, 1, :] .+= Eq.baseflow

    Eq.ifft(u, N)

    outer!(uu, u, u)

    Eq.fft(UU, uu)

    # Differentiate the transformed flux tensor in coefficient space.
    div!(N, UU)

    # The nonlinear contribution enters the momentum equation with minus sign.
    return _store_rhs!(dUdt, N, add, -1)
end

#//////////////////////////////////////////////////////////////////////////////#
#///                             ROTATING FORM                              ///#
#//////////////////////////////////////////////////////////////////////////////#

function (Eq::NonLinearTerm{T, RotatingForm})(   t::Real,
                                                 U::VectorField{S},
                                              dUdt::VectorField{S},
                                               add::Bool=false) where {T, S<:SpectralField{T}}
    u, n, ω, TMP, Ω = Eq.cache

    TMP .= U
    @views TMP[1][1, 1, :] .+= Eq.baseflow

    curl!(Ω, TMP)

    Eq.ifft(u, TMP)
    Eq.ifft(ω, Ω)

    n[1] .= u[2] .* ω[3] .- u[3] .* ω[2]
    n[2] .= u[3] .* ω[1] .- u[1] .* ω[3]
    n[3] .= u[1] .* ω[2] .- u[2] .* ω[1]

    Eq.fft(TMP, n)
    return _store_rhs!(dUdt, TMP, add, 1)
end

#//////////////////////////////////////////////////////////////////////////////#
#///                      SIGNED MOMENTUM CONTRIBUTION                      ///#
#//////////////////////////////////////////////////////////////////////////////#

"""Write or accumulate the signed spectral vector contribution in `dUdt`."""
function _store_rhs!(dUdt::VectorField{S},
                        N::VectorField{S},
                      add::Bool,
                     sign::Int) where {S<:SpectralField}
    for i = 1:3
        add ? (dUdt[i] .+= sign .* N[i]) : (dUdt[i] .= sign .* N[i])
    end
    return dUdt
end
