export NonLinearTerm, RotatingForm

#//////////////////////////////////////////////////////////////////////////////#
#///                          NONLINEAR FORM TAGS                           ///#
#//////////////////////////////////////////////////////////////////////////////#

# Abstract type representing the form used to calculate the nonlinear term.
abstract type NonlinearityForm end

struct ConvectiveForm  <: NonlinearityForm end
struct DivergenceForm  <: NonlinearityForm end
struct AlternatingForm <: NonlinearityForm end
struct RotatingForm    <: NonlinearityForm end

#//////////////////////////////////////////////////////////////////////////////#
#///                    NONLINEAR OPERATOR CONSTRUCTION                     ///#
#//////////////////////////////////////////////////////////////////////////////#

# Functor type that evaluates the nonlinear term of the governing equations
# using a pseudo-spectral approach

"""
    NonLinearTerm(u, U, baseflow;
                  fftwflags=FFTW.EXHAUSTIVE,
                  fftwtimelimit=FFTW.NO_TIMELIMIT,
                  form=ConvectiveForm())

Construct a pseudo-spectral convection operator using scalar physical/spectral
fields `u` and `U` as allocation and transform prototypes.

The Chebyshev coefficients of the stationary streamwise profile `Ub(y)` are
passed explicitly as `baseflow`. The evaluation
`Eq(t, Upert, rhs)` accepts spectral vector fields and computes the selected
nonlinear form using `utotal = upert + Ub(y) e_x`. The rotational form returns
`FFT(utotal × curl(utotal))`; the other forms return negative advection.

For this parallel reference convective self-advection vanishes. The rotational
form instead contributes a pressure gradient that is absorbed into its modified
pressure variable. The sustaining viscous and pressure-gradient balance,
additional forcing and pressure projection are handled outside this operator.
"""
struct NonLinearTerm{T, FORM<:NonlinearityForm, CACHE, IFFT, FFT, B}
       cache::CACHE       # cache specific to the selected nonlinear form
        flag::Ref{Bool}   # toggled at every call in the AlternatingForm
        ifft::IFFT        # concrete callable inverse transform
        fft::FFT         # concrete callable forward transform
    baseflow::B          # Chebyshev coefficients of the stationary profile

    function NonLinearTerm(            u::PhysicalField{T},
                                       U::S,
                               baseflow::AbstractVector;
                               fftwflags::Integer=FFTW.EXHAUSTIVE,
                           fftwtimelimit::Real=FFTW.NO_TIMELIMIT,
                                    form::FORM=ConvectiveForm()
                           ) where {T, S<:SpectralField{T}, FORM<:NonlinearityForm}
        cache = _gencache(form, u, U)
        ifft = InverseFFT!(U; flags=fftwflags, timelimit=fftwtimelimit)
        fft = ForwardFFT!(u; flags=fftwflags, timelimit=fftwtimelimit)
        return new{T, FORM, typeof(cache), typeof(ifft), typeof(fft), typeof(baseflow)}(
            cache, Ref(false), ifft, fft, baseflow)
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

_gencache( ::AlternatingForm,
          u::PhysicalField,
          U::SpectralField) =
    _gencache(ConvectiveForm(), u, U)

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

"""
    Eq(t, U, dUdt, add=false)

Evaluate signed convective acceleration from the spectral perturbation velocity
`U`. Overwrite `dUdt` by default; accumulate into it when `add=true`. Internal
workspaces are reused, so one `Eq` instance must not be evaluated concurrently.
`t` is retained for the Flows API.

The gradient and FFT interfaces used below are still under development in
ChannelFlow; this change establishes the vector/base-flow assembly contract.
"""
function (Eq::NonLinearTerm{T, ConvectiveForm})(   t::Real,
                                                   U::VectorField{S},
                                                dUdt::VectorField{S},
                                                 add::Bool=false) where {T, S<:SpectralField{T}}
    return _convectiveform!(Eq, U, dUdt, add)
end

function _convectiveform!(  Eq::NonLinearTerm,
                             U::VectorField{S},
                          dUdt::VectorField{S},
                           add::Bool) where {S<:SpectralField}

    # get aliases
    u, n, grad, TMP, GRAD  = Eq.cache

    # TMP initially holds the TOTAL spectral velocity. Add the reference's
    # Chebyshev coefficients to the zero Fourier mode here,
    # before either gradient evaluation or inverse transformation. Adding it
    # only to the physical advecting velocity would omit the v*Ub' shear term.
    TMP .= U
    @views TMP[1][:, 1, 1] .+= Eq.baseflow

    grad!(GRAD, TMP)

    # compute transforms
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
    return _divergenceform!(Eq, U, dUdt, add)
end

function _divergenceform!(  Eq::NonLinearTerm,
                             U::VectorField{S},
                          dUdt::VectorField{S},
                           add::Bool) where {S<:SpectralField}

    # get aliases
    u = Eq.cache[1]
    uu = Eq.cache[end-2]
    N  = Eq.cache[end-1]
    UU = Eq.cache[end]

    # Form the total velocity, including the base profile in the zero mode.
    N .= U
    @views N[1][:, 1, 1] .+= Eq.baseflow

    # transform to physical space
    Eq.ifft(u, N)

    # calc outer product
    outer!(uu, u, u)

    # transform to spectral space
    Eq.fft(UU, uu)

    # calculate product in physical space
    div!(N, UU)

    # The nonlinear contribution enters the momentum equation with minus sign.
    return _store_rhs!(dUdt, N, add, -1)
end

#//////////////////////////////////////////////////////////////////////////////#
#///                            ALTERNATING FORM                            ///#
#//////////////////////////////////////////////////////////////////////////////#

function (Eq::NonLinearTerm{T, AlternatingForm})(   t::Real,
                                                    U::VectorField{S},
                                                 dUdt::VectorField{S},
                                                  add::Bool=false) where {T, S<:SpectralField{T}}
    return _alternatingform!(Eq, U, dUdt, add)
end

function _alternatingform!(  Eq::NonLinearTerm,
                              U::VectorField{S},
                           dUdt::VectorField{S},
                            add::Bool) where {S<:SpectralField}
    # The first call is divergent; subsequent calls alternate the two forms.
    Eq.flag[] = !Eq.flag[]
    return Eq.flag[] ? _divergenceform!(Eq, U, dUdt, add) : _convectiveform!(Eq, U, dUdt, add)
end

#//////////////////////////////////////////////////////////////////////////////#
#///                             ROTATING FORM                              ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    Eq(t, U, dUdt, add=false)  # Eq with RotatingForm()

Evaluate `FFT(u_total × ω_total)`, where `ω_total = ∇ × u_total` and
`u_total` includes the streamwise base profile. This is the rotational
momentum form; its pressure variable absorbs `|u_total|²/2`.
"""
function (Eq::NonLinearTerm{T, RotatingForm})(   t::Real,
                                                 U::VectorField{S},
                                              dUdt::VectorField{S},
                                               add::Bool=false) where {T, S<:SpectralField{T}}
    return _rotatingform!(Eq, U, dUdt, add)
end

function _rotatingform!(  Eq::NonLinearTerm,
                           U::VectorField{S},
                        dUdt::VectorField{S},
                         add::Bool) where {S<:SpectralField}
    u, n, ω, TMP, Ω = Eq.cache

    TMP .= U
    @views TMP[1][:, 1, 1] .+= Eq.baseflow

    # Compute the three curl components directly in spectral space.
    ddx3!(Ω[1], TMP[2])
    Ω[1] .*= -1
    ddx2!(Ω[1], TMP[3], true)

    ddx1!(Ω[2], TMP[3])
    Ω[2] .*= -1
    ddx3!(Ω[2], TMP[1], true)

    ddx2!(Ω[3], TMP[1])
    Ω[3] .*= -1
    ddx1!(Ω[3], TMP[2], true)

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
