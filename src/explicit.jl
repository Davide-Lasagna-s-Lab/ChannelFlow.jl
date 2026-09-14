export NonLinearTerm

# -------------------------------------------------------------------------------- #
# Abstract type representing the form used to calculate the nonlinear term.
abstract type NonlinearityForm end

struct ConvectiveForm  <: NonlinearityForm end
struct DivergenceForm  <: NonlinearityForm end
struct AlternatingForm <: NonlinearityForm end
# TODO: add rotating form


# -------------------------------------------------------------------------------- #
# Functor type that evaluates the nonlinear term of the governing equations
# using a pseudo-spectral approach

"""
    NonLinearTerm(u, U;
                  fftwflags=FFTW.EXHAUSTIVE,
                  fftwtimelimit=FFTW.NO_TIMELIMIT,
                  form=ConvectiveForm())

Construct a pseudo-spectral convection operator using scalar physical/spectral
fields `u` and `U` as allocation and transform prototypes.

The stationary streamwise profile `Ub(y)` is taken from `grid(U)`. The evaluation
`Eq(t, Upert, rhs)` accepts spectral vector fields and computes
`-FFT((utotal ⋅ ∇)utotal)`, where
`utotal = upert + Ub(y) e_x`. Both advection and differentiation use this total
velocity, so base advection and shear production are included automatically.

For this parallel reference its self-advection vanishes. The sustaining viscous
and pressure-gradient balance, additional forcing and pressure projection are
handled outside this operator. Base-flow support for the other forms is pending.
"""
struct NonLinearTerm{T, FORM<:NonlinearityForm, CACHE, IFFT, FFT}
       cache::CACHE       # cache specific to the selected nonlinear form
        flag::Ref{Symbol} # swapped at every call in the AlternatingForm
        ifft::IFFT        # concrete callable inverse transform
         fft::FFT         # concrete callable forward transform

    function NonLinearTerm(u::PhysicalField{T},
                           U::S;
                   fftwflags::Integer=FFTW.EXHAUSTIVE,
               fftwtimelimit::Real=FFTW.NO_TIMELIMIT,
                        form::FORM=ConvectiveForm()
                            ) where {T, S<:SpectralField{T}, FORM<:NonlinearityForm}
        # TODO: in realta dobbiamo abilitare anche gli altri modi per il base flow
        form isa ConvectiveForm ||
            throw(ArgumentError("baseflow is currently supported only for ConvectiveForm"))

        cache = _gencache(form, u, U)
        ifft = InverseFFT!(U; flags=fftwflags, timelimit=fftwtimelimit)
        fft = ForwardFFT!(u; flags=fftwflags, timelimit=fftwtimelimit)
        new{T, FORM, typeof(cache), typeof(ifft), typeof(fft)}(
            cache, Ref{Symbol}(:DivergenceForm), ifft, fft)
    end
end

# -------------------------------------------------------------------------------- #
# Convective form

_gencache(::ConvectiveForm, u::PhysicalField, U::SpectralField) =
    (VectorField(u),
     VectorField(u),
     GradientField(u),
     VectorField(U),
     GradientField(U))

"""
    Eq(t, U, dUdt, add=false)

Evaluate signed convective acceleration from the spectral perturbation velocity
`U`. Overwrite `dUdt` by default; accumulate into it when `add=true`. Internal
workspaces are reused, so one `Eq` instance must not be evaluated concurrently.
`t` is retained for the Flows API.

The gradient and FFT interfaces used below are still under development in
CanonicalFlows; this change establishes the vector/base-flow assembly contract.
"""
function (Eq::NonLinearTerm{T, ConvectiveForm})(t::Real,
                                                U::VectorField{S},
                                             dUdt::VectorField{S},
                                              add::Bool = false) where {T, S<:SpectralField{T}}

    # get aliases
    u, n, grad, TMP, GRAD  = Eq.cache

    # TMP initially holds the TOTAL spectral velocity. Add the reference here,
    # before either gradient evaluation or inverse transformation. Adding it
    # only to the physical advecting velocity would omit the v*Ub' shear term.
    for i = 1:3
        TMP[i] .= U[i]
    end
    @views TMP[1][1, 1, :] .+= baseflow(grid(TMP[1]))

    grad!(GRAD, TMP)

    # compute transforms
    Eq.ifft(u, TMP)
    Eq.ifft(grad, GRAD)

    dot!(n, u, grad)

    # dot! produces positive advection. Transform it, then apply the minus
    # sign required by the momentum RHS. TMP can now be reused: total velocity
    # is already consumed. Broadcast on scalar components, since VectorField
    # itself does not provide an AbstractArray/broadcast interface.
    Eq.fft(TMP, n)
    for i = 1:3
        add ? (dUdt[i] .-= TMP[i]) : (dUdt[i] .= .-TMP[i])
    end

    return dUdt
end

# -------------------------------------------------------------------------------- #
# Divergence form

# the cache of the DivergenceForm could be smaller but we make it equal
# to that of the ConvectiveForm so that we can use the AlternatingForm
_gencache(::DivergenceForm, u::PhysicalField, U::SpectralField) =
    _gencache(ConvectiveForm(), u, U)

function (Eq::NonLinearTerm{T, DivergenceForm})(t::Real,
                                                U::S,
                                             dUdt::S,
                                              add::Bool=false) where {T, S<:SpectralField{T}}

    # get aliases
    u, n, uu, N, UU  = Eq.cache

    # transform to physical space
    Eq.ifft(u, U)

    # calc outer product
    outer!(uu, u)

    # transform to spectral space
    Eq.fft(UU, uu)

    # calculate product in physical space
    div!(N, UU)

    # back transform
    add == true ? dUdt .+= N : dUdt .= N

    return dUdt
end

# -------------------------------------------------------------------------------- #
# Alternating form

# The cache here is the same as for the DivergenceForm and ConvectiveForm
_gencache(::AlternatingForm, u::PhysicalField, U::SpectralField) =
    _gencache(ConvectiveForm(), u, U)

function (Eq::NonLinearTerm{T, AlternatingForm})(t::Real,
                                                 U::S,
                                              dUdt::S,
                                               add::Bool=false) where {T, S<:SpectralField{T}}
    # forward call and flip the flag, so next time we call the other form
    Eq.flag[] == :DivergenceForm &&
        (Eq.flag[] = :ConvectiveForm;
         return Eq(t, U, dUdt, add, DivergenceForm()))

    Eq.flag[] == :ConvectiveForm &&
        (Eq.flag[] = :DivergenceForm;
         return Eq(t, U, dUdt, add, ConvectiveForm()))
end
