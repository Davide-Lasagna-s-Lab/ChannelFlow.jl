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

struct NonLinearTerm{T, FORM<:NonlinearityForm, CACHE}
    cache::CACHE       # the cache is specific to the form used 
     flag::Ref{Symbol} # this gets swapped at every call in the AlternatingForm
    ifft!              # transforms are not typed and this has no effect
     fft!              # on performance :)

    function NonLinearTerm(u::PhysicalField{T},
                           U::SpectralField{T},
                   fftwflags::Integer,
               fftwtimelimit::Real,
                        form::FORM) where {T, FORM<:NonlinearityForm}
        cache = _gencache(u, U, form)
        new{T, FORM, typeof(cache)}(cache, 
            Ref{Symbol}(:DivergenceForm),
            ForwardFFT!(u, flags, timelimit),
            InverseFFT!(U, flags, timelimit))
    end
end

# -------------------------------------------------------------------------------- #
# Convective form

_gencache(::ConvectiveForm, u::PhysicalField, U::SpectralField) =
    (similar(u),
     similar(u),
     GradientField(u),
     VectorField(U),
     GradientField(U))

function (Eq::NonLinearTerm{T, ConvectiveForm})(t::Real,
                                                U::S,
                                             dUdt::S,
                                              add::Bool = false) where {T, S<:SpectralField{T}}

    # get aliases
    u, n, grad, TMP, GRAD  = Eq.cache

    # calc gradient field
    grad!(GRAD, U)

    # compute transforms
    Eq.ifft!(u, U)
    Eq.ifft!(grad, GRAD)

    # calculate product in physical space
    dot!(n, u, grad)

    # back transform
    add == true ? (Eq.fft!(TMP, n); dUdt .+= TMP) : Eq.fft!(dUdt, n)

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
    Eq.ifft!(u, U)

    # calc outer product
    outer!(uu, u)

    # transform to spectral space
    Eq.fft!(UU, uu)

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

