#//////////////////////////////////////////////////////////////////////////////#
#///                         CHEBYSHEV TRANSFORM PLANS                       ///#
#//////////////////////////////////////////////////////////////////////////////#

"""FFTW DCT-I plan with a compile-time inverse endpoint-weighting flag."""
struct FFTWChebyshevPlan{INVERSE, P}
    plan::P
end

#//////////////////////////////////////////////////////////////////////////////#
#///                        SPECTRAL MATRIX VIEW                            ///#
#//////////////////////////////////////////////////////////////////////////////#

"""View Fourier systems as rows and Chebyshev coefficients as columns, without copying."""
spectralmatrix(U::SpectralField) = reshape(parent(U), :, size(U, 3))

#//////////////////////////////////////////////////////////////////////////////#
#///                          PLAN CONSTRUCTION                             ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    plan_cheb(U::SpectralField; flags=FFTW.EXHAUSTIVE, timelimit=FFTW.NO_TIMELIMIT)

Plan the forward, unnormalised wall-normal DCT-I for the layout and type of
`U`. Execute with `mul!(dest, plan, src)` using distinct matching buffers.
The enclosing ForwardFFT! applies coefficient normalization afterwards.
FFTW planning may overwrite the supplied `U`.
On GPU fields, planning uses cuFFT instead; FFTW planning options are ignored.
"""
plan_cheb(U::SpectralField; kwargs...) =
    _plan_cheb(U, Val(false); kwargs...)

"""
    plan_icheb(U::SpectralField; flags=FFTW.EXHAUSTIVE, timelimit=FFTW.NO_TIMELIMIT)

Plan inverse wall-normal evaluation, including doubled endpoint inputs.
Execute with `mul!(dest, plan, src)` using distinct matching buffers.
The result is twice the evaluated Chebyshev series: InverseFFT! applies
its remaining factor of 1/2 during Fourier padding. Planning side effects
match [`plan_cheb`](@ref).
"""
plan_icheb(U::SpectralField; kwargs...) =
    _plan_cheb(U, Val(true); kwargs...)

function _plan_cheb(U::SpectralField, ::Val{INVERSE};
                   flags::Integer=FFTW.EXHAUSTIVE,
                   timelimit::Real=FFTW.NO_TIMELIMIT) where {INVERSE}
    plan = FFTW.plan_r2r!(parent(U), FFTW.REDFT00, (3,);
                         flags=flags, timelimit=timelimit)
    return FFTWChebyshevPlan{INVERSE, typeof(plan)}(plan)
end

#//////////////////////////////////////////////////////////////////////////////#
#///                            PLAN EXECUTION                              ///#
#//////////////////////////////////////////////////////////////////////////////#

"""Apply the FFTW plan to distinct buffers, preserving the source coefficients."""
function LinearAlgebra.mul!(dest::SpectralField, plan::FFTWChebyshevPlan{INV}, src::SpectralField) where {INV}
    copyto!(parent(dest), parent(src))
    if INV
        @views parent(dest)[:, :, 1] .*= 2
        @views parent(dest)[:, :, end] .*= 2
    end
    FFTW.unsafe_execute!(plan.plan, parent(dest), parent(dest))
    return dest
end
