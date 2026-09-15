import FFTW

abstract type SolutionMode end
struct ConstantPressureGradient <: SolutionMode end
struct ConstantFlowRate <: SolutionMode end


struct ChannelFlow{NLT<:NonLinearTerm}
    nlterm::NLT


end


function ChannelFlow(grid::Grid,
                     mode::SolutionMode = ConstantPressureGradient,
                     form::NonlinearityForm = AlternatingForm(),
                         ::Type{T<:AbstractFloat} = Float64,
                fftwflags::Integer = FFTW.MEASURE,
            fftwtimelimit::Real = -1.0) where {T}
    
    # define nonlinear term
    u = PhysicalField(grid, T)
    U = SpectralField(Array{Complex{T}}(undef, spectralsize(grid, NotPadded())), grid)
    nlterm = NonLinearTerm(u, U;
        fftwflags=fftwflags, fftwtimelimit=fftwtimelimit, form=form)

    # define poisson solver




end
