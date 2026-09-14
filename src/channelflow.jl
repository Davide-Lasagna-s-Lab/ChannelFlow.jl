import FFTW

abstract type SolutionMode end
struct ConstantPressureGradient <: SolutionMode end
struct ConstantFlowRate <: SolutionMode end


struct ChannelFlow{NLT<:NonLinearTerm}
    nlterm::NLT


end


function ChannelFlow(grid::Grid,
                     mode::SolutionMode = ConstantPressureGradient,
                     form::NonLinearityForm = AlternatingForm,
                         ::Type{T<:AbstractFloat} = Float64,
                fftwflags::Integer = FFTW.MEASURE,
            fftwtimelimit::Real = -1.0) where {T}
    
    # define nonlinear term
    u = PhysicalField(grid, T)
    Ny, Nx, Nz = gridsize(grid)
    U = SpectralField(Array{Complex{T}}(undef, Ny, Nx, Nz ÷ 2 + 1), grid)
    nlterm = NonLinearTerm(u, U;
        fftwflags=fftwflags, fftwtimelimit=fftwtimelimit, form=form)

    # define poisson solver




end
