import FFTW
import MPI

abstract type SolutionMode end
struct ConstantPressureGradient <: SolutionMode end
struct ConstantFlowRate <: SolutionMode end


struct ChannelFlow{NLT<:NonLinearTerm}
    nlterm::NLT


end


function ChannelFlow(comm::MPI.Comm,
               domainsize::NTuple{2, Real},
                     grid::FDGrid,
                 gridsize::NTuple{3, Int},
                     mode::SolutionMode = ConstantPressureGradient,
                     form::NonLinearityForm = AlternatingForm,
                         ::Type{T<:AbstractFloat} = Float64,
                fftwflags::Integer = FFTW.MEASURE,
            fftwtimelimit::Real = -1.0) where {T}
    
    # define nonlinear term
    u = PhysicalField(comm, gridsize, T)
    U = SpectralField(comm, domainsize, gridsize, grid, T)
    nlterm = NonLinearTerm(u, U, fftwflags, fftwtimelimit, form)

    # define poisson solver




end