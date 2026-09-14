import FFTW: unsafe_execute!, plan_rfft, plan_brfft
import Base.Threads

export FFT, IFFT, ForwardFFT!, InverseFFT!

# forward transform
struct ForwardFFT!{SIZE, B, P}
    plan::P
    function ForwardFFT!(        u::Field{T, SIZE, CPU}, 
                             flags::UInt32=FFTW.EXHAUSTIVE,
                         timelimit::Real=FFTW.NO_TIMELIMIT) where {T, SIZE}
        plan = plan_rfft(parent(u), [1, 2]; 
                         flags=flags, timelimit=timelimit)
        new{SIZE, CPU, typeof(plan)}(plan)
    end
end

# callable interface
function (f::ForwardFFT!{SIZE, CPU})(U::FTField{T, SIZE, CPU}, 
                                     u::Field{T, SIZE, CPU}) where {T, SIZE}
    # apply transform
    unsafe_execute!(f.plan, parent(u), parent(U))

    # normalize (understand how to do this regardless of backend)
    Threads.@threads for i = 1:length(U)
        U[i] *= 1/(SIZE[1]*SIZE[2]) 
    end
    return U
end

function (f::ForwardFFT!{SIZE, CPU})(U::VectorField{FT}, 
                                     u::VectorField{F}) where {FT, F, T, SIZE}
    f(U[1], u[1])
    f(U[2], u[2])
    f(U[3], u[3])
    return U
end

function (f::ForwardFFT!{SIZE, CPU})(U::GradientField{FT}, 
                                     u::GradientField{F}) where {FT, F, T, SIZE}
    f(U[1], u[1])
    f(U[2], u[2])
    f(U[3], u[3])
    return U
end

# inverse transform
struct InverseFFT!{SIZE, B, P}
    plan::P
    function InverseFFT!(        U::FTField{T, SIZE, CPU}, 
                             flags::UInt32=FFTW.EXHAUSTIVE,
                         timelimit::Real=FFTW.NO_TIMELIMIT) where {T, SIZE}
        plan = plan_brfft(parent(U), SIZE[1], [1, 2]; 
                          flags=flags, timelimit=timelimit)
        new{SIZE, CPU, typeof(plan)}(plan)
    end
end

# callable interface
function (i::InverseFFT!{SIZE})(u::Field{T, SIZE, CPU}, 
                                U::FTField{T, SIZE, CPU}) where {T, SIZE}
    # apply transform
    unsafe_execute!(i.plan, parent(U), parent(u))

    return u
end