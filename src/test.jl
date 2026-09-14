using CanonicalFlows, 
      FFTW,
      BenchmarkTools,
      LinearAlgebra,
      Base.Threads

function stepping_example(   U::FTField,
                          dUdt::FTField,
                            ex::ExplicitTerm,
                             N::Int,
                            dt::Real)
    for i = 1:N
        ex(0.0, U, dUdt)
        Threads.@threads for j = 1:length(U)
            @inbounds U[j] += dt*dUdt[j]
        end
    end

    return nothing
end

# set num threads
FFTW.set_num_threads(1)

# size
SIZE = (2^6, 2^6, 2^6)

# fields
 U = FTField(SIZE)
dUdt = FTField(SIZE)

# equations
ex = ExplicitTerm(SIZE, CPU(), Float64, FFTW.MEASURE, -1.0)

# benchmark
@time stepping_example($U, $dUdt, $ex, 1, 1e-3);