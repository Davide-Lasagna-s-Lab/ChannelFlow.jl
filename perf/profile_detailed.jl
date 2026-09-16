# Diagnostic instrumentation only: production source is not modified.
using ChannelFlow, FFTW, LinearAlgebra, Random, Statistics, Profile, SHA, Dates
import ChebyshevHelmoltzSolvers
const CF = ChannelFlow
const LABELS = ("Copy velocity + base flow", "Spectral velocity gradient",
    "Inverse transforms: velocity (3)", "Inverse transforms: gradient (9)",
    "Physical convection products", "Forward transforms (3)", "Store nonlinear RHS",
    "RK history", "CN RHS: Laplacians, pressure gradients, broadcasts", "Stokes solve")

# Match the convective production path, with disjoint timing intervals.
function timed_nonlinear!(eq, U, N, costs)
    u,n,grad,tmp,G = eq.cache
    t=time_ns(); tmp .= U
    @views tmp[1][:,1,1] .+= eq.baseflow
    costs[1] += time_ns()-t
    t=time_ns(); CF.grad!(G,tmp); costs[2] += time_ns()-t
    t=time_ns(); eq.ifft(u,tmp); costs[3] += time_ns()-t
    t=time_ns(); eq.ifft(grad,G); costs[4] += time_ns()-t
    t=time_ns(); CF.dot!(n,u,grad); costs[5] += time_ns()-t
    t=time_ns(); eq.fft(tmp,n); costs[6] += time_ns()-t
    t=time_ns(); CF._store_rhs!(N,tmp,false,-1); costs[7] += time_ns()-t
end

# Benchmark-only copy of the three-stage Couette/constant-gradient path.
# Check its final state against production step! before collecting timings.
function timed_step!(c,state,costs)
    s=c.scheme; U=velocity(state); P=pressure(state)
    Q,N,R=s.Q,s.N,s.R
    for j=1:3
        timed_nonlinear!(c.nlterm,U,N,costs)
        t=time_ns()
        if j==1; Q .= N; else; Q .= CF.CNRK2_A[j].*Q .+ N; end
        costs[8] += time_ns()-t
        t=time_ns()
        lambda=inv(CF.CNRK2_C[j]*s.dt); weight=CF.CNRK2_B[j]/CF.CNRK2_C[j]
        for (i,derivative!) in enumerate((ddx1!,ddx2!,ddx3!))
            CF.laplacian!(R[i],U[i]); derivative!(N[i],P)
            R[i] .= lambda.*U[i] .+ s.nu.*R[i] .- N[i] .+ weight.*Q[i]
        end
        @views R[1][:,1,1] .+= 2 .* parent(s.baseviscous)
        R[1][1,1,1] -= c.constraint.pressuregradient[1]
        R[3][1,1,1] -= c.constraint.pressuregradient[2]
        costs[9] += time_ns()-t
        t=time_ns()
        CF.solve!(s.solvers[j],U,P,R; c.constraint...,baseflow=s.baseflow)
        costs[10] += time_ns()-t
    end
end

function restore!(dest,src)
    for i=1:3; copyto!(parent(velocity(dest)[i]),parent(velocity(src)[i])); end
    copyto!(parent(pressure(dest)),parent(pressure(src)))
end
function measure(f::F, reset::R; repeats=21) where {F,R}
    reset(); f()
    samples=Float64[]
    for _=1:repeats
        reset(); t=time_ns(); f(); push!(samples,(time_ns()-t)/1e6)
    end
    reset(); bytes=@allocated f()
    return (;median=median(samples), minimum=minimum(samples), maximum=maximum(samples), bytes)
end
function identity(root)
    paths=sort([joinpath(d,f) for (d,_,fs) in walkdir(joinpath(root,"src")) for f in fs if endswith(f,".jl")])
    (commit=readchomp(`git -C $root rev-parse HEAD`),
     dirty=!isempty(readchomp(`git -C $root status --porcelain -- src Project.toml`)),
     sha256=bytes2hex(sha256(join(read.(paths,String)))))
end

function run_case(io,flags,label)
    g=Grid(35,32,32,2π/1.14,2π/2.5)
    c=Couette(g,1/400,0.025;form=ChannelFlow.ConvectiveForm(),fftwflags=flags)
    rng=MersenneTwister(42)
    physical=PhysicalField(zeros(physicalsize(g,Padded())),g)
    initial=zero_state(g); fft=ForwardFFT!(physical;flags=FFTW.ESTIMATE)
    for u in velocity(initial).components
        randn!(rng,parent(physical)); parent(physical) .*= 0.01; fft(u,physical)
    end
    initial=project!(velocity(initial)); fill!(pressure(initial),0)
    state=copy(initial); reference=copy(initial); costs=zeros(10)
    direct()=step!(c.scheme,c.nlterm,velocity(state),pressure(state),0.;c.constraint...)
    reset()=restore!(state,initial)
    reset(); direct(); restore!(reference,state)
    reset(); timed_step!(c,state,costs)
    error=maximum(maximum(abs,parent(velocity(state)[i])-parent(velocity(reference)[i])) for i=1:3)
    perror=maximum(abs,parent(pressure(state))-parent(pressure(reference)))
    @assert error < 1e-12 && perror < 1e-12
    production=measure(direct,reset)
    for _=1:3; reset(); timed_step!(c,state,costs); end
    samples=zeros(10,21); totals=zeros(21)
    for k=1:21
        reset(); fill!(costs,0); t=time_ns(); timed_step!(c,state,costs)
        totals[k]=(time_ns()-t)/1e6; samples[:,k].=costs./1e6
    end
    println(io,"\n## $label\n")
    println(io,"Production step: median **$(round(production.median,digits=3)) ms**, min $(round(production.minimum,digits=3)), max $(round(production.maximum,digits=3)); Julia heap bytes: $(production.bytes).")
    println(io,"\nInstrumented step: mean $(round(mean(totals),digits=3)) ms, median $(round(median(totals),digits=3)) ms. Agreement with production: velocity max error $error, pressure max error $perror.")
    println(io,"\n| Disjoint phase, summed over three stages | Mean ms/step | Share of instrumented wall time |\n|---|---:|---:|")
    for i=1:10
        v=mean(samples[i,:]); println(io,"| $(LABELS[i]) | $(round(v,digits=3)) | $(round(100v/mean(totals),digits=2))% |")
    end
    residual=mean(totals)-sum(mean(samples,dims=2))
    println(io,"| Timing/control overhead outside intervals | $(round(residual,digits=3)) | $(round(100residual/mean(totals),digits=2))% |")
    # Nested kernel costs: never add this table to the preceding one.
    u,n,grad,tmp,G=c.nlterm.cache
    inv=c.nlterm.ifft; fw=c.nlterm.fft
    a=copy(parent(inv.resolved)); b=copy(parent(inv.padded)); p=copy(parent(u[1]))
    kernels=(("Inverse scalar, complete",()->inv(u[1],tmp[1]),()->nothing,36),
      ("Forward scalar, complete",()->fw(tmp[1],u[1]),()->nothing,9),
      ("Inverse DCT-I, resolved",()->FFTW.unsafe_execute!(inv.chebyplan,parent(inv.resolved),parent(inv.resolved)),()->copyto!(parent(inv.resolved),a),36),
      ("Inverse Fourier brfft",()->FFTW.unsafe_execute!(inv.plan,parent(inv.padded),parent(u[1])),()->copyto!(parent(inv.padded),b),36),
      ("Forward Fourier rfft",()->FFTW.unsafe_execute!(fw.plan,parent(u[1]),parent(fw.padded)),()->copyto!(parent(u[1]),p),9),
      ("Forward DCT-I, resolved",()->FFTW.unsafe_execute!(fw.chebyplan,parent(tmp[1]),parent(tmp[1])),()->copyto!(parent(tmp[1]),a),9),
      ("Scalar Laplacian",()->CF.laplacian!(c.scheme.R[1],velocity(state)[1]),()->nothing,9),
      ("Streamwise derivative",()->ddx1!(c.scheme.N[1],pressure(state)),()->nothing,12),
      ("Wall-normal derivative",()->ddx2!(c.scheme.N[1],pressure(state)),()->nothing,12),
      ("Spanwise derivative",()->ddx3!(c.scheme.N[1],pressure(state)),()->nothing,12))
    println(io,"\nNested kernels (isolated, warmed; **not additive** with phase table):\n\n| Kernel | Median ms/call | Calls/step | Julia bytes/call |\n|---|---:|---:|---:|")
    for (name,f,setup,calls) in kernels
        r=measure(f,setup); println(io,"| $name | $(round(r.median,digits=4)) | $calls | $(r.bytes) |")
    end
    reset(); direct(); Profile.clear()
    Profile.@profile for _=1:100; direct(); end
    open(joinpath(@__DIR__,"cpu-detailed-$(lowercase(label)).txt"),"w") do stream
        Profile.print(stream;format=:flat,sortedby=:count,mincount=10,C=true)
    end
    flush(io)
end

function main()
    BLAS.set_num_threads(1); FFTW.set_num_threads(1)
    root=dirname(dirname(pathof(ChannelFlow)))
    open(joinpath(@__DIR__,"profiling-details.md"),"w") do io
        println(io,"# Detailed DNS CPU profile\n\nMeasured $(Dates.now(Dates.UTC)) UTC. Julia $VERSION; CPU $(Sys.CPU_NAME); Julia threads $(Threads.nthreads()); BLAS/FFTW threads 1.")
        println(io,"\nChannelFlow: `$(identity(root))`\n\nChebyshev backend: `$(identity(dirname(dirname(pathof(ChebyshevHelmoltzSolvers)))))`")
        println(io,"\nCouette Re=400; Ny=35, Nx=Nz=32; padded 35×48×48; dt=0.025; convective form; no monitor. Initialization/planning/compilation excluded. 21 warmed samples per timing; fixed seeded input restored before each whole step. Allocations refer only to Julia's heap, not native FFTW malloc. Instrumentation is a benchmark-only copy of the constant-pressure-gradient convective step, verified against production before timing. It does not cover forcing, constant flux or other nonlinear forms. Raw CPU profiles sample 100 uninstrumented steps including native frames. Kernel timings use separate caches/intervals and cannot be summed as an exact whole-step model.")
        run_case(io,FFTW.ESTIMATE,"ESTIMATE")
        run_case(io,FFTW.MEASURE,"MEASURE")
    end
end
main()
