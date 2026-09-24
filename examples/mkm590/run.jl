# Run from the repository root with: julia --project=test/cuda examples/mkm590/run.jl
using ChannelFlow, CUDA, Adapt, FFTW, LinearAlgebra, Serialization, TOML, Printf
import ChebyshevHelmoltzSolvers as CH
include("statistics.jl")

#//////////////////////////////////////////////////////////////////////////////#
#///                           SETTINGS TO EDIT                            ///#
#//////////////////////////////////////////////////////////////////////////////#

# Resolved grid, in x, y, z order. Fourier products use 3/2 padding internally.
NX = 384
NY = 257
NZ = 384

# Simulation length: N_STEPS is the number of ADDITIONAL steps after a restart.
DT = 0.0025
N_STEPS = 200000
# Stop cleanly before the 12-hour Slurm limit, saving a restart checkpoint.
MAX_WALL_TIME = 11 * 3600

# Initial perturbation and the start of the statistical averaging window.
# Time is measured in h/Ub. Inspect transition before accepting STATS_START.
AMPLITUDE = 0.01
STATS_START = 200.0

# Output cadence, in integer timesteps, is unchanged between restarts.
SAVE_EVERY = 100
CHECKPOINT_EVERY = 1000
OUTPUT = joinpath(@__DIR__, "output")

# Set RESTART to true to continue OUTPUT/checkpoint.bin.
# Set RESET_STATS too if you want a fresh average of the continuing flow.
RESTART = false
RESET_STATS = false

#//////////////////////////////////////////////////////////////////////////////#
#///                         MKM590 PHYSICAL PARAMETERS                     ///#
#//////////////////////////////////////////////////////////////////////////////#

# The reference box has half-height h=1. Velocities here are scaled by Ub=1.
LX = 2π
LZ = π
U_BULK = 1.0

# Quadrature of the published mean profile gives Ub/u_tau = 18.6544.
# Matching Re_bulk reproduces the constant-flux control used by MKM.
# Re_tau is an OUTCOME of the simulation, not a prescribed wall condition.
RE_TAU_REFERENCE = 587.19
U_BULK_PLUS = 18.654399978959994
NU = 1 / (RE_TAU_REFERENCE * U_BULK_PLUS)

# Pass settings into the numerical driver so they become typed local values.
# For a short execution check, run_dns also accepts a modified settings tuple.
settings = (; NX, NY, NZ, DT, N_STEPS, MAX_WALL_TIME, AMPLITUDE, STATS_START,
              SAVE_EVERY, CHECKPOINT_EVERY, OUTPUT, RESTART, RESET_STATS,
              LX, LZ, U_BULK, RE_TAU_REFERENCE, NU)

#//////////////////////////////////////////////////////////////////////////////#
#///                       INITIAL VELOCITY AND PRESSURE                    ///#
#//////////////////////////////////////////////////////////////////////////////#

"""Construct smooth no-slip rolls and oblique perturbations of the laminar flow."""
function initial_state(problem, amplitude)
    g = problem.grid
    y, x, z = map(CuArray, points(g, Padded()))

    # These physical arrays contain perturbations only. The parallel laminar
    # profile belongs to problem and is included by the nonlinear operator.
    u = VectorField(ntuple(3) do _
        PhysicalField(CUDA.zeros(Float64, physicalsize(g, Padded())), g)
    end)
    a, b, c = map(parent, u.components)

    A = amplitude / 4

    # Differentiate a smooth vector potential analytically. The (1-y²)²
    # envelope makes every component vanish at y=±1; the terms cancel in div(u).
    # Streamwise-independent rolls drive lift-up, while the x-dependent terms
    # break streamwise symmetry. No reference turbulent field is imposed.
    @. a = -4A*y*(1-y^2) * (cos(x)*cos(2z) + 0.5sin(2x+0.3)*cos(4z+0.2))
    @. b = A*(1-y^2)^2 * (sin(x)*cos(2z) - cos(2x+0.3)*cos(4z+0.2) +
                          2cos(2z) + 2cos(4z+0.4))
    @. c = 4A*y*(1-y^2) * (sin(2z) + 0.5sin(4z+0.4))

    # Transform once, then reconstruct the pressure required by the first RK
    # stage. The analytic velocity is already divergence-free and no-slip.
    prototype = SpectralField(CUDA.zeros(ComplexF64, spectralsize(g, NotPadded())), g)
    U = VectorField(prototype)
    problem.nlterm.fft(U, u)
    return State(U, pressure(U, problem))
end

#//////////////////////////////////////////////////////////////////////////////#
#///                           RESTART CHECKPOINT                           ///#
#//////////////////////////////////////////////////////////////////////////////#

"""Atomically save the state, step number, configuration and profile statistics."""
function checkpoint(output, state, step, config, stats)
    path = joinpath(output, "checkpoint.bin")

    # Transfer the checkpoint to host memory. Both velocity and stage pressure
    # are needed to resume the same discrete evolution; time is stored separately.
    record = (; state=adapt(Array, state), step, config, stats)
    serialize(path * ".tmp", record)
    mv(path * ".tmp", path; force=true)
end

#//////////////////////////////////////////////////////////////////////////////#
#///                           SIMULATION DRIVER                           ///#
#//////////////////////////////////////////////////////////////////////////////#

function run_dns(s)
    (; NX, NY, NZ, DT, N_STEPS, MAX_WALL_TIME, AMPLITUDE, STATS_START,
       SAVE_EVERY, CHECKPOINT_EVERY, OUTPUT, RESTART, RESET_STATS,
       LX, LZ, U_BULK, RE_TAU_REFERENCE, NU) = s
    started_at = time()

    CUDA.functional() || error("This example requires an NVIDIA GPU")
    CUDA.allowscalar(false)
    FFTW.set_num_threads(4)
    BLAS.set_num_threads(4)

    # Record the parameters that must agree when resuming a checkpoint.
    # N_STEPS is deliberately excluded: a restart extends the existing run.

    config = Dict("Nx" => NX, "Ny" => NY, "Nz" => NZ,
                  "Lx" => Float64(LX), "Lz" => Float64(LZ), "dt" => DT, "nu" => NU,
                  "bulk_velocity" => U_BULK, "amplitude" => AMPLITUDE,
                  "Re_tau_reference" => RE_TAU_REFERENCE,
                  "save_every" => SAVE_EVERY, "statistics_start" => STATS_START)

    mkpath(OUTPUT)
    saved = RESTART ? deserialize(joinpath(OUTPUT, "checkpoint.bin")) : nothing

    # A stopped job may have written slices after its last checkpoint. Remove
    # those on restart, so the output cannot contain duplicated times or samples.
    if RESTART
        oldconfig = copy(saved.config)
        RESET_STATS && (oldconfig["statistics_start"] = STATS_START)
        oldconfig == config || error("Restart configuration differs from checkpoint")
        for file in readdir(OUTPUT)
            matchstep = match(r"^slice-(\d+)\.bin$", file)
            if matchstep !== nothing && parse(Int, matchstep[1]) > saved.step
                rm(joinpath(OUTPUT, file))
            end
        end
        history = joinpath(OUTPUT, "history.csv")
        lines = filter(readlines(history)) do line
            startswith(line, "step,") || parse(Int, first(split(line, ','))) <= saved.step
        end
        write(history, join(lines, '\n') * "\n")
    else
        isfile(joinpath(OUTPUT, "config.toml")) &&
            error("Output exists: set RESTART=true or choose another OUTPUT directory")
        write(joinpath(OUTPUT, "history.csv"),
              "step,time,time_friction,bulk_velocity,Re_tau_wall,cfl,perturbation_energy\n")
    end

    open(io -> TOML.print(io, config), joinpath(OUTPUT, "config.toml"), "w")

    #//////////////////////////////////////////////////////////////////////////#
    #///                   BUILD AND TRANSFER THE SOLVER                    ///#
    #//////////////////////////////////////////////////////////////////////////#

    g = Grid(NX, NY, NZ, LX, LZ)
    println("Building factors and transferring to GPU: ", (NX, NY, NZ))
    flush(stdout)

    # The laminar profile 1.5Ub(1-y²) has bulk velocity Ub. The solver adjusts
    # the streamwise pressure gradient to keep the TOTAL bulk velocity fixed.
    cpu = ChannelFlowProblem(g, y -> 1.5U_BULK*(1-y^2), NU, DT;
                             bulkvelocity=(U_BULK, 0.0), fftwflags=FFTW.ESTIMATE)
    problem = adapt(CuArray, cpu)
    cpu = nothing
    GC.gc()
    CUDA.reclaim()

    println("Initializing state")
    flush(stdout)

    state = RESTART ? adapt(CuArray, saved.state) : initial_state(problem, AMPLITUDE)
    first_step = RESTART ? saved.step : 0
    stats = RESTART && !RESET_STATS ? saved.stats : ProfileStatistics(NY)

    # Roll derived profiles back with the state, or explicitly discard the
    # old averaging window. The checkpoint is the authoritative restart record.
    if stats.count == 0
        for file in ("profiles.csv", "statistics.toml")
            rm(joinpath(OUTPUT, file); force=true)
        end
    else
        save_profiles(stats, g.y, OUTPUT)
    end

    saved = nothing
    GC.gc()
    CUDA.reclaim()

    #//////////////////////////////////////////////////////////////////////////#
    #///                  WORKSPACE FOR SLICES AND PROFILES                 ///#
    #//////////////////////////////////////////////////////////////////////////#

    physical = VectorField(ntuple(3) do _
        PhysicalField(CUDA.zeros(Float64, physicalsize(g, Padded())), g)
    end)

    Nxp, Nzp, _ = physicalsize(g, Padded())
    z = collect(range(0, LZ; length=Nzp+1))[1:end-1]
    base = CuArray(reshape(1.5U_BULK .* (1 .- g.y.^2), 1, 1, NY))

    # Use the smaller adjacent wall-normal spacing for a conservative local
    # advective CFL diagnostic on the nonuniform Chebyshev grid.
    dy = [minimum(abs(g.y[j]-g.y[k]) for k in max(1,j-1):min(NY,j+1) if k != j)
          for j in 1:NY]
    invdy = CuArray(reshape(1 ./ dy, 1, 1, NY))

    #//////////////////////////////////////////////////////////////////////////#
    #///                      SAMPLE AND SAVE THE FLOW                     ///#
    #//////////////////////////////////////////////////////////////////////////#

    function record(step)
        t = step * DT
        problem.nlterm.ifft(physical, velocity(state))
        u, v, w = map(parent, physical.components)
        cfl = DT * maximum(abs.(u .+ base)./(LX/Nxp) .+
                           abs.(v).*invdy .+ abs.(w)./(LZ/Nzp))
        isfinite(cfl) && cfl < 0.8 || error("Unsafe advective CFL=$cfl; reduce DT")

        # The (0,0) Fourier mode is the plane mean, in Chebyshev coefficients.
        # Add the laminar coefficients before evaluating wall shear and flux.

        mean_u = real.(Array(@view parent(velocity(state)[1])[1,1,:]))
        mean_u[1] += 0.75U_BULK
        mean_u[3] -= 0.75U_BULK

        shear = NU * (CH.diff(mean_u, :left) - CH.diff(mean_u, :right)) / 2
        retau = sqrt(max(0, shear)) / NU

        bulk = sum(mean_u[j]/(1-(j-1)^2) for j in 1:2:NY)
        energy = kinetic_energy(velocity(state))

        # Store TOTAL streamwise velocity at x=0, including the laminar base.
        # Binary layout: Nz, Ny (Int64); time, z, y, u[z,y] (Float64, column-major).
        slice = Array(@view u[1,:,:]) .+ reshape(1.5U_BULK .* (1 .- g.y.^2), 1, NY)

        path = joinpath(OUTPUT, @sprintf("slice-%08d.bin", step))
        open(path * ".tmp", "w") do io
            write(io, Int64(Nzp), Int64(NY), Float64(t), z, g.y, slice)
        end
        mv(path * ".tmp", path; force=true)

        open(joinpath(OUTPUT, "history.csv"), "a") do io
            println(io, join((step, t, t*NU*RE_TAU_REFERENCE, bulk, retau, cfl, energy), ','))
        end

        @printf("step=%d t=%.3f Re_tau_wall=%.2f CFL=%.3f E'=%.6g\n",
                step, t, retau, cfl, energy)
        flush(stdout)

        # Sample only the requested averaging window. The profile collector
        # includes both spatial fluctuations and changes in the plane means.
        if t >= STATS_START
            u .+= base
            sample!(stats, physical, t)
            save_profiles(stats, g.y, OUTPUT)
        end
    end

    #//////////////////////////////////////////////////////////////////////////#
    #///                       ADVANCE AND CHECKPOINT                      ///#
    #//////////////////////////////////////////////////////////////////////////#

    if !RESTART
        record(0)
        checkpoint(OUTPUT, state, 0, config, stats)
    end

    for step in first_step+1:first_step+N_STEPS
        step!(problem.scheme, problem.nlterm, velocity(state), stagepressure(state),
              (step-1)*DT; forcing=problem.forcing, problem.constraint...)

        # Integer-step sampling keeps uniform spacing across successive runs.
        step % SAVE_EVERY == 0 && record(step)
        if step % CHECKPOINT_EVERY == 0 || step == first_step+N_STEPS
            checkpoint(OUTPUT, state, step, config, stats)
        end

        # Touch OUTPUT/STOP to request a clean stop without killing the process.
        # Also stop before the scheduler limit, preserving the current state.
        # Do not add an off-cadence statistics sample at this final step.
        if isfile(joinpath(OUTPUT, "STOP")) || time()-started_at >= MAX_WALL_TIME
            checkpoint(OUTPUT, state, step, config, stats)
            break
        end
    end

    save_profiles(stats, g.y, OUTPUT)
    println("Finished; set RESTART=true to continue.")
end

# Execute when run as a script; including this file only defines the example.
if abspath(PROGRAM_FILE) == @__FILE__
    run_dns(settings)
end
