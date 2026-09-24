# Included only after loading CUDA: CPU profiling does not require that package.
function profile_cuda(p, s)
    advance() = step!(p.scheme, p.nlterm, velocity(s), stagepressure(s), 0.0; p.constraint...)
    for _ = 1:10
        advance()
    end
    CUDA.synchronize()
    result = CUDA.@profile begin
        for _ = 1:PROFILE_STEPS
            advance()
        end
        CUDA.synchronize()
    end
    # In a script the macro's result is not displayed automatically.
    show(IOContext(stdout, :limit => false), result)
    println()

    # All captured events belong to the warmed profiling region above.
    # CUDA 6 exposes event tables in seconds. Preserve device activity grouped
    # by kernel name; host API times overlap execution and must not be added.
    groups = Dict{String,Tuple{Float64,Int}}()
    for i in eachindex(result.device.id)
        name = string(result.device.name[i])
        time, count = get(groups, name, (0.0, 0))
        groups[name] = (time + result.device.stop[i] - result.device.start[i], count+1)
    end
    path = get(ENV,"CHANNEL_GPU_PROFILE_CSV",joinpath(@__DIR__,"results","gpu-kernels.csv"))
    mkpath(dirname(path))
    open(path,"w") do io
        println(io,"name,seconds,calls")
        for (name,(seconds,calls)) in sort(collect(groups);by=x->last(x)[1],rev=true)
            println(io,'"',replace(name,"\""=>"\"\""),'"',',',seconds,',',calls)
        end
    end
end
