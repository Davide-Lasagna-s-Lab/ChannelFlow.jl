# Give each sample exactly one category from its call stack. Unlike inclusive
# flat-profile counts, these counts can be added and plotted as percentages.
function save_profile_summary(path)
    data, frames = Profile.retrieve(; include_meta=false)
    categories = Dict{String,Int}()
    folded = Dict{String,Int}()
    stack = Any[]
    for address in data
        if address != 0
            append!(stack, get(frames,address,[]))
            continue
        end
        isempty(stack) && continue
        files = string.(getproperty.(stack,:file))
        containsfile(s) = any(f -> occursin(s,f), files)
        category = if containsfile("ChebyshevHelmoltzSolvers")
            "Helmholtz solves"
        elseif containsfile("transforms/chebyshev.jl")
            "Chebyshev transforms"
        elseif containsfile("FFTW")
            "Fourier transforms"
        elseif containsfile("batchedinfluence.jl")
            "Influence and tau"
        elseif containsfile("fields/operators.jl")
            "Spectral derivatives"
        elseif containsfile("ffts.jl")
            "Padding and normalization"
        elseif containsfile("nonlinear.jl")
            "Nonlinear products"
        else
            "Stage assembly and runtime"
        end
        categories[category] = get(categories,category,0)+1
        # Folded stacks preserve the evidence behind the category summary.
        labels = [replace(string(f.func), ';'=>'/')*" ("*basename(string(f.file))*":"*string(f.line)*")"
                  for f in reverse(stack) if !f.from_c]
        key = join(labels,';')
        folded[key] = get(folded,key,0)+1
        empty!(stack)
    end
    mkpath(dirname(path))
    open(path,"w") do io
        println(io,"category,samples")
        for (name,count) in sort(collect(categories);by=last,rev=true)
            println(io,name,",",count)
        end
    end
    open(replace(path,".csv"=>".folded"),"w") do io
        for (stack,count) in sort(collect(folded);by=first)
            println(io,stack," ",count)
        end
    end
end
