using DelimitedFiles

#//////////////////////////////////////////////////////////////////////////////#
#///                     PLANE AND TIME AVERAGED PROFILES                   ///#
#//////////////////////////////////////////////////////////////////////////////#

"""Accumulate means and all six velocity covariances at uniformly spaced times."""
mutable struct ProfileStatistics
    count::Int
    first_time::Float64
    last_time::Float64
    mean::Matrix{Float64} # Ny × 3: total u, v, w
    moment::Matrix{Float64} # Ny × 6: covariance sum, including changes of plane means
end
ProfileStatistics(Ny) = ProfileStatistics(0, NaN, NaN, zeros(Ny,3), zeros(Ny,6))
const STRESS_PAIRS = ((1,1),(2,2),(3,3),(1,2),(1,3),(2,3))

"""
    sample!(stats, velocity, t)

Add one uniformly spaced sample of TOTAL physical velocity, with storage
`(x,z,y)`. Average over x and z on the array's device and transfer only the
small profiles to the CPU. The final covariance is relative to the combined
plane-and-time mean, not to each instantaneous plane mean.
"""
function sample!(s::ProfileStatistics, u, t)
    a = map(parent, u.components)
    plane = map(v -> sum(v; dims=(1,2)) ./ (size(v,1)*size(v,2)), a)
    μ = hcat((vec(Array(m)) for m in plane)...)
    # Parallel-variance merge: spatial covariance + the covariance of changing
    # plane means. This avoids subtracting two large accumulated raw moments.
    δ = μ - s.mean
    n = s.count + 1
    for (k,(i,j)) in enumerate(STRESS_PAIRS)
        cov = sum((a[i] .- plane[i]).*(a[j] .- plane[j]); dims=(1,2)) ./ (size(a[i],1)*size(a[i],2))
        s.moment[:,k] .+= vec(Array(cov)) .+ (s.count/n).*δ[:,i].*δ[:,j]
    end
    s.mean .+= δ ./ n
    s.count == 0 && (s.first_time = t)
    s.count = n
    s.last_time = t
    return s
end

"""Write dimensional profiles; normal stresses are variances, not RMS values."""
function save_profiles(s, y, directory)
    s.count == 0 && return
    open(joinpath(directory, "profiles.csv.tmp"), "w") do io
        println(io, "y,U,V,W,uu,vv,ww,uv,uw,vw")
        writedlm(io, hcat(y,s.mean,s.moment./s.count), ',')
    end
    mv(joinpath(directory,"profiles.csv.tmp"), joinpath(directory,"profiles.csv"); force=true)
    open(joinpath(directory,"statistics.toml"), "w") do io
        TOML.print(io, Dict("samples"=>s.count,"first_time"=>s.first_time,
                           "last_time"=>s.last_time,"averaging"=>"uniform time samples and x-z plane"))
    end
end
