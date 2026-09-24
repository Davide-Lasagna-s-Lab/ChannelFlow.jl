#//////////////////////////////////////////////////////////////////////////////#
#///                        INFLUENCE AND TAU KERNELS                       ///#
#//////////////////////////////////////////////////////////////////////////////#

function _influence!(p, v, pp, pm, vp, vm, a, b, c, d)
    s=(blockIdx().x-1)*blockDim().x+threadIdx().x
    if s<=size(p, 1)
        upper=lower=zero(eltype(v))
        for j = 2:size(p, 2)
            term=(j-1)^2*v[s, j]
            upper-=term
            lower-=(iseven(j) ? 1 : -1)*term
        end
        δp=a[s]*upper+b[s]*lower
        δm=c[s]*upper+d[s]*lower
        for j = 1:size(p, 2)
            p[s, j]+=δp*pp[s, j]+δm*pm[s, j]
            v[s, j]+=δp*vp[s, j]+δm*vm[s, j]
        end
    end
    return
end
function CF._batchinfluence!(h::CuInfluence, p, v)
    B=size(p, 1)
    @cuda threads=256 blocks=cld(B, 256) _influence!(p, v, h.responses[1:4]..., h.influence...)
    return nothing
end
function _tau!(p, v, dp, Ry, p0, v0, shift, σ01, σ02)
    s=(blockIdx().x-1)*blockDim().x+threadIdx().x
    if s<=size(p, 1)
        Ny=size(p, 2)
        σ1=(shift[s]*v[s, Ny-1]+dp[s, Ny-1]-Ry[s, Ny-1])/(1-σ01[s])
        σ2=(shift[s]*v[s, Ny]+dp[s, Ny]-Ry[s, Ny])/(1-σ02[s])
        for j = 1:Ny
            p[s, j]+=(isodd(j) ? σ1 : σ2)*p0[s, j]
            v[s, j]+=(isodd(j) ? σ2 : σ1)*v0[s, j]
        end
    end
    return
end
function CF._batchtau!(h::CuInfluence, p, v, dp, Ry)
    B=size(p, 1)
    @cuda threads=256 blocks=cld(B, 256) _tau!(
        p,
        v,
        dp,
        Ry,
        h.responses[5:6]...,
        h.shift,
        h.sigma...,
    )
    return nothing
end

#//////////////////////////////////////////////////////////////////////////////#
#///                         MEAN MODE AND CONSTRAINTS                      ///#
#//////////////////////////////////////////////////////////////////////////////#

# The horizontal mean velocities have already been solved by the velocity
# bank with k=0. Only pressure integration and the uniform-gradient response
# remain. There is no second factorisation, packing or host field transfer.
struct CUDAMeanMode{V}
    response::V              # velocity response to a unit uniform gradient
    response_mean::Float64    # wall-normal average of that response
    gradients::V             # two gradients returned by the mean-mode kernel
end
function Adapt.adapt_structure(to::Type{<:CuArray}, s::CF.StokesSolver)
    mean=CUDAMeanMode(
        Adapt.adapt(to, s.mean.response),
        s.mean.response_mean,
        CUDA.zeros(Float64, 2),
    )
    return CF.StokesSolver(s.grid, Adapt.adapt(to, s.modes), mean)
end

function _mean!(u, v, w, p, Ry, response, response_mean, gradients, gx, gz, ux, wz, fixedbulk)
    Ny=size(p, 2)
    for j = 2:Ny
        n=j-1
        lower=n==1 ? Ry[1, 1] : Ry[1, n]/2
        upper=n+1<Ny-1 ? Ry[1, n+2]/2 : zero(eltype(p))
        p[1, j]=(lower-upper)/n
    end
    p[1, 1]=0
    mean=zero(eltype(p))
    um=zero(eltype(u))
    wm=zero(eltype(w))
    for j = 1:2:Ny
        weight=1/(1-(j-1)^2)
        mean+=weight*p[1, j]
        um+=weight*u[1, j]
        wm+=weight*w[1, j]
    end
    p[1, 1]=-mean
    gx=fixedbulk ? (ux-real(um))/response_mean : gx
    gz=fixedbulk ? (wz-real(wm))/response_mean : gz
    gradients[1]=gx
    gradients[2]=gz
    for j = 1:Ny
        u[1, j]+=gx*response[j]
        w[1, j]+=gz*response[j]
        v[1, j]=0
    end
    return
end
function CH.solve!(
    s::CF.StokesSolver{G,H,M},
    U::CF.VectorField{F},
    P::F,
    R::CF.VectorField{F};
    pressuregradient = nothing,
    bulkvelocity = nothing,
    baseflow = nothing,
) where {G,H,M<:CUDAMeanMode,F<:CuSpectral{Float64}}
    isnothing(pressuregradient) ||
        isnothing(bulkvelocity) ||
        throw(ArgumentError("specify one mean-flow constraint"))
    fields=(U.components..., P, R.components...)
    all(f -> CF.grid(f)==s.grid && size(f)==CF.spectralsize(s.grid, CF.NotPadded()), fields) ||
        throw(DimensionMismatch("Stokes fields must match the solver grid"))
    matrices=map(CF.spectralmatrix, fields)
    CH.solve!(s.modes, matrices...)
    gradients=isnothing(pressuregradient) ? (0.0, 0.0) : pressuregradient
    target=isnothing(bulkvelocity) ? (0.0, 0.0) : bulkvelocity
    # Base-profile mean is only required for the optional constant-flux path.
    basemean=isnothing(baseflow) || isnothing(bulkvelocity) ? 0.0 : CF._bulkmean(Array(baseflow))
    @cuda threads=1 blocks=1 _mean!(
        matrices[1:4]...,
        matrices[6],
        s.mean.response,
        s.mean.response_mean,
        s.mean.gradients,
        gradients...,
        target[1]-basemean,
        target[2],
        !isnothing(bulkvelocity),
    )
    for field in (U.components..., P)
        CF.zero_nyquist!(field)
    end
    return isnothing(bulkvelocity) ? gradients : Tuple(Array(s.mean.gradients))
end

function CF._bulkpressuregradient(
    s::CF.CNRK2{S,F},
    U::CF.VectorField,
    N::CF.VectorField,
) where {S,F<:CuSpectral}
    # Constant-flux operation needs two host scalar gradients in the public
    # step API. Transfer only zero-mode profiles, never full velocity fields.
    base=CF._bulkmean(Array(s.basecurvature))
    return ntuple(2) do i
        c=i==1 ? 1 : 3
        u=Array(view(parent(U[c]), 1, 1, :))
        n=Array(view(parent(N[c]), 1, 1, :))
        s.nu*real(CH.diff(u, :right)-CH.diff(u, :left))/2 +
        (i==1 ? s.nu*base : 0.0) +
        real(CF._bulkmean(n))
    end
end
