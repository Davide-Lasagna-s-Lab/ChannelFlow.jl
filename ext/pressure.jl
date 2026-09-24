#//////////////////////////////////////////////////////////////////////////////#
#///                         PRESSURE INITIALIZATION                        ///#
#//////////////////////////////////////////////////////////////////////////////#

function _wallpressure!(plus, minus, a, v, nu)
    s=(blockIdx().x-1)*blockDim().x+threadIdx().x
    if s<=size(a, 1)
        upper=lower=zero(eltype(a))
        for j = 1:size(a, 2)
            n=j-1
            term=a[s, j]+nu*(n^2*(n^2-1)/3)*v[s, j]
            upper+=term
            lower+=(iseven(n) ? term : -term)
        end
        plus[s]=upper
        minus[s]=lower
    end
    return
end
function _pressuremean!(p, a)
    Ny=size(p, 2)
    for j = 2:Ny
        n=j-1
        lower=n==1 ? a[1, 1] : a[1, n]/2
        upper=n+1<Ny-1 ? a[1, n+2]/2 : zero(eltype(a))
        p[1, j]=(lower-upper)/n
    end
    p[1, 1]=0
    mean=zero(eltype(p))
    for j = 3:2:Ny
        mean+=p[1, j]/(1-(j-1)^2)
    end
    p[1, 1]=-mean
    return
end
function CF._pressure_poisson(A::CF.VectorField{F}, v::F, nu::Real) where {F<:CuSpectral{Float64}}
    g=CF.grid(v)
    nx, nz, ny=size(v)
    Lx, _, Lz=CF.domainsize(g)
    κ²=vec([(2π/Lx)^2*(ix-1)^2+(2π/Lz)^2*(iz<=nz÷2+1 ? iz-1 : iz-1-nz)^2 for ix = 1:nx, iz = 1:nz])
    # The mean pressure is integrated separately: never request a singular
    # Neumann factorisation on the device.
    κ²[1]=1
    h=CH.BatchedHelmoltzSolver(ny-1, nx*nz; neum = true)
    CH.update!(h, ones(nx*nz), κ²)
    device=adapt(CuArray, h)
    rhs=similar(v)
    p=similar(v)
    CF.div!(rhs, A)
    plus=CUDA.zeros(ComplexF64, nx*nz)
    minus=similar(plus)
    @cuda threads=256 blocks=cld(nx*nz, 256) _wallpressure!(
        plus,
        minus,
        CF.spectralmatrix(A[2]),
        CF.spectralmatrix(v),
        nu,
    )
    CH.solve!(device, CF.spectralmatrix(p), CF.spectralmatrix(rhs), plus, minus)
    @cuda threads=1 blocks=1 _pressuremean!(CF.spectralmatrix(p), CF.spectralmatrix(A[2]))
    CF.zero_nyquist!(p)
    return p
end
