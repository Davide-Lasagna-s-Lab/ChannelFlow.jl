#//////////////////////////////////////////////////////////////////////////////#
#///                      IN-PLACE SYMMETRY PROJECTION                       ///#
#//////////////////////////////////////////////////////////////////////////////#

function _project_symmetry_kernel!(a, S, sign)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    Nk, Nz, Ny = size(a)
    pairs = Nz÷2 + 1
    if i <= Nk*pairs*Ny
        ix = (i-1)%Nk + 1
        iz = ((i-1)÷Nk)%pairs + 1
        iy = (i-1)÷(Nk*pairs) + 1
        # No two threads touch the same pair, including self-paired planes.
        CF._project_symmetry_pair!(a, S, sign, ix, iz, iy)
    end
    return nothing
end

function CF._project_symmetry!(U::CuSpectral, S::CF.ChannelSymmetry, sign)
    Nk, Nz, Ny = size(U)
    @cuda threads=256 blocks=cld(Nk*(Nz÷2+1)*Ny, 256) _project_symmetry_kernel!(parent(U), S, sign)
    return U
end
