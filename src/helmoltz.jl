using DecomposedArrays
using FDGrids

import LinearAlgebra

struct HelmoltzSolver{T, G<:FDGrid}
    lufacts::Matrix{}
    buffs::NTuple{2, Vector{T}}
end

function LinearAlgebra.ldiv!(OUT::SpectralField{T, <:DecomposedArray},
                              hs::HelmoltzSolver,
                               F::SpectralField{T, <:DecomposedArray})

    # check that F and OUT have data transposed in the correct format
    isaligned(parent(OUT), 3) ||
        throw(ArgumentError("invalid transposition state"))


    for (j, k) in hs.wavenumbers
        hs.buffs[1] .= @view F[j, k, :]
        LinearAlgebra.ldiv!(hs.buffs[2], hs.lufacts[j, k], hs.buffs[1])
        OUT[j, k, :] .= hs.buffs[2]
    end

end
