export ShiftReflect, ShiftRotate, apply!

#//////////////////////////////////////////////////////////////////////////////#
#///                         DISCRETE SYMMETRIES                             ///#
#//////////////////////////////////////////////////////////////////////////////#

abstract type ChannelSymmetry end

"""
    ShiftReflect()

The channel-flow symmetry
`(u,v,w)(x,y,z) → (u,v,-w)(x+Lx/2,y,-z)`.
For a scalar field only the coordinates change.

`S(U)` allocates the transformed spectral field; `S(dest, src)` writes it
into `dest`. Use [`apply!`](@ref) to enforce invariance under the symmetry.
"""
struct ShiftReflect <: ChannelSymmetry end

"""
    ShiftRotate()

The plane-Couette symmetry
`(u,v,w)(x,y,z) → (-u,-v,w)(-x+Lx/2,-y,z+Lz/2)`.
For a scalar field only the coordinates change. This symmetry preserves the
Couette base profile and opposite wall velocities; it is not a symmetry of
the usual pressure-driven Poiseuille configuration.

`S(U)` allocates the transformed spectral field; `S(dest, src)` writes it
into `dest`. Use [`apply!`](@ref) to enforce invariance under the symmetry.
"""
struct ShiftRotate <: ChannelSymmetry end

#//////////////////////////////////////////////////////////////////////////////#
#///                     FOURIER--CHEBYSHEV ACTION                          ///#
#//////////////////////////////////////////////////////////////////////////////#

# Half-box translations give (-1)^k and (-1)^l; reflection in y gives
# T_n(-y)=(-1)^n T_n(y). For real fields, the missing negative-x coefficients
# obey a[-k,l,n]=conj(a[k,-l,n]). Thus both actions read the reflected z index.
_symmetry_value(::ShiftReflect, a, k, iz, n, Nz) = isodd(k) ? -a : a
function _symmetry_value(::ShiftRotate, a, k, iz, n, Nz)
    l = iz <= Nz ÷ 2 ? iz : iz - Nz
    return isodd(k+l+n) ? -conj(a) : conj(a)
end

_component_signs(::ShiftReflect) = (1, 1, -1)
_component_signs(::ShiftRotate) = (-1, -1, 1)

function (S::ChannelSymmetry)(dest::SpectralField, src::SpectralField)
    size(dest) == size(src) || throw(DimensionMismatch("symmetry fields must have equal sizes"))

    # Copy only for aliased input/output. The two slices keep z=0 fixed and
    # reverse all remaining z indices, including the self-paired Nyquist mode.
    a = Base.mightalias(parent(dest), parent(src)) ? copy(parent(src)) : parent(src)
    b = parent(dest)
    Nk, Nz, Ny = size(a)
    k = reshape(0:Nk-1, Nk, 1, 1)
    n = reshape(0:Ny-1, 1, 1, Ny)
    for (target, source) in ((1:1, 1:1), (2:Nz, Nz:-1:2))
        iz = reshape(target .- 1, 1, length(target), 1)
        @views b[:, target, :] .= _symmetry_value.(Ref(S), a[:, source, :], k, iz, n, Nz)
    end
    return dest
end

function (S::ChannelSymmetry)(dest::VectorField{<:SpectralField}, src::VectorField{<:SpectralField})
    # Preserve the source even when components share storage across dest/src.
    input = any(Base.mightalias(parent(d), parent(s)) for d in dest for s in src) ? copy(src) : src
    for (d, s, sign) in zip(dest, input, _component_signs(S))
        S(d, s)
        sign == -1 && (d .*= -1)
    end
    return dest
end

function (S::ChannelSymmetry)(dest::State, src::State)
    S(velocity(dest), velocity(src))
    S(stagepressure(dest), stagepressure(src))
    return dest
end

(S::ChannelSymmetry)(src::Union{SpectralField, VectorField{<:SpectralField}, State}) = S(similar(src), src)

#//////////////////////////////////////////////////////////////////////////////#
#///                      PROJECTION ONTO INVARIANT FIELDS                  ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    apply!(U, S)
    apply!(U, (S1, S2, ...))

Enforce a discrete symmetry by replacing `U` with `(U + S(U))/2`. Accept a
spectral scalar field, spectral velocity field, or `State` (velocity and
stage pressure). Return the modified input.

For a tuple, apply each projector in order. `ShiftReflect()` and
`ShiftRotate()` are commuting involutions, so this enforces both symmetries.
The field must represent a real physical field. Projection is performed in
place without allocating a temporary field.
This projects onto a fixed symmetry subspace, not a continuous-translation
symmetry-reduced representation.
"""
apply!(U::SpectralField, S::ChannelSymmetry) = _project_symmetry!(U, S, 1)

function apply!(U::VectorField{<:SpectralField}, S::ChannelSymmetry)
    for (Ui, sign) in zip(U, _component_signs(S))
        _project_symmetry!(Ui, S, sign)
    end
    return U
end

function apply!(state::State, S::ChannelSymmetry)
    apply!(velocity(state), S)
    apply!(stagepressure(state), S)
    return state
end

function apply!(U, symmetries::Tuple)
    for S in symmetries
        apply!(U, S)
    end
    return U
end

# Each (k,l,n) couples only to (k,-l,n). Read both old values before
# overwriting either, so no work array is needed. This helper is also used
# by the GPU kernel, where one thread owns an entire pair.
@inline function _project_symmetry_pair!(a, S, sign, ix, iz, iy)
    Nz = size(a, 2)
    jz = iz == 1 ? 1 : Nz - iz + 2
    @inbounds begin
        left, right = a[ix, iz, iy], a[ix, jz, iy]
        a[ix, iz, iy] = (left + sign*_symmetry_value(S, right, ix-1, iz-1, iy-1, Nz))/2
        if jz != iz
            a[ix, jz, iy] = (right + sign*_symmetry_value(S, left, ix-1, jz-1, iy-1, Nz))/2
        end
    end
    return nothing
end

function _project_symmetry!(U::SpectralField, S::ChannelSymmetry, sign)
    a = parent(U)
    Nk, Nz, Ny = size(a)
    # Visit each pair once, including the self-paired zero and (for even Nz)
    # Nyquist planes. The x index is innermost for contiguous CPU accesses.
    for iy in 1:Ny, iz in 1:Nz÷2+1, ix in 1:Nk
        _project_symmetry_pair!(a, S, sign, ix, iz, iy)
    end
    return U
end
