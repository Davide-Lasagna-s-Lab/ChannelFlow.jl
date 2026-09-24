#//////////////////////////////////////////////////////////////////////////////#
#///                        COMMON SCALAR FIELD TYPE                        ///#
#//////////////////////////////////////////////////////////////////////////////#

"""Common type alias for the package's physical and spectral scalar fields."""
const AbstractField = Union{SpectralField, PhysicalField}

#//////////////////////////////////////////////////////////////////////////////#
#///                        STORAGE-AWARE BROADCAST                         ///#
#//////////////////////////////////////////////////////////////////////////////#

_fielddata(x) = x
_fielddata(x::AbstractField) = parent(x)
_fielddata(bc::Base.Broadcast.Broadcasted) =
    Base.Broadcast.broadcasted(bc.f, map(_fielddata,bc.args)...)

function Base.Broadcast.materialize!(dest::AbstractField, bc::Base.Broadcast.Broadcasted)
    Base.Broadcast.materialize!(parent(dest), _fielddata(bc))
    return dest
end
Base.fill!(field::AbstractField, value) = (fill!(parent(field),value); field)
