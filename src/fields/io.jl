export savefield, loadfield

#//////////////////////////////////////////////////////////////////////////////#
#///                         FIELD AND STATE FILES                          ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    savefield(path, field)

Save a `PhysicalField`, `SpectralField`, `VectorField`, `GradientField`, or
`State` to a binary Julia Serialization file, replacing an existing file.
Preserve numerical types, data, padding, grids and shared references. A state
includes both velocity and stage pressure; no transforms are performed.
Return `path`. The input is not modified.

These files are intended for use with compatible Julia and ChannelFlow type
definitions, not as a stable archival or cross-language format. The simulation
problem, time, forcing and time-stepper caches are not saved.
"""
function savefield(path::AbstractString,
                   field::Union{PhysicalField, SpectralField, VectorField, GradientField, State})
    Serialization.serialize(path, field)
    return path
end

"""
    loadfield(path)

Load a field or state written by [`savefield`](@ref). Its type and grid are
restored automatically, with independent storage from the original object.
Only load trusted files: Julia deserialization is not a safe input validator.

To resume integration, construct the problem using the loaded state's grid,
`grid(velocity(state)[1])`, and the original physical parameters, timestep,
nonlinear form and forcing. An existing problem may also be used when its grid has identical resolution,
domain lengths and wall-normal nodes.
The simulation time must be supplied separately.
"""
function loadfield(path::AbstractString)
    field = Serialization.deserialize(path)
    field isa Union{PhysicalField, SpectralField, VectorField, GradientField, State} ||
        throw(ArgumentError("file does not contain a ChannelFlow field or state"))
    return field
end
