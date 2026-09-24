#//////////////////////////////////////////////////////////////////////////////#
#///                         DEVICE STORAGE TRANSFERS                       ///#
#//////////////////////////////////////////////////////////////////////////////#

# Grids contain small host metadata. Only numerical field storage is moved.
Adapt.adapt_structure(to, U::SpectralField) = SpectralField(Adapt.adapt(to, parent(U)), grid(U))
Adapt.adapt_structure(to, u::PhysicalField) = PhysicalField(Adapt.adapt(to, parent(u)), grid(u))
Adapt.adapt_structure(to, U::VectorField) = VectorField(Adapt.adapt(to, U.components))
Adapt.adapt_structure(to, G::GradientField) = GradientField(Adapt.adapt(to, G.components))
Adapt.adapt_structure(to, s::State) =
    State(Adapt.adapt(to, velocity(s)), Adapt.adapt(to, stagepressure(s)))

function Adapt.adapt_structure(to, h::BatchedInfluenceSolver)
    BatchedInfluenceSolver(
        Adapt.adapt(to, h.pressure),
        Adapt.adapt(to, h.velocity),
        Adapt.adapt(to, h.responses),
        Adapt.adapt(to, h.influence),
        Adapt.adapt(to, h.sigma),
        Adapt.adapt(to, h.shift),
        Adapt.adapt(to, h.kx),
        Adapt.adapt(to, h.kz),
        Adapt.adapt(to, h.boundary),
        Adapt.adapt(to, h.work),
    )
end

function Adapt.adapt_structure(to, scheme::CNRK2)
    CNRK2(
        scheme.nu,
        scheme.dt,
        Adapt.adapt(to, scheme.solvers),
        Adapt.adapt(to, scheme.Q),
        Adapt.adapt(to, scheme.N),
        Adapt.adapt(to, scheme.R),
        Adapt.adapt(to, scheme.baseflow),
        Adapt.adapt(to, scheme.basecurvature),
    )
end

# Preserve the active storage backend when constructing auxiliary solvers.
_same_storage(object, ::SpectralField) = object
