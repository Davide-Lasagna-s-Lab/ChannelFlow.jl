# Common initial condition for timing and profiling runs.
function initial(g, p)
    # A smooth three-dimensional perturbation excites the complete nonlinear
    # path. Project once and reconstruct a consistent initial pressure;
    # initialization and planning are outside every timing interval.
    u=VectorField(
        PhysicalField(g, (x, y, z)->0.1*(1-y^2)*cos(x)*sin(z)),
        PhysicalField(g, (x, y, z)->0.05*(1-y^2)^2*cos(z)),
        PhysicalField(g, (x, y, z)->0.2*y*(1-y^2)*sin(z)),
    )
    U=project!(FFT(u), p)
    State(U, pressure(U, p))
end
