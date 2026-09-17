export zero_state, random_state

#//////////////////////////////////////////////////////////////////////////////#
#///                         ZERO STATE ALLOCATION                          ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    zero_state(grid::Grid)

Allocate `State(U, Q)` with zero spectral perturbation velocity and stage
pressure. Access these fields with `velocity(state)` and
`stagepressure(state)`.
A laminar base profile belongs to the problem, so it is not added here.
Zero stage pressure need not be a consistent initial value for time stepping.
"""
zero_state(grid::Grid) = State(VectorField(SpectralField(grid)), SpectralField(grid))

#//////////////////////////////////////////////////////////////////////////////#
#///                     RANDOM VELOCITY INITIALIZATION                     ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    random_state(problem::ChannelFlowProblem, epsilon)

Create three random physical velocity components on the padded grid, transform
and project them onto the divergence-free, no-slip space. Return a `State`
with the physical or modified pressure appropriate to `problem`.
`epsilon` scales Gaussian samples before projection; it does not prescribe
the final RMS or kinetic energy. Use Julia's default RNG; call `Random.seed!`
beforehand for reproducibility.
"""
function random_state(problem::ChannelFlowProblem, epsilon::Real)
    # sanity check
    epsilon >= 0 || throw(ArgumentError("epsilon must be non-negative"))
    
    # random function
    eps_rand(x, y, z) = epsilon * Random.randn()

    # create non-zero-divergence velocity field in physical space
    u = VectorField(PhysicalField(problem.grid, eps_rand, Padded()),
                    PhysicalField(problem.grid, eps_rand, Padded()),
                    PhysicalField(problem.grid, eps_rand, Padded()))
    
    # transform to spectral space and project 
    U = project!(FFT(u), problem)

    # obtain pressure field
    P = pressure(U, problem)

    return State(U, P)
end
