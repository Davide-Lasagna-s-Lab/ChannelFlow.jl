export NoForcing

"""
    NoForcing()

Callable zero forcing. `forcing(t, U, R)` leaves the accumulated spectral
RHS `R` and velocity `U` unchanged.
"""
struct NoForcing end

(::NoForcing)(t, U, R) = nothing
