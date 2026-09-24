# Fields and transforms

## Coordinates, components and storage order

Directions and array axes are different concepts:

| Object | Order | Meaning |
|---|---|---|
| `Grid(Nx, Ny, Nz, Lx, Lz)` | constructor arguments `(x,y,z)` | resolved physical point counts, before padding |
| `PhysicalField` | array indices `(ix,iz,iy)` | periodic samples and descending Lobatto nodes |
| `SpectralField` | array indices `(ix,iz,n+1)` | Fourier amplitudes and ordinary Chebyshev coefficients |
| `VectorField` components | `(u,v,w)` | streamwise, wall-normal, spanwise components |
| `State` | velocity and stage pressure | data needed to continue CNRK2 |

Julia array indices start at **1**. Fourier mode numbers and Chebyshev degrees
start at **0**; they are not array indices. The examples below use lowercase
`u` for physical values and uppercase `U` for spectral coefficients.

## Construct a physical scalar field from a function

Pass a function of the physical coordinates, always in the order `f(x,y,z)`:

```@example fields
using ChannelFlow, FFTW

Nx, Ny, Nz = 16, 17, 16
Lx, Lz = 2π, π
grid = Grid(Nx, Ny, Nz, Lx, Lz)
α, β = 2π/Lx, 2π/Lz

f(x, y, z) = (1-y^2)*exp(cos(α*x) + cos(β*z))
u = PhysicalField(grid, f)

@assert size(u) == (24, 24, 17) # padded x, padded z, unchanged y
nothing # hide
```

The constructor evaluates `f` at every grid point and stores the resulting
real values. It does not store the function or transform its values to
coefficients. The factors `α` and `β` make the function periodic on the chosen
domain lengths. The example vanishes at both walls.

`PhysicalField(grid, f)` defaults to **3/2-padded** periodic coordinates,
which are required by the standard transform interface. The following
explicit form is equivalent:

```julia
u = PhysicalField(grid, f, Padded())
```

Use `PhysicalField(grid, f, NotPadded())` only when you specifically need samples
on the resolved grid. Such an array is not the padded input expected by `FFT`.
`PhysicalField(grid)` allocates a zero padded scalar field.

## Index physical values and take slices

The stored value `u[ix,iz,iy]` is the sample at `(x[ix],y[iy],z[iz])`.
Notice that **the function arguments are `(x,y,z)` but the array axes are
`(x,z,y)`**:

```@example fields
ix, iz, iy = 3, 5, 9

# points returns broadcast-shaped coordinates in (y,x,z) order.
y, x, z = points(grid, Padded())
value = u[ix, iz, iy]
@assert value ≈ f(x[ix,1,1], y[1,1,iy], z[1,iz,1])

profile = u[ix, iz, :]          # copy of a physical profile along y
section = u[ix, :, :]           # copy of a (z,y) plane at fixed x
midplane = u[:, :, (Ny+1)÷2]    # copy of the (x,z) plane at y ≈ 0

# A view shares the original storage; an ordinary slice above copies it.
profile_view = @view u[ix, iz, :]

# Scalar assignment and broadcasting modify a field in place.
u_copy = copy(u)
u_copy[ix, iz, iy] = 0.0
u_copy .*= 0.5
nothing # hide
```

The wall-normal points descend from `y=+1` at `iy=1` to `y=-1` at `iy=Ny`.
For odd `Ny`, the middle index is the centreplane. Periodic coordinates omit
the repeated endpoint: neither `x=Lx` nor `z=Lz` is stored.

`parent(u)` returns the underlying array **without copying**. Editing it or a
view changes `u`; use `copy(u)` for an independent field.

## Transform between physical values and spectral coefficients

The allocating transforms are

```@example fields
U = FFT(u)
u_reconstructed = IFFT(U)

@assert size(U) == (Nx÷2 + 1, Nz, Ny) # (9,16,17)
@assert size(u_reconstructed) == size(u)
reconstruction_error = maximum(abs, parent(u_reconstructed) .- parent(u))
nothing # hide
```

`FFT` performs **both** the Fourier transforms in `x,z` and the Chebyshev
transform in `y`. It truncates the padded Fourier spectrum to the resolved
modes and sets the excluded Nyquist modes to zero. `IFFT` reconstructs a
real field on the padded physical grid. Neither allocating operation changes
its input.

The analytic function above contains infinitely many Fourier harmonics.
Consequently, `IFFT(FFT(u))` reconstructs its retained spectral approximation;
it need not reproduce all original samples exactly. Band-limited fields with
no excluded Nyquist content round-trip to floating-point accuracy.

## Index spectral coefficients and Fourier profiles

With ``\alpha=2\pi/L_x`` and ``\beta=2\pi/L_z``, `U` represents

```math
u(x,y,z)=\sum_{k,l}\sum_{n=0}^{N_y-1}
 a_{k,l,n}T_n(y)e^{i(\alpha k x+\beta l z)}.
```

The first axis stores nonnegative streamwise modes, with `ix=k+1`.
The second axis uses FFT order: zero, positive, then negative spanwise modes.
For `Nz=16`, its mode labels are `0,1,…,7,8,-7,…,-1`; the Nyquist plane labelled
`8` is excluded. For a retained signed spanwise mode, `iz=mod(l,Nz)+1`.
The third index is `n+1`, where `n` is the Chebyshev degree:

```@example fields
k, l, n = 2, -3, 4
ix = k + 1
iz = mod(l, Nz) + 1
coefficient = U[ix, iz, n+1] # multiplies T₄(y) exp(i(2αx-3βz))

mode_coefficients = U[ix, iz, :] # all Chebyshev coefficients of this mode
mean_coefficients = U[1, 1, :]   # coefficients of the instantaneous plane mean
mode_view = @view U[ix, iz, :]  # same profile without copying
nothing # hide
```

**A spectral slice `U[ix,iz,:]` contains coefficients, not values at the
wall-normal points.** Evaluating that Fourier amplitude requires the sum
``\widehat u_{k,l}(y)=\sum_n a_{k,l,n}T_n(y)``. To reconstruct the
complete physical field, use `IFFT(U)` before taking physical slices.

Negative streamwise modes are omitted using the reality condition
``a_{-k,-l,n}=\overline{a_{k,l,n}}``. Directly editing coefficients
must respect this condition on the self-conjugate planes, as well as the
excluded Nyquist modes, if the field is to represent real physical data.

## Construct and transform a vector field

Construct three scalar fields and wrap them in physical component order.
The following example creates a streamwise-independent roll:

```@example fields
amplitude = 0.1
u₀(x, y, z) = 0.0
v₀(x, y, z) = amplitude*(1-y^2)^2*cos(β*z)
w₀(x, y, z) = 4amplitude/β*y*(1-y^2)*sin(β*z)

uvec = VectorField(
    PhysicalField(grid, u₀),
    PhysicalField(grid, v₀),
    PhysicalField(grid, w₀),
)

Uvec = FFT(uvec)              # three spectral components
uvec_reconstructed = IFFT(Uvec)

normal_value = uvec[2][3, 5, 9] # wall-normal component at a physical point
streamwise_mode = Uvec[1][1, 2, :] # u coefficients for k=0, l=1
nothing # hide
```

`uvec[1]`, `uvec[2]` and `uvec[3]` are the streamwise, wall-normal and spanwise
scalar fields. The component is selected first, then its spatial or spectral
indices. `VectorField(u,v,w)` wraps these fields without copying them;
`copy(uvec)` copies all three components. You can also iterate with
`for component in uvec`.

Transforming a vector field does not project it onto the divergence-free
subspace or construct its pressure. The roll above satisfies continuity and
no slip analytically. For a general initial velocity, use the projection and
pressure procedures in [Initialisation and pressure](pressure.md).
For an existing `State`, `velocity(state)` gives the spectral vector field
and `stagepressure(state)` the scalar spectral stage pressure.

## Reuse plans and output buffers

For repeated transforms, allocate destinations and plans once:

```@example fields
U_buffer = SpectralField(grid)
u_buffer = PhysicalField(grid)
forward = ForwardFFT!(u; flags=FFTW.ESTIMATE)
backward = InverseFFT!(U_buffer; flags=FFTW.ESTIMATE)

forward(U_buffer, u)          # destination first; overwrite U_buffer
backward(u_buffer, U_buffer)  # overwrite u_buffer

# The same scalar plans can transform compatible vector components.
Uvec_buffer = VectorField(SpectralField(grid))
uvec_buffer = VectorField(PhysicalField(grid))
forward(Uvec_buffer, uvec)
backward(uvec_buffer, Uvec_buffer)
nothing # hide
```

Plans are tied to compatible grid dimensions, storage and element types.
The input is preserved during execution; destinations are overwritten.
`ForwardFFT(grid)` and `InverseFFT(grid)` are alternative plan constructors
that take a grid directly. See [GPU execution](gpu.md) for device fields;
the scalar indexing examples above are intended for CPU arrays.

## Matrix view for batched operations

With `Nxh = floor(Nx/2)+1`, a spectral field has shape `(Nxh,Nz,Ny)`.
`reshape(parent(U), Nxh*Nz, Ny)` is a **zero-copy** matrix: each row is one
Fourier system and each column a Chebyshev degree. This is the batched
Helmholtz layout. Adjacent CPU SIMD lanes or GPU threads process adjacent
systems without assembling a packed right-hand side.

`physicalsize(grid, Padded())` and `spectralsize(grid, NotPadded())` return
array shapes in storage order. The grid constructor takes `(Nx,Ny,Nz)`,
whereas array sizes follow `(x,z,y)` and `(k,l,n)` respectively.
