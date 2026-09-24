# Fields and transforms

Directions and array axes are different concepts:

| Object | Array order | Meaning |
|---|---|---|
| `Grid(Nx, Ny, Nz, Lx, Lz)` | constructor arguments `(x,y,z)` | resolved physical point counts |
| `PhysicalField` | `(x,z,y)` | periodic samples and descending Lobatto nodes |
| `SpectralField` | `(kx,kz,n)` | Fourier amplitudes and ordinary Chebyshev coefficients |
| `VectorField` components | `(u,v,w)` | streamwise, wall-normal, spanwise velocity |
| `State` | velocity and stage pressure | data needed to continue CNRK2 |

With `Nxh = floor(Nx/2)+1`, a spectral field has shape `(Nxh,Nz,Ny)`.
`reshape(parent(U), Nxh*Nz, Ny)` is a **zero-copy** matrix: each row is one
Fourier system. This is the batched Helmholtz layout. Adjacent CPU SIMD lanes
or GPU threads process adjacent systems without assembling a packed RHS.

The x half-spectrum stores nonnegative wavenumbers. The z axis follows FFT
order: zero, positive, then negative wavenumbers. The integer pair `(kx,kz)`
corresponds to physical wavenumbers `α = 2π*kx/Lx`, `β = 2π*kz/Lz`.
Chebyshev degree `n` is stored at index `n+1`. Select a wall-normal coefficient profile with `U[ix,iz,:]`. The grid
constructor takes `(Nx,Ny,Nz)`, whereas array sizes follow their storage axes.

`physicalsize(grid, Padded())` and `spectralsize(grid, NotPadded())` return
array shapes. Physical fields constructed from a function default to padded
storage; the function is always called as `f(x,y,z)`.

## Transform interface

`FFT(u)` and `IFFT(U)` allocate their outputs. Repeated transforms should use
`ForwardFFT!(u)` and `InverseFFT!(U)` plans and preallocated destinations:

```julia
u = PhysicalField(grid, (x,y,z) -> (1-y^2)*cos(x))
U = SpectralField(grid)
forward = ForwardFFT!(u; flags=FFTW.ESTIMATE)
backward = InverseFFT!(U; flags=FFTW.ESTIMATE)
forward(U, u)
backward(u, U)
```

Use periodic functions compatible with your chosen `Lx,Lz`; `cos(x)` assumes
`Lx=2π` (or an integer multiple). Both allocating and planned transforms also
accept vector fields. `parent(field)` exposes the storage array; changing it
changes the field. The grid is metadata and is not duplicated on the GPU.
