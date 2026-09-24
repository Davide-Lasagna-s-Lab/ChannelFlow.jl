# GPU execution

For a complete runnable simulation, start with the
[GPU quick start](quickstart.md#GPU-quick-start). This page explains device
transfer, transform implementation and performance considerations.

Install CUDA.jl in the environment once. Construct the problem and initial
condition on the CPU, then transfer them explicitly:

```julia
using CUDA, Adapt
CUDA.allowscalar(false)

gpu_problem = adapt(CuArray, problem)
gpu_state = adapt(CuArray, state)

for j = 0:99
    step!(gpu_problem.scheme, gpu_problem.nlterm,
          velocity(gpu_state), stagepressure(gpu_state), j*0.01;
          forcing=gpu_problem.forcing, gpu_problem.constraint...)
end
CUDA.synchronize()             # only needed here to wait for completion
state = adapt(Array, gpu_state) # explicit download for storage or CPU analysis
```

Adaptation transfers the cached factors and creates new device transform
plans. It preserves `Float64`/`ComplexF64` precision. A timestep contains no
full-field host transfers. The constant-bulk-velocity interface currently retrieves mean
mode information and its two pressure gradients on the host each stage.
GPU timings must include a final `CUDA.synchronize()`.

The device Chebyshev transform uses cuFFT on an even extension. CPU Chebyshev
transforms use FFTW's DCT-I. Periodic transforms use FFTW on CPU and cuFFT on
GPU. The field's storage determines which implementation is used; no backend
selection is needed. CUDA is optional and is loaded through a Julia extension.

## Data ownership and performance

Construct and transfer a problem once, outside the integration loop. Its cached
Helmholtz factors, homogeneous responses and work arrays are reused by every
stage. Recreating the problem or transferring full fields each step defeats
this design. A separate problem is needed for each concurrently evolving trajectory.

A spectral array has shape `(k,l,n)`. A zero-copy matrix view places independent
Fourier systems in rows and Chebyshev coefficients in columns. Adjacent device
threads therefore access adjacent systems during coefficient recurrences.

The GPU even-extension preparation combines inverse endpoint weighting with its
copy operation. Forward normalization combines transform scaling, Chebyshev
endpoint weights and Fourier Nyquist filtering. These fused kernels reduce
launch overhead and repeated memory accesses without changing the numerical method.

The factor and workspace memory grows with all three resolved dimensions, and
the nonlinear evaluation additionally stores padded physical fields. Check free
device memory before selecting a large grid. The [benchmark protocol](benchmarks.md)
excludes setup and monitoring; it does not predict memory capacity for every case.

## Common pitfalls

- `CUDA.allowscalar(false)` catches accidental host scalar indexing of device arrays.
- GPU kernels run asynchronously. Synchronize when timing or before consuming results on the host.
- The standard fixed-gradient step keeps full fields on device. Fixed-flux driving requires small host-visible mean-mode reductions.
- Custom forcing must add to the supplied device RHS and preserve input velocity.
- Observables such as energy and pressure reconstruction may allocate or synchronize; sample them at a suitable cadence.
