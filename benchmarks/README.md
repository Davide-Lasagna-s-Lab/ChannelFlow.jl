# Transform benchmarks

`chebyshev.jl` compares forward and inverse wall-normal transforms using
GEMM and FFTW. It increases `Nx = Nz = Ny - 1`, so the number of Fourier
columns grows with the wall-normal resolution. This measures the Chebyshev
plans alone, not the periodic FFT or a complete DNS step.

Use the existing profiling environment, adding PyPlot once:

```sh
julia --project=perf -e 'using Pkg; Pkg.add("PyPlot")'
julia --startup-file=no --project=perf benchmarks/chebyshev.jl
open benchmarks/results/chebyshev.png
```

Choose sizes and BLAS thread counts through environment variables:

```sh
NY=17,33,65 BLAS_THREADS=1,2,4 julia --project=perf benchmarks/chebyshev.jl
```

BLAS threading is enabled with `BLAS.set_num_threads`; it does not require
multiple Julia threads. FFTW uses one thread throughout. The script restores
both thread settings afterwards. Avoid running concurrent benchmarks.

Plans use FFTW.MEASURE. Compilation and planning are excluded from timing.
Each backend is checked against FFTW on identical input, then measured with
nine batches; the CSV reports median milliseconds and bytes per call.
Plots (PNG and SVG), CSV data and environment details go in `results/`.
