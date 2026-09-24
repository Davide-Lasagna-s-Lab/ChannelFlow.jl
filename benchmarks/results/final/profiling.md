# Where timestep execution spends time

Profiles cover warmed complete CNRK2 steps, separately from throughput timings.
CPU percentages classify sampled DNS call stacks into exclusive categories. They are sampling estimates, not exact phase timers; FFT worker activity is not a Julia call stack. Consult the raw call tree for context.

GPU percentages are fractions of summed device activity duration. They exclude host launch overhead and idle gaps; overlapping activities must not be interpreted as additive wall time. Raw logs also contain CUDA host API activity.

## profile-cpu-1-N128

[Raw data](profile-cpu-1-N128.csv) · [Full profiler output](profile-cpu-1-N128.txt)

| Kernel / category | Share | Samples |
|---|---:|---:|
| `Fourier transforms` | 30.4% | 10004 |
| `Chebyshev transforms` | 28.2% | 9275 |
| `Influence and tau` | 9.3% | 3044 |
| `Padding and normalization` | 9.1% | 3001 |
| `Nonlinear products` | 7.9% | 2599 |
| `Helmholtz solves` | 5.6% | 1850 |
| `Spectral derivatives` | 5.2% | 1718 |
| `Stage assembly and runtime` | 4.3% | 1414 |

## profile-cpu-1-N256

[Raw data](profile-cpu-1-N256.csv) · [Full profiler output](profile-cpu-1-N256.txt)

| Kernel / category | Share | Samples |
|---|---:|---:|
| `Fourier transforms` | 32.7% | 89514 |
| `Chebyshev transforms` | 26.0% | 71085 |
| `Influence and tau` | 11.5% | 31561 |
| `Padding and normalization` | 8.4% | 23070 |
| `Nonlinear products` | 6.8% | 18720 |
| `Helmholtz solves` | 5.9% | 16035 |
| `Spectral derivatives` | 4.7% | 12943 |
| `Stage assembly and runtime` | 3.9% | 10774 |

## profile-cpu-1-N64

[Raw data](profile-cpu-1-N64.csv) · [Full profiler output](profile-cpu-1-N64.txt)

| Kernel / category | Share | Samples |
|---|---:|---:|
| `Chebyshev transforms` | 30.1% | 997 |
| `Fourier transforms` | 24.6% | 817 |
| `Influence and tau` | 9.9% | 328 |
| `Nonlinear products` | 9.7% | 321 |
| `Spectral derivatives` | 6.8% | 224 |
| `Padding and normalization` | 6.6% | 219 |
| `Helmholtz solves` | 6.2% | 207 |
| `Stage assembly and runtime` | 6.1% | 202 |

## profile-gpu-N128

[Raw data](profile-gpu-N128.csv) · [Full profiler output](profile-gpu-N128.txt)

| Kernel / category | Share | Total device ms | Calls |
|---|---:|---:|---:|
| `_solve_kernel_(BatchedHelmoltzSolver<Float64, 8320, BatchedQuasiTridiagonal<Float64, 8320, 65, CuDeviceArray<Float64, 2, 1>>, BatchedQuasiTridiagonal<Float64, 8320, 64, CuDeviceArray<Float64, 2, 1>>, CuDeviceArray<Float64, 1, 1>>, CuDeviceArray<Complex<Float64>, 2, 1>, CuDeviceArray<Complex<Float64>, 2, 1>, CuDeviceArray<Float64, 1, 1>, CuDeviceArray<Float64, 1, 1>)` | 19.3% | 75.198 | 240 |
| `_derivative_(CuDeviceArray<Complex<Float64>, 2, 1>, CuDeviceArray<Complex<Float64>, 2, 1>, Bool, Bool, Int64, Int64, Float64, Float64)` | 10.9% | 42.442 | 540 |
| `void regular_fft<192u, EPT<12u, 16u>, 16u, 2u, (padding_t)6, (twiddle_t)0, (loadstore_modifier_t)2, (layout_t)1, unsigned int, double, HostConfigPlaceholder>(kernel_arguments_t<unsigned int>)` | 8.5% | 32.953 | 540 |
| `gpu_broadcast_kernel_cartesian(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndices<3, Tuple<OneTo<Int64>, OneTo<Int64>, OneTo<Int64>>>, NDRange<3, DynamicSize, DynamicSize, CartesianIndices<3, Tuple<OneTo<Int64>, OneTo<Int64>, OneTo<Int64>>>, CartesianIndices<3, Tuple<OneTo<Int64>, OneTo<Int64>, OneTo<Int64>>>>>, CuDeviceArray<Float64, 3, 1>, Broadcasted<CuArrayStyle<3, DeviceMemory>, Tuple<OneTo<Int64>, OneTo<Int64>, OneTo<Int64>>, _, Tuple<Broadcasted<CuArrayStyle<3, DeviceMemory>, void, _, Tuple<Extruded<CuDeviceArray<Float64, 3, 1>, Tuple<Bool, Bool, Bool>, Tuple<Int64, Int64, Int64>>, Extruded<CuDeviceArray<Float64, 3, 1>, Tuple<Bool, Bool, Bool>, Tuple<Int64, Int64, Int64>>>>, Broadcasted<CuArrayStyle<3, DeviceMemory>, void, _, Tuple<Extruded<CuDeviceArray<Float64, 3, 1>, Tuple<Bool, Bool, Bool>, Tuple<Int64, Int64, Int64>>, Extruded<CuDeviceArray<Float64, 3, 1>, Tuple<Bool, Bool, Bool>, Tuple<Int64, Int64, Int64>>>>>>)` | 5.8% | 22.660 | 180 |
| `void regular_fft<256u, EPT<16u>, 8u, 4u, (padding_t)14, (twiddle_t)0, (loadstore_modifier_t)2, (layout_t)1, unsigned int, double, HostConfigPlaceholder>(kernel_arguments_t<unsigned int>)` | 5.4% | 21.198 | 540 |
| `_cheb_extend_(CuDeviceArray<Complex<Float64>, 2, 1>, CuDeviceArray<Complex<Float64>, 2, 1>, Bool)` | 5.3% | 20.707 | 540 |
| `void regular_fft_c2r<192u, EPT<12u, 16u>, 4u, 10u, (padding_t)6, (twiddle_t)0, (loadstore_modifier_t)2, (layout_t)0, unsigned int, double, HostConfigPlaceholder>(kernel_arguments_t<unsigned int>)` | 5.2% | 20.092 | 360 |
| `[copy device to device memory]` | 4.7% | 18.229 | 480 |
| `_influence_(CuDeviceArray<Complex<Float64>, 2, 1>, CuDeviceArray<Complex<Float64>, 2, 1>, CuDeviceArray<Float64, 2, 1>, CuDeviceArray<Float64, 2, 1>, CuDeviceArray<Float64, 2, 1>, CuDeviceArray<Float64, 2, 1>, CuDeviceArray<Float64, 1, 1>, CuDeviceArray<Float64, 1, 1>, CuDeviceArray<Float64, 1, 1>, CuDeviceArray<Float64, 1, 1>)` | 3.9% | 15.020 | 60 |
| `_tau_(CuDeviceArray<Complex<Float64>, 2, 1>, CuDeviceArray<Complex<Float64>, 2, 1>, CuDeviceArray<Complex<Float64>, 2, 1>, CuDeviceArray<Complex<Float64>, 2, 1>, CuDeviceArray<Float64, 2, 1>, CuDeviceArray<Float64, 2, 1>, CuDeviceArray<Float64, 1, 1>, CuDeviceArray<Float64, 1, 1>, CuDeviceArray<Float64, 1, 1>)` | 3.0% | 11.757 | 60 |
| `gpu_broadcast_kernel_cartesian(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndices<3, Tuple<OneTo<Int64>, OneTo<Int64>, OneTo<Int64>>>, NDRange<3, DynamicSize, DynamicSize, CartesianIndices<3, Tuple<OneTo<Int64>, OneTo<Int64>, OneTo<Int64>>>, CartesianIndices<3, Tuple<OneTo<Int64>, OneTo<Int64>, OneTo<Int64>>>>>, SubArray<Complex<Float64>, 3, CuDeviceArray<Complex<Float64>, 3, 1>, Tuple<UnitRange<Int64>, UnitRange<Int64>, Slice<OneTo<Int64>>>, false>, Broadcasted<CuArrayStyle<3, DeviceMemory>, Tuple<OneTo<Int64>, OneTo<Int64>, OneTo<Int64>>, _, Tuple<Float64, Extruded<SubArray<Complex<Float64>, 3, CuDeviceArray<Complex<Float64>, 3, 1>, Tuple<Slice<OneTo<Int64>>, UnitRange<Int64>, Slice<OneTo<Int64>>>, false>, Tuple<Bool, Bool, Bool>, Tuple<Int64, Int64, Int64>>>>)` | 3.0% | 11.581 | 720 |
| `gpu_broadcast_kernel_cartesian(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndices<2, Tuple<OneTo<Int64>, OneTo<Int64>>>, NDRange<2, DynamicSize, DynamicSize, CartesianIndices<2, Tuple<OneTo<Int64>, OneTo<Int64>>>, CartesianIndices<2, Tuple<OneTo<Int64>, OneTo<Int64>>>>>, CuDeviceArray<Complex<Float64>, 2, 1>, Broadcasted<CuArrayStyle<2, DeviceMemory>, Tuple<OneTo<Int64>, OneTo<Int64>>, identity, Tuple<Extruded<CuDeviceArray<Complex<Float64>, 2, 1>, Tuple<Bool, Bool>, Tuple<Int64, Int64>>>>)` | 2.9% | 11.389 | 540 |
| `_mean_(CuDeviceArray<Complex<Float64>, 2, 1>, CuDeviceArray<Complex<Float64>, 2, 1>, CuDeviceArray<Complex<Float64>, 2, 1>, CuDeviceArray<Complex<Float64>, 2, 1>, CuDeviceArray<Complex<Float64>, 2, 1>, CuDeviceArray<Float64, 1, 1>, Float64, CuDeviceArray<Float64, 1, 1>, Float64, Float64, Float64, Float64, Bool)` | 2.9% | 11.383 | 60 |
| `gpu_broadcast_kernel_cartesian(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndices<3, Tuple<OneTo<Int64>, OneTo<Int64>, OneTo<Int64>>>, NDRange<3, DynamicSize, DynamicSize, CartesianIndices<3, Tuple<OneTo<Int64>, OneTo<Int64>, OneTo<Int64>>>, CartesianIndices<3, Tuple<OneTo<Int64>, OneTo<Int64>, OneTo<Int64>>>>>, CuDeviceArray<Complex<Float64>, 3, 1>, Broadcasted<CuArrayStyle<3, DeviceMemory>, Tuple<OneTo<Int64>, OneTo<Int64>, OneTo<Int64>>, _, Tuple<Broadcasted<CuArrayStyle<3, DeviceMemory>, void, _, Tuple<Broadcasted<CuArrayStyle<3, DeviceMemory>, void, _, Tuple<Broadcasted<CuArrayStyle<3, DeviceMemory>, void, _, Tuple<Float64, Extruded<CuDeviceArray<Complex<Float64>, 3, 1>, Tuple<Bool, Bool, Bool>, Tuple<Int64, Int64, Int64>>>>, Broadcasted<CuArrayStyle<3, DeviceMemory>, void, _, Tuple<Float64, Extruded<CuDeviceArray<Complex<Float64>, 3, 1>, Tuple<Bool, Bool, Bool>, Tuple<Int64, Int64, Int64>>>>>>, Extruded<CuDeviceArray<Complex<Float64>, 3, 1>, Tuple<Bool, Bool, Bool>, Tuple<Int64, Int64, Int64>>>>, Broadcasted<CuArrayStyle<3, DeviceMemory>, void, _, Tuple<Float64, Extruded<CuDeviceArray<Complex<Float64>, 3, 1>, Tuple<Bool, Bool, Bool>, Tuple<Int64, Int64, Int64>>>>>>)` | 2.7% | 10.517 | 180 |
| `gpu_fill_kernel_(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndices<1, Tuple<OneTo<Int64>>>, NDRange<1, DynamicSize, DynamicSize, CartesianIndices<1, Tuple<OneTo<Int64>>>, CartesianIndices<1, Tuple<OneTo<Int64>>>>>, CuDeviceArray<Complex<Float64>, 3, 1>, Complex<Float64>)` | 2.7% | 10.437 | 360 |

Top 15 kernels shown; percentages use all captured device activities.

## profile-gpu-N256

[Raw data](profile-gpu-N256.csv) · [Full profiler output](profile-gpu-N256.txt)

| Kernel / category | Share | Total device ms | Calls |
|---|---:|---:|---:|
| `void regular_fft_c2r<384u, EPT<4u, 6u>, 2u, 6u, (padding_t)14, (twiddle_t)0, (loadstore_modifier_t)2, (layout_t)0, unsigned int, double, HostConfigPlaceholder>(kernel_arguments_t<unsigned int>)` | 10.0% | 232.605 | 360 |
| `void regular_fft<384u, EPT<8u, 12u>, 8u, 2u, (padding_t)14, (twiddle_t)0, (loadstore_modifier_t)2, (layout_t)1, unsigned int, double, HostConfigPlaceholder>(kernel_arguments_t<unsigned int>)` | 9.9% | 229.365 | 540 |
| `_solve_kernel_(BatchedHelmoltzSolver<Float64, 33024, BatchedQuasiTridiagonal<Float64, 33024, 129, CuDeviceArray<Float64, 2, 1>>, BatchedQuasiTridiagonal<Float64, 33024, 128, CuDeviceArray<Float64, 2, 1>>, CuDeviceArray<Float64, 1, 1>>, CuDeviceArray<Complex<Float64>, 2, 1>, CuDeviceArray<Complex<Float64>, 2, 1>, CuDeviceArray<Float64, 1, 1>, CuDeviceArray<Float64, 1, 1>)` | 9.4% | 218.599 | 240 |
| `void regular_fft<512u, EPT<8u>, 8u, 2u, (padding_t)6, (twiddle_t)0, (loadstore_modifier_t)2, (layout_t)1, unsigned int, double, HostConfigPlaceholder>(kernel_arguments_t<unsigned int>)` | 9.0% | 207.576 | 540 |
| `gpu_broadcast_kernel_cartesian(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndices<3, Tuple<OneTo<Int64>, OneTo<Int64>, OneTo<Int64>>>, NDRange<3, DynamicSize, DynamicSize, CartesianIndices<3, Tuple<OneTo<Int64>, OneTo<Int64>, OneTo<Int64>>>, CartesianIndices<3, Tuple<OneTo<Int64>, OneTo<Int64>, OneTo<Int64>>>>>, CuDeviceArray<Float64, 3, 1>, Broadcasted<CuArrayStyle<3, DeviceMemory>, Tuple<OneTo<Int64>, OneTo<Int64>, OneTo<Int64>>, _, Tuple<Broadcasted<CuArrayStyle<3, DeviceMemory>, void, _, Tuple<Extruded<CuDeviceArray<Float64, 3, 1>, Tuple<Bool, Bool, Bool>, Tuple<Int64, Int64, Int64>>, Extruded<CuDeviceArray<Float64, 3, 1>, Tuple<Bool, Bool, Bool>, Tuple<Int64, Int64, Int64>>>>, Broadcasted<CuArrayStyle<3, DeviceMemory>, void, _, Tuple<Extruded<CuDeviceArray<Float64, 3, 1>, Tuple<Bool, Bool, Bool>, Tuple<Int64, Int64, Int64>>, Extruded<CuDeviceArray<Float64, 3, 1>, Tuple<Bool, Bool, Bool>, Tuple<Int64, Int64, Int64>>>>>>)` | 7.5% | 173.379 | 180 |
| `_cheb_extend_(CuDeviceArray<Complex<Float64>, 2, 1>, CuDeviceArray<Complex<Float64>, 2, 1>, Bool)` | 7.4% | 170.944 | 540 |
| `[copy device to device memory]` | 5.8% | 133.769 | 480 |
| `_derivative_(CuDeviceArray<Complex<Float64>, 2, 1>, CuDeviceArray<Complex<Float64>, 2, 1>, Bool, Bool, Int64, Int64, Float64, Float64)` | 5.2% | 120.563 | 540 |
| `void regular_fft_r2c<384u, EPT<4u, 6u>, 2u, 6u, (padding_t)14, (twiddle_t)0, (loadstore_modifier_t)2, (layout_t)0, unsigned int, double, HostConfigPlaceholder>(kernel_arguments_t<unsigned int>)` | 4.8% | 110.732 | 180 |
| `gpu_broadcast_kernel_cartesian(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndices<2, Tuple<OneTo<Int64>, OneTo<Int64>>>, NDRange<2, DynamicSize, DynamicSize, CartesianIndices<2, Tuple<OneTo<Int64>, OneTo<Int64>>>, CartesianIndices<2, Tuple<OneTo<Int64>, OneTo<Int64>>>>>, CuDeviceArray<Complex<Float64>, 2, 1>, Broadcasted<CuArrayStyle<2, DeviceMemory>, Tuple<OneTo<Int64>, OneTo<Int64>>, identity, Tuple<Extruded<CuDeviceArray<Complex<Float64>, 2, 1>, Tuple<Bool, Bool>, Tuple<Int64, Int64>>>>)` | 3.9% | 91.536 | 540 |
| `gpu_fill_kernel_(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndices<1, Tuple<OneTo<Int64>>>, NDRange<1, DynamicSize, DynamicSize, CartesianIndices<1, Tuple<OneTo<Int64>>>, CartesianIndices<1, Tuple<OneTo<Int64>>>>>, CuDeviceArray<Complex<Float64>, 3, 1>, Complex<Float64>)` | 3.5% | 80.690 | 360 |
| `gpu_broadcast_kernel_cartesian(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndices<3, Tuple<OneTo<Int64>, OneTo<Int64>, OneTo<Int64>>>, NDRange<3, DynamicSize, DynamicSize, CartesianIndices<3, Tuple<OneTo<Int64>, OneTo<Int64>, OneTo<Int64>>>, CartesianIndices<3, Tuple<OneTo<Int64>, OneTo<Int64>, OneTo<Int64>>>>>, CuDeviceArray<Complex<Float64>, 3, 1>, Broadcasted<CuArrayStyle<3, DeviceMemory>, Tuple<OneTo<Int64>, OneTo<Int64>, OneTo<Int64>>, _, Tuple<Broadcasted<CuArrayStyle<3, DeviceMemory>, void, _, Tuple<Broadcasted<CuArrayStyle<3, DeviceMemory>, void, _, Tuple<Broadcasted<CuArrayStyle<3, DeviceMemory>, void, _, Tuple<Float64, Extruded<CuDeviceArray<Complex<Float64>, 3, 1>, Tuple<Bool, Bool, Bool>, Tuple<Int64, Int64, Int64>>>>, Broadcasted<CuArrayStyle<3, DeviceMemory>, void, _, Tuple<Float64, Extruded<CuDeviceArray<Complex<Float64>, 3, 1>, Tuple<Bool, Bool, Bool>, Tuple<Int64, Int64, Int64>>>>>>, Extruded<CuDeviceArray<Complex<Float64>, 3, 1>, Tuple<Bool, Bool, Bool>, Tuple<Int64, Int64, Int64>>>>, Broadcasted<CuArrayStyle<3, DeviceMemory>, void, _, Tuple<Float64, Extruded<CuDeviceArray<Complex<Float64>, 3, 1>, Tuple<Bool, Bool, Bool>, Tuple<Int64, Int64, Int64>>>>>>)` | 3.2% | 75.024 | 180 |
| `gpu_broadcast_kernel_cartesian(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndices<3, Tuple<OneTo<Int64>, OneTo<Int64>, OneTo<Int64>>>, NDRange<3, DynamicSize, DynamicSize, CartesianIndices<3, Tuple<OneTo<Int64>, OneTo<Int64>, OneTo<Int64>>>, CartesianIndices<3, Tuple<OneTo<Int64>, OneTo<Int64>, OneTo<Int64>>>>>, SubArray<Complex<Float64>, 3, CuDeviceArray<Complex<Float64>, 3, 1>, Tuple<UnitRange<Int64>, UnitRange<Int64>, Slice<OneTo<Int64>>>, false>, Broadcasted<CuArrayStyle<3, DeviceMemory>, Tuple<OneTo<Int64>, OneTo<Int64>, OneTo<Int64>>, _, Tuple<Float64, Extruded<SubArray<Complex<Float64>, 3, CuDeviceArray<Complex<Float64>, 3, 1>, Tuple<Slice<OneTo<Int64>>, UnitRange<Int64>, Slice<OneTo<Int64>>>, false>, Tuple<Bool, Bool, Bool>, Tuple<Int64, Int64, Int64>>>>)` | 2.9% | 68.365 | 720 |
| `_influence_(CuDeviceArray<Complex<Float64>, 2, 1>, CuDeviceArray<Complex<Float64>, 2, 1>, CuDeviceArray<Float64, 2, 1>, CuDeviceArray<Float64, 2, 1>, CuDeviceArray<Float64, 2, 1>, CuDeviceArray<Float64, 2, 1>, CuDeviceArray<Float64, 1, 1>, CuDeviceArray<Float64, 1, 1>, CuDeviceArray<Float64, 1, 1>, CuDeviceArray<Float64, 1, 1>)` | 1.9% | 45.055 | 60 |
| `gpu_broadcast_kernel_cartesian(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndices<3, Tuple<OneTo<Int64>, OneTo<Int64>, OneTo<Int64>>>, NDRange<3, DynamicSize, DynamicSize, CartesianIndices<3, Tuple<OneTo<Int64>, OneTo<Int64>, OneTo<Int64>>>, CartesianIndices<3, Tuple<OneTo<Int64>, OneTo<Int64>, OneTo<Int64>>>>>, CuDeviceArray<Complex<Float64>, 3, 1>, Broadcasted<CuArrayStyle<3, DeviceMemory>, Tuple<OneTo<Int64>, OneTo<Int64>, OneTo<Int64>>, identity, Tuple<Extruded<CuDeviceArray<Complex<Float64>, 3, 1>, Tuple<Bool, Bool, Bool>, Tuple<Int64, Int64, Int64>>>>)` | 1.8% | 41.190 | 240 |

Top 15 kernels shown; percentages use all captured device activities.

## profile-gpu-N64

[Raw data](profile-gpu-N64.csv) · [Full profiler output](profile-gpu-N64.txt)

| Kernel / category | Share | Total device ms | Calls |
|---|---:|---:|---:|
| `_solve_kernel_(BatchedHelmoltzSolver<Float64, 2112, BatchedQuasiTridiagonal<Float64, 2112, 33, CuDeviceArray<Float64, 2, 1>>, BatchedQuasiTridiagonal<Float64, 2112, 32, CuDeviceArray<Float64, 2, 1>>, CuDeviceArray<Float64, 1, 1>>, CuDeviceArray<Complex<Float64>, 2, 1>, CuDeviceArray<Complex<Float64>, 2, 1>, CuDeviceArray<Float64, 1, 1>, CuDeviceArray<Float64, 1, 1>)` | 23.5% | 24.677 | 240 |
| `_derivative_(CuDeviceArray<Complex<Float64>, 2, 1>, CuDeviceArray<Complex<Float64>, 2, 1>, Bool, Bool, Int64, Int64, Float64, Float64)` | 13.6% | 14.326 | 540 |
| `_influence_(CuDeviceArray<Complex<Float64>, 2, 1>, CuDeviceArray<Complex<Float64>, 2, 1>, CuDeviceArray<Float64, 2, 1>, CuDeviceArray<Float64, 2, 1>, CuDeviceArray<Float64, 2, 1>, CuDeviceArray<Float64, 2, 1>, CuDeviceArray<Float64, 1, 1>, CuDeviceArray<Float64, 1, 1>, CuDeviceArray<Float64, 1, 1>, CuDeviceArray<Float64, 1, 1>)` | 5.6% | 5.906 | 60 |
| `void regular_fft_c2r<96u, EPT<4u, 6u>, 8u, 6u, (padding_t)6, (twiddle_t)0, (loadstore_modifier_t)2, (layout_t)0, unsigned int, double, HostConfigPlaceholder>(kernel_arguments_t<unsigned int>)` | 5.1% | 5.378 | 360 |
| `_tau_(CuDeviceArray<Complex<Float64>, 2, 1>, CuDeviceArray<Complex<Float64>, 2, 1>, CuDeviceArray<Complex<Float64>, 2, 1>, CuDeviceArray<Complex<Float64>, 2, 1>, CuDeviceArray<Float64, 2, 1>, CuDeviceArray<Float64, 2, 1>, CuDeviceArray<Float64, 1, 1>, CuDeviceArray<Float64, 1, 1>, CuDeviceArray<Float64, 1, 1>)` | 4.8% | 5.082 | 60 |
| `void regular_fft<128u, EPT<8u>, 16u, 4u, (padding_t)6, (twiddle_t)0, (loadstore_modifier_t)2, (layout_t)1, unsigned int, double, HostConfigPlaceholder>(kernel_arguments_t<unsigned int>)` | 4.5% | 4.740 | 540 |
| `_mean_(CuDeviceArray<Complex<Float64>, 2, 1>, CuDeviceArray<Complex<Float64>, 2, 1>, CuDeviceArray<Complex<Float64>, 2, 1>, CuDeviceArray<Complex<Float64>, 2, 1>, CuDeviceArray<Complex<Float64>, 2, 1>, CuDeviceArray<Float64, 1, 1>, Float64, CuDeviceArray<Float64, 1, 1>, Float64, Float64, Float64, Float64, Bool)` | 4.3% | 4.513 | 60 |
| `void regular_fft<96u, EPT<8u, 12u>, 16u, 3u, (padding_t)6, (twiddle_t)0, (loadstore_modifier_t)2, (layout_t)1, unsigned int, double, HostConfigPlaceholder>(kernel_arguments_t<unsigned int>)` | 4.1% | 4.332 | 540 |
| `gpu_broadcast_kernel_cartesian(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndices<3, Tuple<OneTo<Int64>, OneTo<Int64>, OneTo<Int64>>>, NDRange<3, DynamicSize, DynamicSize, CartesianIndices<3, Tuple<OneTo<Int64>, OneTo<Int64>, OneTo<Int64>>>, CartesianIndices<3, Tuple<OneTo<Int64>, OneTo<Int64>, OneTo<Int64>>>>>, SubArray<Complex<Float64>, 3, CuDeviceArray<Complex<Float64>, 3, 1>, Tuple<UnitRange<Int64>, UnitRange<Int64>, Slice<OneTo<Int64>>>, false>, Broadcasted<CuArrayStyle<3, DeviceMemory>, Tuple<OneTo<Int64>, OneTo<Int64>, OneTo<Int64>>, _, Tuple<Float64, Extruded<SubArray<Complex<Float64>, 3, CuDeviceArray<Complex<Float64>, 3, 1>, Tuple<Slice<OneTo<Int64>>, UnitRange<Int64>, Slice<OneTo<Int64>>>, false>, Tuple<Bool, Bool, Bool>, Tuple<Int64, Int64, Int64>>>>)` | 3.5% | 3.648 | 720 |
| `gpu_broadcast_kernel_cartesian(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndices<3, Tuple<OneTo<Int64>, OneTo<Int64>, OneTo<Int64>>>, NDRange<3, DynamicSize, DynamicSize, CartesianIndices<3, Tuple<OneTo<Int64>, OneTo<Int64>, OneTo<Int64>>>, CartesianIndices<3, Tuple<OneTo<Int64>, OneTo<Int64>, OneTo<Int64>>>>>, CuDeviceArray<Float64, 3, 1>, Broadcasted<CuArrayStyle<3, DeviceMemory>, Tuple<OneTo<Int64>, OneTo<Int64>, OneTo<Int64>>, _, Tuple<Broadcasted<CuArrayStyle<3, DeviceMemory>, void, _, Tuple<Extruded<CuDeviceArray<Float64, 3, 1>, Tuple<Bool, Bool, Bool>, Tuple<Int64, Int64, Int64>>, Extruded<CuDeviceArray<Float64, 3, 1>, Tuple<Bool, Bool, Bool>, Tuple<Int64, Int64, Int64>>>>, Broadcasted<CuArrayStyle<3, DeviceMemory>, void, _, Tuple<Extruded<CuDeviceArray<Float64, 3, 1>, Tuple<Bool, Bool, Bool>, Tuple<Int64, Int64, Int64>>, Extruded<CuDeviceArray<Float64, 3, 1>, Tuple<Bool, Bool, Bool>, Tuple<Int64, Int64, Int64>>>>>>)` | 3.1% | 3.212 | 180 |
| `gpu_broadcast_kernel_cartesian(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndices<2, Tuple<OneTo<Int64>, OneTo<Int64>>>, NDRange<2, DynamicSize, DynamicSize, CartesianIndices<2, Tuple<OneTo<Int64>, OneTo<Int64>>>, CartesianIndices<2, Tuple<OneTo<Int64>, OneTo<Int64>>>>>, SubArray<Complex<Float64>, 2, CuDeviceArray<Complex<Float64>, 3, 1>, Tuple<Int64, Slice<OneTo<Int64>>, Slice<OneTo<Int64>>>, true>, Broadcasted<CuArrayStyle<2, DeviceMemory>, Tuple<OneTo<Int64>, OneTo<Int64>>, identity, Tuple<Int64>>)` | 3.0% | 3.143 | 600 |
| `void regular_fft_r2c<96u, EPT<4u, 6u>, 8u, 8u, (padding_t)6, (twiddle_t)0, (loadstore_modifier_t)2, (layout_t)0, unsigned int, double, HostConfigPlaceholder>(kernel_arguments_t<unsigned int>)` | 3.0% | 3.142 | 180 |
| `[copy device to device memory]` | 2.9% | 3.018 | 480 |
| `_cheb_extend_(CuDeviceArray<Complex<Float64>, 2, 1>, CuDeviceArray<Complex<Float64>, 2, 1>, Bool)` | 2.8% | 2.920 | 540 |
| `gpu_broadcast_kernel_cartesian(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndices<2, Tuple<OneTo<Int64>, OneTo<Int64>>>, NDRange<2, DynamicSize, DynamicSize, CartesianIndices<2, Tuple<OneTo<Int64>, OneTo<Int64>>>, CartesianIndices<2, Tuple<OneTo<Int64>, OneTo<Int64>>>>>, CuDeviceArray<Complex<Float64>, 2, 1>, Broadcasted<CuArrayStyle<2, DeviceMemory>, Tuple<OneTo<Int64>, OneTo<Int64>>, identity, Tuple<Extruded<CuDeviceArray<Complex<Float64>, 2, 1>, Tuple<Bool, Bool>, Tuple<Int64, Int64>>>>)` | 2.6% | 2.702 | 540 |

Top 15 kernels shown; percentages use all captured device activities.
