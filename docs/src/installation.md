# Installation

Use Julia 1.11 or later. Clone the repository and instantiate its environment;
its `Project.toml` supplies the source URLs for the unregistered dependencies.

```sh
git clone https://github.com/Davide-Lasagna-s-Lab/ChannelFlow.jl.git
cd ChannelFlow.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## NVIDIA GPU environment

The repository provides an environment containing CUDA and a path dependency
on the local ChannelFlow checkout:

```sh
julia --project=test/cuda -e 'using Pkg; Pkg.instantiate()'
julia --project=test/cuda
```

In Julia, check the device before running a simulation:

```julia
using ChannelFlow, CUDA, Adapt
CUDA.functional() || error("A working NVIDIA GPU and driver are required")
CUDA.versioninfo()
CUDA.allowscalar(false)
```

The CUDA dependency is optional: CPU-only users do not need it. The package's
CUDA extension loads when both ChannelFlow and CUDA are loaded. GPU calculations
use `Float64`/`ComplexF64`; an accelerator's double-precision performance matters.

For a separate application environment, develop the checkout and add its source
dependencies explicitly if they are not already available:

```julia
using Pkg
Pkg.activate("my-channel-simulation")
Pkg.add(url="https://github.com/Davide-Lasagna-s-Lab/ChebyshevHelmoltzSolvers.jl.git")
Pkg.develop(path="/path/to/ChannelFlow.jl")
Pkg.add(["CUDA", "Adapt", "FFTW"])
```

Run from that environment thereafter. Restart Julia after package updates that
change types. If an existing notebook reports a missing dependency after an
update, resolve and instantiate its active environment, then restart its kernel.

Flows.jl is optional. Install it separately to use `Flows.flow` and monitors,
as shown in the [quick start](quickstart.md); manual `step!` needs neither
Flows installation nor loading.
