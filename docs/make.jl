using Documenter, ChannelFlow, FFTW, Flows, LinearAlgebra

# Ship the measured data and figures with the static site. Keep one source copy.
assets = joinpath(@__DIR__, "src", "assets", "benchmarks")
mkpath(dirname(assets))
cp(joinpath(@__DIR__, "..", "benchmarks", "results", "final"), assets; force=true)

makedocs(
    sitename = "ChannelFlow.jl",
    authors = "Davide Lasagna",
    modules = [ChannelFlow],
    checkdocs = :exports,
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://Davide-Lasagna-s-Lab.github.io/ChannelFlow.jl/",
        edit_link = "main",
    ),
    pages = [
        "Home" => "index.md",
        "User guide" => [
            "Installation" => "installation.md",
            "Quick start" => "quickstart.md",
            "GPU execution" => "gpu.md",
            "Fields and transforms" => "fields.md",
            "Initialisation and pressure" => "pressure.md",
            "Diagnostics" => "diagnostics.md",
            "Restart and sampling" => "restart.md",
        ],
        "Numerical method" => [
            "Equations and configuration" => "equations.md",
            "Spatial method and influence matrix" => "spatial.md",
            "CNRK2 time integration" => "timestepping.md",
            "References" => "references.md",
        ],
        "Examples" => ["MKM590 turbulent channel" => "mkm590.md"],
        "Benchmarks" => "benchmarks.md",
        "Validation" => "validation.md",
        "API reference" => [
            "Index" => "api.md",
            "Configuration and state" => "api/configuration.md",
            "Fields and operators" => "api/fields.md",
            "Transforms" => "api/transforms.md",
            "Initialisation and diagnostics" => "api/initialization.md",
            "Time integration" => "api/timestepping.md",
            "Solver internals" => "api/solvers.md",
            "Module and indexing macro" => "api/module.md",
        ],
        "Developer guide" => "contributing.md",
    ],
)
