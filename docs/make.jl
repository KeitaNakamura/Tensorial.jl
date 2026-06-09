using Documenter
using Tensorial

# Setup for doctests in docstrings
DocMeta.setdocmeta!(Tensorial, :DocTestSetup, recursive = true,
    quote
        using Tensorial
        using Random
        Random.seed!(1234)
    end
)

makedocs(;
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    modules = [Tensorial],
    sitename = "Tensorial.jl",
    pages=[
        "Home" => "index.md",
        "Getting started.md",
        "Manual" => [
            "Tensor types and spaces.md",
            "Constructors.md",
            "Operations.md",
            "Automatic differentiation.md",
            "Direct sum.md",
            "Voigt form.md",
            "Quaternion.md",
            "Practical tips.md",
        ],
        "API Reference" => "API reference.md",
        "Benchmarks.md",
    ],
    doctest = true, # :fix
    warnonly = [:missing_docs],
)

deploydocs(
    repo = "github.com/KeitaNakamura/Tensorial.jl.git",
    devbranch = "main",
)
