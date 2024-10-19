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
            "Constructors.md",
            "Operations.md",
            "Voigt form.md",
            "Broadcast.md",
            "Automatic differentiation.md",
            "Quaternion.md",
        ],
        "Benchmarks.md",
    ],
    doctest = true, # :fix
    warnonly = [:missing_docs],
)

deploydocs(
    repo = "github.com/KeitaNakamura/Tensorial.jl.git",
    devbranch = "main",
)
