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
        "Getting Started" => [
            "Cheat Sheet.md",
        ],
        "Manual" => [
            "Tensor Type.md",
            "Constructors.md",
            "Tensor Operations.md",
            "Continuum Mechanics.md",
            "Voigt form.md",
            "Tensor Inversion.md",
            "Broadcast.md",
            "Automatic differentiation.md",
            "Einstein summation.md",
            "Quaternion.md",
        ],
        "Benchmarks.md",
    ],
    doctest = true, # :fix
)

deploydocs(
    repo = "github.com/KeitaNakamura/Tensorial.jl.git",
    devbranch = "main",
)
