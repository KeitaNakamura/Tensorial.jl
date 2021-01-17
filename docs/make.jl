using Documenter
using Tensorial

# Setup for doctests in docstrings
DocMeta.setdocmeta!(Tensorial, :DocTestSetup, recursive = true,
    quote
        using Tensorial
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
            # "Constructors.md",
        ],
    ],
    doctest = true, # :fix
)

deploydocs(
    repo = "github.com/KeitaNakamura/Tensorial.jl.git",
)
