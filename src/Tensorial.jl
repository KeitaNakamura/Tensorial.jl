module Tensorial

using Base: @pure

export
    Symmetry,
    @Symmetry

include("utils.jl")
include("symmetry.jl")
include("indexing.jl")

end # module
