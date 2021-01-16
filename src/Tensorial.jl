module Tensorial

using StaticArrays

using Base: @pure, @_inline_meta

export
    Symmetry,
    @Symmetry

include("utils.jl")
include("symmetry.jl")
include("indexing.jl")
include("tensor.jl")

end # module
