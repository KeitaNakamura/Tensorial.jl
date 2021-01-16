module Tensorial

using LinearAlgebra
using StaticArrays
using Base: @pure, @_inline_meta

import LinearAlgebra: dot

export
    Symmetry,
    @Symmetry

include("utils.jl")
include("symmetry.jl")
include("indexing.jl")
include("tensor.jl")
include("ops.jl")

end # module
