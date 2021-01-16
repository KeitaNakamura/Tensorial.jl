module Tensorial

using LinearAlgebra
# re-exports from LinearAlgebra
export ⋅, dot

using StaticArrays
using Base: @pure, @_inline_meta

import LinearAlgebra: dot

export
# Types
    Symmetry,
    @Symmetry,
    Tensor,
    SecondOrderTensor,
    ThirdOrderTensor,
    FourthOrderTensor,
    SymmetricSecondOrderTensor,
    SymmetricThirdOrderTensor,
    SymmetricFourthOrderTensor,
    Vec,
    Mat,
# operations
    contract,
    ⊗,
    ⊡

include("utils.jl")
include("symmetry.jl")
include("indexing.jl")
include("tensor.jl")
include("ops.jl")

const ⊗ = otimes
const ⊡ = dcontract

end # module
