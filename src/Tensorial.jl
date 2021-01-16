module Tensorial

using LinearAlgebra, Statistics
# re-exports from LinearAlgebra and Statistics
export ⋅, ×, dot, tr, det, norm, mean

using StaticArrays
using Base: @pure, @_inline_meta

import Base: transpose, inv
import LinearAlgebra: dot, norm, tr, adjoint, det, cross
import Statistics: mean

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
    vol,
    dev,
    symmetric,
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
