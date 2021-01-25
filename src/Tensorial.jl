module Tensorial

using LinearAlgebra, Statistics
# re-exports from LinearAlgebra and Statistics
export ⋅, ×, dot, tr, det, norm, mean

using StaticArrays
using Base: @pure, @_inline_meta, @_propagate_inbounds_meta
using ForwardDiff: Dual, value, partials

import Base: transpose, inv
import LinearAlgebra: dot, norm, tr, adjoint, det, cross
import Statistics: mean

export
# Types
    Symmetry,
    @Symmetry,
    Size,
    Tensor,
    SecondOrderTensor,
    ThirdOrderTensor,
    FourthOrderTensor,
    SymmetricSecondOrderTensor,
    SymmetricThirdOrderTensor,
    SymmetricFourthOrderTensor,
    Vec,
    Mat,
# macros
    @Vec,
    @Mat,
    @Tensor,
# operations
    contraction,
    otimes,
    dotdot,
    vol,
    dev,
    symmetric,
    ⊗,
    ⊡,
# ad
    gradient,
    hessian


include("utils.jl")
include("symmetry.jl")
include("size.jl")
include("indexing.jl")
include("einsum.jl")
include("tensor.jl")
include("ops.jl")
include("voigt.jl")
include("ad.jl")

const ⊗ = otimes
const ⊡ = dcontraction

end # module
