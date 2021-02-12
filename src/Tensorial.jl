module Tensorial

using LinearAlgebra, Statistics
# re-exports from LinearAlgebra and Statistics
export ⋅, ×, dot, tr, det, norm, mean, I, eigen, eigvals, eigvecs

using StaticArrays
using Base: @pure, @_inline_meta, @_propagate_inbounds_meta
using ForwardDiff: Dual, value, partials
import SIMD

import Base: transpose, inv
import LinearAlgebra: dot, norm, tr, adjoint, det, cross, eigen, eigvals, eigvecs
import Statistics: mean

export
# Symmetry/Skew/Space
    Symmetry,
    @Symmetry,
    Skew,
    @Skew,
    Space,
# AbstractTensor
    AbstractTensor,
    AbstractSecondOrderTensor,
    AbstractFourthOrderTensor,
    AbstractSymmetricSecondOrderTensor,
    AbstractSkewSymmetricSecondOrderTensor,
    AbstractSymmetricFourthOrderTensor,
    AbstractVec,
    AbstractMat,
# Tensor
    Tensor,
    SecondOrderTensor,
    FourthOrderTensor,
    SymmetricSecondOrderTensor,
    SkewSymmetricSecondOrderTensor,
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
    skew,
    ⊗,
    ⊡,
    rotmat,
    rotmatx,
    rotmaty,
    rotmatz,
# ad
    gradient,
    hessian


include("utils.jl")
include("Symmetry.jl")
include("Space.jl")
include("indexing.jl")
include("einsum.jl")
include("AbstractTensor.jl")
include("Tensor.jl")
include("ops.jl")
include("voigt.jl")
include("ad.jl")
include("simd.jl")

const ⊗ = otimes
const ⊡ = double_contraction

end # module
