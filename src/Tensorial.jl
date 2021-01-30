module Tensorial

using LinearAlgebra, Statistics
# re-exports from LinearAlgebra and Statistics
export ⋅, ×, dot, tr, det, norm, mean, I, eigen, eigvals, eigvecs

using StaticArrays
using Base: @pure, @_inline_meta, @_propagate_inbounds_meta
using ForwardDiff: Dual, value, partials

import Base: transpose, inv
import LinearAlgebra: dot, norm, tr, adjoint, det, cross, eigen, eigvals, eigvecs
import Statistics: mean

export
# Symmetry/Size
    Symmetry,
    @Symmetry,
    Size,
# AbstractTensor
    AbstractTensor,
    AbstractSecondOrderTensor,
    AbstractFourthOrderTensor,
    AbstractSymmetricSecondOrderTensor,
    AbstractSymmetricFourthOrderTensor,
    AbstractVec,
    AbstractMat,
# Tensor
    Tensor,
    SecondOrderTensor,
    FourthOrderTensor,
    SymmetricSecondOrderTensor,
    SymmetricFourthOrderTensor,
    Vec,
    Mat,
# MTensor
    MTensor,
    MSecondOrderTensor,
    MFourthOrderTensor,
    MSymmetricSecondOrderTensor,
    MSymmetricFourthOrderTensor,
    MVec,
    MMat,
# macros
    @Vec,
    @Mat,
    @Tensor,
    @MVec,
    @MMat,
    @MTensor,
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
include("Size.jl")
include("indexing.jl")
include("einsum.jl")
include("AbstractTensor.jl")
include("Tensor.jl")
include("MTensor.jl")
include("ops.jl")
include("voigt.jl")
include("ad.jl")

const ⊗ = otimes
const ⊡ = double_contraction

end # module
