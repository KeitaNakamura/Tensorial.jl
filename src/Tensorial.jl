module Tensorial

using LinearAlgebra, Statistics
# re-exports from LinearAlgebra and Statistics
export ⋅, ×, dot, tr, det, norm, normalize, mean, I, cross, eigen, eigvals, eigvecs

using StaticArrays
using Base: @pure, @_inline_meta, @_propagate_inbounds_meta
using ForwardDiff: Dual, value, partials
import SIMD
import StaticArrays: qr, lu, svd # defined in LinearAlgebra, but call methods in StaticArrays
# re-exports from StaticArrays
export qr, lu, svd

import Base: transpose, inv
import LinearAlgebra: dot, norm, normalize, tr, adjoint, det, cross, eigen, eigvals, eigvecs
import Statistics: mean

export
# Symmetry/Space
    Symmetry,
    @Symmetry,
    Space,
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
# macros
    @Vec,
    @Mat,
    @Tensor,
    @einsum,
# operations
    contraction,
    otimes,
    dotdot,
    symmetric,
    skew,
    ⊗,
    ⊡,
    rotmat,
    rotmatx,
    rotmaty,
    rotmatz,
    rotate,
# continuum mechanics
    vol,
    dev,
    stress_invariants,
    deviatoric_stress_invariants,
# voigt
    tovoigt,
    fromvoigt,
    tomandel,
    frommandel,
# ad
    gradient,
    hessian,
# quaternion
    Quaternion,
    quaternion


include("utils.jl")
include("Symmetry.jl")
include("Space.jl")
include("indexing.jl")
include("AbstractTensor.jl")
include("einsum.jl")
include("Tensor.jl")
include("permute.jl")
include("ops.jl")
include("continuum_mechanics.jl")
include("inv.jl")
include("voigt.jl")
include("ad.jl")
include("simd.jl")
include("broadcast.jl")

include("quaternion.jl")

const ⊗ = otimes
const ⊡ = double_contraction

end # module
