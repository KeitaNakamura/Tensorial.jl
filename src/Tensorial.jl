module Tensorial

using LinearAlgebra, Statistics
# re-exports from LinearAlgebra and Statistics
export ⋅, ×, dot, tr, det, norm, normalize, mean, I, cross, eigen, eigvals, eigvecs

using StaticArrays
using Base: @pure, @_inline_meta, @_propagate_inbounds_meta
import StaticArrays: qr, lu, svd, diag, diagm # defined in LinearAlgebra, but call methods in StaticArrays
# re-exports from StaticArrays
export SArray, qr, lu, svd, diag, diagm

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
# special tensors
    levicivita,
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
    vonmises,
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
    quaternion,
    angleaxis


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
include("eigen.jl")
include("voigt.jl")
include("ad.jl")
# include("simd.jl")
include("broadcast.jl")
include("abstractarray.jl")

include("quaternion.jl")

const ⊗ = otimes
const ⊡ = double_contraction

end # module
