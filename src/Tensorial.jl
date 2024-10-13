module Tensorial

using Base: @_inline_meta, @_propagate_inbounds_meta
using Base.Cartesian: @ntuple, @nexprs

using LinearAlgebra, Statistics
# re-exports from LinearAlgebra and Statistics
export ⋅, ×, dot, tr, det, norm, normalize, mean, I, cross, eigen, eigvals, eigvecs

using StaticArrays
import StaticArrays: qr, lu, svd, diag, diagm # defined in LinearAlgebra, but call methods in StaticArrays
# re-exports from StaticArrays
export SArray, qr, lu, svd, diag, diagm

using ForwardDiff: Dual, Tag, value, partials

import Base: transpose, inv
import LinearAlgebra: dot, norm, normalize, tr, adjoint, det, cross, eigen, eigvals, eigvecs
import Statistics: mean

export
# Symmetry
    Symmetry,
    @Symmetry,
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
    contract,
    contract1,
    contract2,
    contract3,
    otimes,
    dotdot,
    symmetric,
    minorsymmetric,
    skew,
    ⊗,
    ⊡,
    rotmat,
    rotmatx,
    rotmaty,
    rotmatz,
    rotate,
# misc
    resize,
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

const contract1 = dot

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
# include("simd.jl")
include("broadcast.jl")
include("abstractarray.jl")

include("quaternion.jl")

const ⊡ = contract2
const ⊗ = otimes

@deprecate contraction contract true
@deprecate double_contraction contract2 true

end # module
