using Tensorial
using Test, Random

using LinearAlgebra: Symmetric, Eigen
using StaticArrays: SArray, SVector, SOneTo, SUnitRange

import Combinatorics # for levicivita

if VERSION < v"1.4"
    using Tensorial: only
end

include("Symmetry.jl")
include("Space.jl")
include("AbstractTensor.jl")
include("Tensor.jl")
include("permute.jl")
include("einsum.jl")
include("ops.jl")
include("continuum_mechanics.jl")
include("inv.jl")
include("eigen.jl")
include("voigt.jl")
include("ad.jl")
include("broadcast.jl")
include("abstractarray.jl")

include("quaternion.jl")
