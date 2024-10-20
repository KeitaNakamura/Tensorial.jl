# Tensorial.jl

*Statically sized tensors and related operations for Julia*

[![CI](https://github.com/KeitaNakamura/Tensorial.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/KeitaNakamura/Tensorial.jl/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/KeitaNakamura/Tensorial.jl/branch/main/graph/badge.svg?token=V58DXDI1R5)](https://codecov.io/gh/KeitaNakamura/Tensorial.jl)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13955151.svg)](https://doi.org/10.5281/zenodo.13955151)

Tensorial.jl provides statically sized `Tensor` which is compatible with the `AbstractArray`, similar to the `SArray` in [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl).
In addition to the basic operations for `AbstractArray`, the package also offers a *tensorial* interface and several powerful features:

* Contraction, tensor product (`⊗`), and a flexible `@einsum` macro for Einstein summation convention
* A `@Symmetry` macro to define the tensor symmetries, which eliminates unnecessary calculations
* Automatic differentiation through `gradient` and `hessian` functions
* Performance comparable to `SArray` (see [benchmarks](https://keitanakamura.github.io/Tensorial.jl/stable/Benchmarks/))

## Documentation

[![Stable](https://img.shields.io/badge/docs-latest%20release-blue.svg)](https://KeitaNakamura.github.io/Tensorial.jl/stable)

## Quick start

```julia
julia> using Tensorial

julia> x = Vec{3}(rand(3)); # constructor similar to SArray

julia> A = @Mat rand(3,3); # @Vec, @Mat and @Tensor, analogous to @SVector, @SMatrix and @SArray

julia> A ⋅ x ≈ A * x # single contraction (⋅)
true

julia> A ⊡ A ≈ tr(A'A) # double contraction (⊡)
true

julia> x ⊗ x ≈ x * x' # tensor product (⊗)
true

julia> (@einsum x[i] * A[j,i] * x[j]) ≈ x ⋅ A' ⋅ x # Einstein summation (@einsum)
true

julia> S = rand(Tensor{Tuple{@Symmetry{3,3}}}); # specify symmetry S₍ᵢⱼ₎

julia> SS = rand(Tensor{Tuple{@Symmetry{3,3}, @Symmetry{3,3}}}); # SS₍ᵢⱼ₎₍ₖₗ₎

julia> inv(SS) ⊡ S ≈ @einsum inv(SS)[i,j,k,l] * S[k,l] # it just works
true

julia> δ = one(Mat{3,3}) # identity tensor
3×3 Tensor{Tuple{3, 3}, Float64, 2, 9}:
 1.0  0.0  0.0
 0.0  1.0  0.0
 0.0  0.0  1.0

julia> gradient(identity, S) ≈ one(SS) # ∂Sᵢⱼ/∂Sₖₗ = (δᵢₖδⱼₗ + δᵢₗδⱼₖ) / 2
true
```

## Other tensor packages

* [Einsum.jl](https://github.com/ahwillia/Einsum.jl)
* [TensorOprations.jl](https://github.com/Jutho/TensorOperations.jl)
* [Tensors.jl](https://github.com/Ferrite-FEM/Tensors.jl)
* [Tullio.jl](https://github.com/mcabbott/Tullio.jl)

## Inspiration

Some functionalities are inspired from the following packages:

* [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl)
* [Tensors.jl](https://github.com/Ferrite-FEM/Tensors.jl)
