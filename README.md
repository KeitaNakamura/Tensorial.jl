# Tensorial.jl

*Tensorial operations, symmetries, and differentiation for Julia*

[![CI](https://github.com/KeitaNakamura/Tensorial.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/KeitaNakamura/Tensorial.jl/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/KeitaNakamura/Tensorial.jl/branch/main/graph/badge.svg?token=V58DXDI1R5)](https://codecov.io/gh/KeitaNakamura/Tensorial.jl)

Tensorial.jl provides a statically sized `Tensor` type compatible with `AbstractArray`, similar to `SArray` from [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl), together with a *tensorial* interface for concise and efficient computations.

In addition to basic `AbstractArray` operations, Tensorial.jl supports tensor symmetries through `@Symmetry`, allowing adjacent groups of indices to be treated symmetrically. These symmetries are consistently respected in tensor operations such as contraction, inversion, and automatic differentiation, reducing unnecessary computations while preserving the intended tensor structure.

Key features of Tensorial.jl include:

* Contraction, tensor product (`⊗`), and a flexible `@einsum` macro for Einstein summation
* A `@Symmetry` macro for defining tensor symmetries on adjacent index groups, consistently respected in tensor operations and automatic differentiation
* [Automatic differentiation](https://keitanakamura.github.io/Tensorial.jl/stable/Automatic%20differentiation/) via `∂`, `gradient`, and `hessian`, leveraging [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl)
* [Direct sums](https://keitanakamura.github.io/Tensorial.jl/stable/Direct%20sum/) for mixed tensor and scalar variables, preserving block structure in differentiation and linear algebra
* Performance comparable to `SArray` (see [benchmarks](https://keitanakamura.github.io/Tensorial.jl/stable/Benchmarks/))

## Documentation

[![Stable](https://img.shields.io/badge/docs-latest%20release-blue.svg)](https://KeitaNakamura.github.io/Tensorial.jl/stable)

## Quick start

```julia
julia> using Tensorial

julia> x = Vec{3}(rand(3)); # constructor analogous to SArray.jl

julia> A = @Mat rand(3,3); # @Vec, @Mat, and @Tensor, analogous to @SVector, @SMatrix, and @SArray

julia> A ⊡ x ≈ A * x # single contraction (⊡)
true

julia> A ⊡₂ A ≈ A ⋅ A # double contraction (⊡₂)
true

julia> x ⊗ x ≈ x * x' # tensor product (⊗)
true

julia> (@einsum x[i] * A[j,i] * x[j]) ≈ x ⋅ (A' * x) # Einstein summation with @einsum
true

julia> S = rand(Tensor{Tuple{@Symmetry{3,3}}}); # symmetric tensor S₍ᵢⱼ₎

julia> SS = rand(Tensor{Tuple{@Symmetry{3,3}, @Symmetry{3,3}}}); # symmetric tensor SS₍ᵢⱼ₎₍ₖₗ₎

julia> inv(SS) ⊡₂ S ≈ @einsum inv(SS)[i,j,k,l] * S[k,l] # works as expected
true

julia> δ = one(Mat{3,3}) # identity tensor
3×3 Tensor{Tuple{3, 3}, Float64, 2, 9}:
 1.0  0.0  0.0
 0.0  1.0  0.0
 0.0  0.0  1.0

julia> gradient(identity, S) ≈ one(SS) # ∂Sᵢⱼ/∂Sₖₗ = (δᵢₖδⱼₗ + δᵢₗδⱼₖ) / 2
true
```

## Change log

### v0.19

#### New features

* `∂` is now the standard API for automatic differentiation. `gradient` and `hessian` remain available as aliases for `∂{1}` and `∂{2}`, respectively, so existing code using them continues to work.
* Automatic differentiation now supports multiple inputs and multiple outputs (see [docs](https://keitanakamura.github.io/Tensorial.jl/stable/Automatic%20differentiation/)).
* Repeated differentiation with respect to `Vec` now accounts for symmetry. In particular, Hessians and higher-order derivatives with respect to `Vec` are returned as symmetric tensors when appropriate.
* Added support for direct sums of mixed tensor and scalar variables, preserving block structure in differentiation and linear algebra.

#### Breaking changes

* The Hessian with respect to `Vec` is now returned as a symmetric tensor type rather than a non-symmetric tensor type.

### v0.18

#### Breaking changes

* Tensorial.jl is now built on [TensorCore.jl](https://github.com/JuliaMath/TensorCore.jl).
* Single contraction: `⋅` has been replaced by `⊡` (`⋅` now behaves as in `LinearAlgebra`).
* Double contraction: `⊡` has been replaced by `⊡₂` (which can be typed by `\boxdot<tab>\_2<tab>`).
* Broadcasting: Scalar-like behavior has been removed. Broadcasting now behaves the same as with other `AbstractArray`s.
* `mean`: The specialized `mean` definition in `Statistics` has been removed.

## Other tensor packages

* [Einsum.jl](https://github.com/ahwillia/Einsum.jl)
* [TensorOperations.jl](https://github.com/Jutho/TensorOperations.jl)
* [Tensors.jl](https://github.com/Ferrite-FEM/Tensors.jl)
* [Tullio.jl](https://github.com/mcabbott/Tullio.jl)

## Inspiration

Some functionalities are inspired from the following packages:

* [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl)
* [Tensors.jl](https://github.com/Ferrite-FEM/Tensors.jl)

## Citation

If you find Tensorial.jl useful in your work, I kindly request that you cite it as below:

```bibtex
@software{NakamuraTensorial2024,
    title = {Tensorial.jl: a {J}ulia package for tensor operations},
   author = {Nakamura, Keita},
      doi = {10.5281/zenodo.13955151},
     year = {2024},
      url = {https://github.com/KeitaNakamura/Tensorial.jl}
  licence = {MIT},
}
```
