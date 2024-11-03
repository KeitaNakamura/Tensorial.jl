# Tensorial.jl

*Statically sized tensors and related operations for Julia*

[![CI](https://github.com/KeitaNakamura/Tensorial.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/KeitaNakamura/Tensorial.jl/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/KeitaNakamura/Tensorial.jl/branch/main/graph/badge.svg?token=V58DXDI1R5)](https://codecov.io/gh/KeitaNakamura/Tensorial.jl)

Tensorial.jl provides statically sized `Tensor` type that is compatible with `AbstractArray`, similar to `SArray` from [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl).
In addition to supporting basic `AbstractArray` operations, the package offers a *tensorial* interface and several advanced features:

* Contraction, tensor product (`⊗`), and a flexible `@einsum` macro for Einstein summation convention
* A `@Symmetry` macro to define the tensor symmetries, eliminating unnecessary calculations
* Automatic differentiation leveraging [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl)
* Performance comparable to `SArray` (see [benchmarks](https://keitanakamura.github.io/Tensorial.jl/stable/Benchmarks/))

## Documentation

[![Stable](https://img.shields.io/badge/docs-latest%20release-blue.svg)](https://KeitaNakamura.github.io/Tensorial.jl/stable)

## Breaking changes (v0.18)

Starting from version 0.18, Tensorial.jl is now built on [TensorCore.jl](https://github.com/JuliaMath/TensorCore.jl). The breaking changes are as follows:

* Single contraction: `⋅` has been replaced by `⊡` (`⋅` now behaves as in `LinearAlgebra`).
* Double contraction: `⊡` has been replaced by `⊡₂` (which can be typed by `\boxdot<tab>\_2<tab>`).
* `@einsum`: The syntax now aligns with other tensor packages.
* Broadcasting: Scalar-like behavior has been removed. Broadcasting now behaves the same as with other `AbstractArray`s.
* `mean`: The specialized `mean` definition in `Statistics` has been removed.

## Quick start

```julia
julia> using Tensorial

julia> x = Vec{3}(rand(3)); # constructor similar to SArray.jl

julia> A = @Mat rand(3,3); # @Vec, @Mat and @Tensor, analogous to @SVector, @SMatrix and @SArray

julia> A ⊡ x ≈ A * x # single contraction (⊡)
true

julia> A ⊡₂ A ≈ A ⋅ A # double contraction (⊡₂)
true

julia> x ⊗ x ≈ x * x' # tensor product (⊗)
true

julia> (@einsum y := x[i] * A[j,i] * x[j]) ≈ x ⊡ A' ⊡ x # Einstein summation (@einsum)
true

julia> As = rand(Tensor{Tuple{@Symmetry{3,3}}}); # specify symmetry S₍ᵢⱼ₎

julia> AAs = rand(Tensor{Tuple{@Symmetry{3,3}, @Symmetry{3,3}}}); # SS₍ᵢⱼ₎₍ₖₗ₎

julia> inv(AAs) ⊡₂ As ≈ @einsum Bs[i,j] := inv(AAs)[i,j,k,l] * As[k,l] # it just works
true

julia> δ = one(Mat{3,3}) # identity tensor
3×3 Tensor{Tuple{3, 3}, Float64, 2, 9}:
 1.0  0.0  0.0
 0.0  1.0  0.0
 0.0  0.0  1.0

julia> gradient(identity, As) ≈ one(AAs) # ∂Asᵢⱼ/∂Asₖₗ = (δᵢₖδⱼₗ + δᵢₗδⱼₖ) / 2
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
