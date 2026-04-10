# Tensorial.jl

## Introduction

Tensorial.jl provides a statically sized `Tensor` type compatible with `AbstractArray`, similar to `SArray` from [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl), together with a *tensorial* interface for concise and efficient computations.

In addition to basic `AbstractArray` operations, Tensorial.jl supports tensor symmetries through `@Symmetry`, allowing adjacent groups of indices to be treated symmetrically. These symmetries are consistently respected in tensor operations such as contraction, inversion, and automatic differentiation, reducing unnecessary computations while preserving the intended tensor structure.

* Contraction, tensor product (`⊗`), and a flexible `@einsum` macro for Einstein summation
* A `@Symmetry` macro for defining tensor symmetries on adjacent index groups, consistently respected in tensor operations and automatic differentiation
* Automatic differentiation via `∂`, `gradient`, and `hessian`, leveraging [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl)
* Performance comparable to `SArray` (see [benchmarks](https://keitanakamura.github.io/Tensorial.jl/stable/Benchmarks/))

## Installation

```julia
pkg> add Tensorial
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
