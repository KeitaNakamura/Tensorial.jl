# Tensorial.jl

## Introduction

Tensorial.jl provides statically sized `Tensor` which is compatible with the `AbstractArray`, similar to the `SArray` in [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl).
In addition to the basic operations for `AbstractArray`, the package also offers a *tensorial* interface and several powerful features:

* Contraction, tensor product (`⊗`), and a flexible `@einsum` macro for Einstein summation convention
* A `@Symmetry` macro to define the tensor symmetries, which eliminates unnecessary calculations
* Automatic differentiation through `gradient` and `hessian` functions
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
