# Tensorial.jl

## Introduction

Tensorial.jl provides statically sized `Tensor` type that is compatible with `AbstractArray`, similar to `SArray` from [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl).
In addition to supporting basic `AbstractArray` operations, the package offers a *tensorial* interface and several advanced features:

* Contraction, tensor product (`⊗`), and a flexible `@einsum` macro for Einstein summation convention
* A `@Symmetry` macro to define the tensor symmetries, eliminating unnecessary calculations
* Automatic differentiation via `gradient` and `hessian` functions, leveraging [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl)
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
