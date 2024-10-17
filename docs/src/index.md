# Tensorial

## Introduction

Tensorial.jl provides statically sized `Tensor` which is compatible with the `AbstractArray`, similar to the `SArray` in [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl).
In addition to the basic operations for `AbstractArray`, the package also offers a *tensorial* interface and several convenient features:

* Contraction, tensor product (`⊗`), and a flexible `@einsum` macro for Einstein summation convention
* A `@Symmetry` macro to define the tensor symmetries, which eliminates unnecessary calculations
* Automatic differentiation through `gradient` and `hessian` functions

## Installation

```julia
pkg> add Tensorial
```

## Other tensor packages

- [Einsum.jl](https://github.com/ahwillia/Einsum.jl)
- [TensorOprations.jl](https://github.com/Jutho/TensorOperations.jl)
- [Tensors.jl](https://github.com/KristofferC/Tensors.jl)

## Inspiration

- [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl)
- [Tensors.jl](https://github.com/KristofferC/Tensors.jl)
