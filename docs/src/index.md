# Tensorial

## Introduction

Tensorial provides useful tensor operations (e.g., contraction; tensor product, `⊗`; `inv`; etc.) written in the [Julia programming language](https://julialang.org).
The library supports arbitrary size of non-symmetric and symmetric tensors, where symmetries should be specified to avoid wasteful duplicate computations.
The way to give a size of the tensor is similar to [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl), and symmetries of tensors can be specified by using `@Symmetry`.
For example, symmetric fourth-order tensor (symmetrizing tensor) is represented in this library as `Tensor{Tuple{@Symmetry{3,3}, @Symmetry{3,3}}}`.
Einstein summation macro and automatic differentiation functions are also provided.

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
