# Tensorial

## Introduction

This package provides symbolic operations for tensors written in Julia.
The main motivation behind the development of this package is to provide useful tensor operations
(e.g., contraction; tensor product, `⊗`; `inv`; etc.) for arbitrary order and size of tensors.
The symmetry of the tensor is also supported for fast computations.
The way to give size of the tensor is similar to [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl), except that symmetry can be specified by `Symmetry`.
For example, symmetric fourth-order tensor can be represented as `Tensor{Tuple{@Symmetry{3,3}, @Symmetry{3,3}}}`.
All of these tensors can also be used in provided automatic differentiation functions.

## Installation

```julia
pkg> add https://github.com/KeitaNakamura/Tensorial.jl.git
```

## Other tensor packages

- [Einsum.jl](https://github.com/ahwillia/Einsum.jl)
- [TensorOprations.jl](https://github.com/Jutho/TensorOperations.jl)
- [Tensors.jl](https://github.com/KristofferC/Tensors.jl)

## Inspiration

- [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl)
- [Tensors.jl](https://github.com/KristofferC/Tensors.jl)
