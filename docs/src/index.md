# Tensorial

## Introduction

Tensorial provides useful tensor operations, such as contraction, tensor product (`⊗`), and inversion (`inv`), implemented in the Julia programming language. The library supports tensors of arbitrary size, including both symmetric and non-symmetric tensors, where symmetries can be specified to avoid redundant computations. The approach for defining the size of a tensor is similar to that used in [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl), and tensor symmetries can be specified using the `@Symmetry` macro. For instance, a symmetric fourth-order tensor (a symmetrized tensor) is represented in this library as `Tensor{Tuple{@Symmetry{3,3}, @Symmetry{3,3}}}`. The library also includes an Einstein summation macro `@einsum` and functions for automatic differentiation, such as `gradient` and `hessian`.

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
