# Tensorial.jl

Tensorial.jl provides statically sized tensor types for Julia. They behave like
`AbstractArray`s, but carry tensor-specific structure such as symmetry.

This manual is organized by what you want to do. If you are new to Tensorial,
start with [Getting Started](@ref); it gives a short executable tour of the main
pieces: tensor construction, operations, `@einsum`, symmetry, automatic
differentiation, and direct sums.

## Installation

Install Tensorial with Julia's package manager:

```julia-repl
pkg> add Tensorial
```

## Choose by task

- Understand tensor spaces, aliases, and symmetry in tensor types:
  [Tensor Types and Spaces](@ref).
- Create vectors, matrices, higher-order tensors, and symmetric tensors:
  [Constructing Tensors](@ref).
- Write contractions, tensor products, and indexed formulas:
  [Operations](@ref).
- Differentiate tensor formulas and get tensor results back:
  [Automatic Differentiation](@ref).
- Pack several related unknowns into one block-structured state:
  [Direct Sum](@ref).
- Convert tensors at array boundaries with Voigt or Mandel notation:
  [Voigt Form](@ref).
- Work with 3D rotations and quaternions:
  [Quaternion](@ref).
- Check practical notes for writing Tensorial code:
  [Practical Tips](@ref).

## Reference pages

- Look up public signatures and docstrings:
  [API Reference](@ref).
- Check small-tensor performance against `Array` and `SArray`:
  [Benchmarks](@ref "Benchmarks").
