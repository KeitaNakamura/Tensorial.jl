# Tensorial.jl

*Tensorial operations, symmetries, and differentiation for Julia*

[![CI](https://github.com/KeitaNakamura/Tensorial.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/KeitaNakamura/Tensorial.jl/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/KeitaNakamura/Tensorial.jl/branch/main/graph/badge.svg?token=V58DXDI1R5)](https://codecov.io/gh/KeitaNakamura/Tensorial.jl)

Tensorial.jl provides statically sized tensor types for Julia. They behave like
`AbstractArray`s, but carry tensor-specific structure such as symmetry.
Contractions, tensor products, indexed formulas, automatic differentiation, and
direct sums work on those tensor values directly, so code can follow the
formulas.

A fourth-order tensor can act on a symmetric tensor without flattening it by
hand:

```julia
using Tensorial

ε = symmetric(@Mat [0.02 0.01 0.0; 0.01 0.00 0.0; 0.0 0.0 -0.01])
ℂ = rand(SymmetricFourthOrderTensor{3})

σ = ℂ ⊡₂ ε                            # tensor notation
@einsum σ[i,j] := ℂ[i,j,k,l] * ε[k,l] # index notation
```

## Why Tensorial?

Tensorial.jl is useful when you want tensor formulas to show up clearly in
Julia code:

- Carry tensor-space structure in the type, including symmetry
  ([Tensor types and spaces](https://keitanakamura.github.io/Tensorial.jl/stable/Tensor%20types%20and%20spaces/)).
- Write contractions and tensor products directly with `⊡`, `⊡₂`, `⊗`,
  and `@einsum`
  ([Operations](https://keitanakamura.github.io/Tensorial.jl/stable/Operations/)).
- Differentiate tensor formulas with `gradient`, `hessian`, and `∂`
  ([Automatic differentiation](https://keitanakamura.github.io/Tensorial.jl/stable/Automatic%20differentiation/)).
- Build block-structured states with direct sums for related unknowns:
  `pack(σ, Δγ)`
  ([Direct sum](https://keitanakamura.github.io/Tensorial.jl/stable/Direct%20sum/)).
- Use small tensors efficiently, with performance comparable to `SArray`
  ([Benchmarks](https://keitanakamura.github.io/Tensorial.jl/stable/Benchmarks/)).

## Installation

Install Tensorial with Julia's package manager:

```julia-repl
pkg> add Tensorial
```

## Documentation

See the [documentation](https://keitanakamura.github.io/Tensorial.jl/stable)
for the full manual. Start with
[Getting started](https://keitanakamura.github.io/Tensorial.jl/stable/Getting%20started/)
for a guided tour with executable examples of tensor construction, symmetry,
contractions, `@einsum`, automatic differentiation, and direct sums.

## Change log

See [CHANGELOG.md](CHANGELOG.md) for release history.

## Other tensor packages

* [Einsum.jl](https://github.com/ahwillia/Einsum.jl)
* [TensorOperations.jl](https://github.com/Jutho/TensorOperations.jl)
* [Tensors.jl](https://github.com/Ferrite-FEM/Tensors.jl)
* [Tullio.jl](https://github.com/mcabbott/Tullio.jl)

## Inspiration

Some functionality is inspired by the following packages:

* [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl)
* [Tensors.jl](https://github.com/Ferrite-FEM/Tensors.jl)

## Citation

If Tensorial.jl is useful in your work, please cite it as follows:

```bibtex
@software{NakamuraTensorial2024,
    title = {Tensorial.jl: a {J}ulia package for tensor operations},
   author = {Nakamura, Keita},
      doi = {10.5281/zenodo.13955151},
     year = {2024},
      url = {https://github.com/KeitaNakamura/Tensorial.jl},
  license = {MIT},
}
```
