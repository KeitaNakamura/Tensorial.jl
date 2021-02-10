# Tensorial

*Statically sized tensors and related operations for Julia*

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://KeitaNakamura.github.io/Tensorial.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://KeitaNakamura.github.io/Tensorial.jl/dev)
[![Build Status](https://github.com/KeitaNakamura/Tensorial.jl/workflows/CI/badge.svg)](https://github.com/KeitaNakamura/Tensorial.jl/actions)
[![codecov](https://codecov.io/gh/KeitaNakamura/Tensorial.jl/branch/main/graph/badge.svg?token=V58DXDI1R5)](https://codecov.io/gh/KeitaNakamura/Tensorial.jl)

Tensorial provides useful tensor operations (e.g., contraction; tensor product, `⊗`; `inv`; etc.) written in the [Julia programming language](https://julialang.org).
The library supports arbitrary size of non-symmetric and symmetric tensors, where symmetries should be specified to avoid wasteful duplicate computations.
The way to give a size of the tensor is similar to [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl), and symmetries of tensors can be specified by using `@Symmetry`.
For example, symmetric fourth-order tensor (symmetrizing tensor) is represented in this library as `Tensor{Tuple{@Symmetry{3,3}, @Symmetry{3,3}}}`.
Any tensors can also be used in provided automatic differentiation functions.

## Speed

```
a = rand(Vec{3})
A = rand(SecondOrderTensor{3})
S = rand(SymmetricSecondOrderTensor{3})
B = rand(Tensor{Tuple{3,3,3}})
AA = rand(FourthOrderTensor{3})
SS = rand(SymmetricFourthOrderTensor{3})
```

| Operation  | `Tensor` | `Array` | speed-up |
|:-----------|---------:|--------:|---------:|
| **Single contraction** | | | |
| `a ⋅ a` | 1.381 ns | 12.612 ns | ×9.1 |
| `A ⋅ a` | 2.080 ns | 76.358 ns | ×36.7 |
| `S ⋅ a` | 2.080 ns | 75.877 ns | ×36.5 |
| **Double contraction** | | | |
| `A ⊡ A` | 3.134 ns | 14.662 ns | ×4.7 |
| `S ⊡ S` | 2.403 ns | 14.918 ns | ×6.2 |
| `B ⊡ A` | 5.247 ns | 187.017 ns | ×35.6 |
| `AA ⊡ A` | 6.338 ns | 195.995 ns | ×30.9 |
| `SS ⊡ S` | 3.720 ns | 196.076 ns | ×52.7 |
| **Tensor product** | | | |
| `a ⊗ a` | 2.335 ns | 53.558 ns | ×22.9 |
| **Determinant and Inverse** | | | |
| `det(A)` | 1.977 ns | 231.455 ns | ×117.1 |
| `det(S)` | 2.083 ns | 231.941 ns | ×111.3 |
| `inv(A)` | 7.105 ns | 559.054 ns | ×78.7 |
| `inv(S)` | 5.103 ns | 558.768 ns | ×109.5 |

## Installation

```julia
pkg> add Tensorial
```

## Cheat Sheet

```julia
# identity tensors
one(Tensor{Tuple{3,3}})            == Matrix(I,3,3) # second-order identity tensor
one(Tensor{Tuple{@Symmetry{3,3}}}) == Matrix(I,3,3) # symmetric second-order identity tensor
I  = one(Tensor{NTuple{4,3}})               # fourth-order identity tensor
Is = one(Tensor{NTuple{2, @Symmetry{3,3}}}) # symmetric fourth-order identity tensor

# zero tensors
zero(Tensor{Tuple{2,3}}) == zeros(2, 3)
zero(Tensor{Tuple{@Symmetry{3,3}}}) == zeros(3, 3)

# random tensors
rand(Tensor{Tuple{2,3}})
randn(Tensor{Tuple{2,3}})

# macros (same interface as StaticArrays.jl)
@Vec [1,2,3]
@Vec rand(4)
@Mat [1 2
      3 4]
@Mat rand(4,4)
@Tensor rand(2,2,2)

# contraction and tensor product
x = rand(Tensor{Tuple{2,2}})
y = rand(Tensor{Tuple{@Symmetry{2,2}}})
x ⊗ y isa Tensor{Tuple{2,2,@Symmetry{2,2}}} # tensor product
x ⋅ y isa Tensor{Tuple{2,2}}                # single contraction (x_ij * y_jk)
x ⊡ y isa Real                              # double contraction (x_ij * y_ij)

# norm/tr/mean/vol/dev
x = rand(SecondOrderTensor{3}) # equal to rand(Tensor{Tuple{3,3}})
v = rand(Vec{3})
norm(v)
tr(x)
mean(x) == tr(x) / 3 # useful for computing mean stress
vol(x) + dev(x) == x # decomposition into volumetric part and deviatoric part

# det/inv for 2nd-order tensor
A = rand(SecondOrderTensor{3})          # equal to one(Tensor{Tuple{3,3}})
S = rand(SymmetricSecondOrderTensor{3}) # equal to one(Tensor{Tuple{@Symmetry{3,3}}})
det(A); det(S)
inv(A) ⋅ A ≈ one(A)
inv(S) ⋅ S ≈ one(S)

# inv for 4th-order tensor
AA = rand(FourthOrderTensor{3})          # equal to one(Tensor{Tuple{3,3,3,3}})
SS = rand(SymmetricFourthOrderTensor{3}) # equal to one(Tensor{Tuple{@Symmetry{3,3}, @Symmetry{3,3}}})
inv(AA) ⊡ AA ≈ one(AA)
inv(SS) ⊡ SS ≈ one(SS)
```

## Other tensor packages

- [Einsum.jl](https://github.com/ahwillia/Einsum.jl)
- [TensorOprations.jl](https://github.com/Jutho/TensorOperations.jl)
- [Tensors.jl](https://github.com/KristofferC/Tensors.jl)

## Inspiration

- [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl)
- [Tensors.jl](https://github.com/KristofferC/Tensors.jl)
