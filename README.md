# Tensorial.jl

*Statically sized tensors and related operations for Julia*

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://KeitaNakamura.github.io/Tensorial.jl/stable)
[![CI](https://github.com/KeitaNakamura/Tensorial.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/KeitaNakamura/Tensorial.jl/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/KeitaNakamura/Tensorial.jl/branch/main/graph/badge.svg?token=V58DXDI1R5)](https://codecov.io/gh/KeitaNakamura/Tensorial.jl)

Tensorial.jl provides statically sized `Tensor` which is compatible with the `AbstractArray`, similar to the `SArray` in [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl).
In addition to the basic operations for `AbstractArray`, the package also offers a *tensorial* interface and several convenient features:

* Contraction, tensor product (`⊗`), and a flexible `@einsum` macro for Einstein summation convention
* A `@Symmetry` macro to define the tensor symmetries, which eliminates unnecessary calculations
* Automatic differentiation through `gradient` and `hessian` functions

## Speed

```julia
a = rand(Vec{3})                                        # vector of length 3
A = rand(Mat{3,3})                                      # 3x3 second order tensor
S = rand(Tensor{Tuple{@Symmetry{3,3}}})                 # 3x3 symmetric second order tensor
B = rand(Tensor{Tuple{3,3,3}})                          # 3x3x3 third order tensor
AA = rand(Tensor{NTuple{4,3}})                          # 3x3x3x3 fourth order tensor
SS = rand(Tensor{Tuple{@Symmetry{3,3},@Symmetry{3,3}}}) # 3x3x3x3 symmetric fourth order tensor (symmetrizing tensor)
```

| Operation  | `Tensor` | `Array` | speed-up |
|:-----------|---------:|--------:|---------:|
| **Single contraction** | | | |
| `a ⋅ a` | 1.417 ns | 5.208 ns | ×3.7 |
| `A ⋅ a` | 1.708 ns | 39.270 ns | ×23.0 |
| `S ⋅ a` | 1.708 ns | 39.396 ns | ×23.1 |
| **Double contraction** | | | |
| `A ⊡ A` | 1.667 ns | 4.416 ns | ×2.6 |
| `S ⊡ S` | 1.666 ns | 4.416 ns | ×2.7 |
| `B ⊡ A` | 2.833 ns | 76.139 ns | ×26.9 |
| `AA ⊡ A` | 5.625 ns | 84.587 ns | ×15.0 |
| `SS ⊡ S` | 2.709 ns | 84.631 ns | ×31.2 |
| **Tensor product** | | | |
| `a ⊗ a` | 1.666 ns | 24.138 ns | ×14.5 |
| **Cross product** | | | |
| `a × a` | 1.666 ns | 24.138 ns | ×14.5 |
| **Determinant** | | | |
| `det(A)` | 1.708 ns | 125.931 ns | ×73.7 |
| `det(S)` | 1.708 ns | 105.032 ns | ×61.5 |
| **Inverse** | | | |
| `inv(A)` | 3.750 ns | 335.784 ns | ×89.5 |
| `inv(S)` | 2.875 ns | 337.416 ns | ×117.4 |
| `inv(AA)` | 647.071 ns | 1.108 μs | ×1.7 |
| `inv(SS)` | 217.057 ns | 1.108 μs | ×5.1 |

The benchmarks are generated by
[`runbenchmarks.jl`](https://github.com/KeitaNakamura/Tensorial.jl/blob/master/benchmark/runbenchmarks.jl)
on the following system:

```julia
julia> versioninfo()
Julia Version 1.10.5
Commit 6f3fdf7b362 (2024-08-27 14:19 UTC)
Build Info:
  Official https://julialang.org/ release
Platform Info:
  OS: macOS (arm64-apple-darwin22.4.0)
  CPU: 8 × Apple M2
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-15.0.7 (ORCJIT, apple-m1)

```

## Installation

```julia
pkg> add Tensorial
```

## Cheat Sheet

```julia
# tensor aliases
rand(Vec{3})                        # vector
rand(Mat{2,3})                      # matrix
rand(SecondOrderTensor{3})          # 3x3 second-order tensor (this is the same as the Mat{3,3})
rand(SymmetricSecondOrderTensor{3}) # 3x3 symmetric second-order tensor (3x3 symmetric matrix)
rand(FourthOrderTensor{3})          # 3x3x3x3 fourth-order tensor
rand(SymmetricFourthOrderTensor{3}) # 3x3x3x3 symmetric fourth-order tensor

# identity tensors
one(SecondOrderTensor{3,3})        # second-order identity tensor
one(SymmetricSecondOrderTensor{3}) # symmetric second-order identity tensor
one(FourthOrderTensor{3})          # fourth-order identity tensor
one(SymmetricFourthOrderTensor{3}) # symmetric fourth-order identity tensor (symmetrizing tensor)

# zero tensors
zero(Mat{2,3}) == zeros(2,3)
zero(SymmetricSecondOrderTensor{3}) == zeros(3,3)

# random tensors
rand(Mat{2,3})
randn(Mat{2,3})

# macros (same interface as StaticArrays.jl)
@Vec [1,2,3]
@Vec rand(4)
@Mat [1 2
      3 4]
@Mat rand(4,4)
@Tensor rand(2,2,2)

# statically sized getindex by `@Tensor`
x = @Mat [1 2
          3 4
          5 6]
@Tensor(x[2:3, :])   === @Mat [3 4
                               5 6]
@Tensor(x[[1,3], :]) === @Mat [1 2
                               5 6]

# contraction and tensor product
x = rand(Mat{2,2})
y = rand(SymmetricSecondOrderTensor{2})
x ⊗ y isa Tensor{Tuple{2,2,@Symmetry{2,2}}} # tensor product
x ⋅ y isa Tensor{Tuple{2,2}}                # single contraction (x_ij * y_jk)
x ⊡ y isa Real                              # double contraction (x_ij * y_ij)

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

# Einstein summation convention
A = rand(Mat{3,3})
B = rand(Mat{3,3})
(@einsum (i,j) -> A[i,k] * B[k,j]) == A ⋅ B
(@einsum A[i,j] * B[i,j]) == A ⊡ B

# Automatic differentiation
gradient(tr, rand(Mat{3,3})) == one(Mat{3,3}) # Tensor -> Real
gradient(identity, rand(SymmetricSecondOrderTensor{3})) == one(SymmetricFourthOrderTensor{3}) # Tensor -> Tensor
```

## Other tensor packages

- [Einsum.jl](https://github.com/ahwillia/Einsum.jl)
- [TensorOprations.jl](https://github.com/Jutho/TensorOperations.jl)
- [Tensors.jl](https://github.com/KristofferC/Tensors.jl)

## Inspiration

- [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl)
- [Tensors.jl](https://github.com/KristofferC/Tensors.jl)
