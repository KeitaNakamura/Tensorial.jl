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

```julia
a = rand(Vec{3})                         # vector of length 3
A = rand(SecondOrderTensor{3})           # 3x3 second order tensor
S = rand(SymmetricSecondOrderTensor{3})  # 3x3 symmetric second order tensor
B = rand(Tensor{Tuple{3,3,3}})           # 3x3x3 third order tensor
AA = rand(FourthOrderTensor{3})          # 3x3x3x3 fourth order tensor
SS = rand(SymmetricFourthOrderTensor{3}) # 3x3x3x3 symmetric fourth order tensor (symmetrizing tensor)
```

See [here](https://keitanakamura.github.io/Tensorial.jl/stable/Cheat%20Sheet/#Aliases) for above aliases.

| Operation  | `Tensor` | `Array` | speed-up |
|:-----------|---------:|--------:|---------:|
| **Single contraction** | | | |
| `a ⋅ a` | 1.311 ns | 13.794 ns | ×10.5 |
| `A ⋅ a` | 2.027 ns | 75.799 ns | ×37.4 |
| `S ⋅ a` | 2.028 ns | 76.192 ns | ×37.6 |
| **Double contraction** | | | |
| `A ⊡ A` | 3.110 ns | 13.315 ns | ×4.3 |
| `S ⊡ S` | 2.285 ns | 13.575 ns | ×5.9 |
| `B ⊡ A` | 5.116 ns | 182.141 ns | ×35.6 |
| `AA ⊡ A` | 6.059 ns | 190.892 ns | ×31.5 |
| `SS ⊡ S` | 3.747 ns | 190.834 ns | ×50.9 |
| **Tensor product** | | | |
| `a ⊗ a` | 2.279 ns | 51.074 ns | ×22.4 |
| **Cross product** | | | |
| `a × a` | 2.279 ns | 51.074 ns | ×22.4 |
| **Determinant** | | | |
| `det(A)` | 1.783 ns | 225.507 ns | ×126.5 |
| `det(S)` | 2.029 ns | 227.032 ns | ×111.9 |
| **Inverse** | | | |
| `inv(A)` | 6.929 ns | 541.723 ns | ×78.2 |
| `inv(S)` | 4.979 ns | 517.937 ns | ×104.0 |
| `inv(AA)` | 894.583 ns | 1.665 μs | ×1.9 |
| `inv(SS)` | 372.068 ns | 1.661 μs | ×4.5 |

The benchmarks are generated by
[`runbenchmarks.jl`](https://github.com/KeitaNakamura/Tensorial.jl/blob/master/benchmark/runbenchmarks.jl)
on the following system:

```julia
julia> versioninfo()
Julia Version 1.5.3
Commit 788b2c77c1 (2020-11-09 13:37 UTC)
Platform Info:
  OS: macOS (x86_64-apple-darwin18.7.0)
  CPU: Intel(R) Core(TM) i7-7567U CPU @ 3.50GHz
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-9.0.1 (ORCJIT, skylake)
```

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
