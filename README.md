# Tensorial

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://KeitaNakamura.github.io/Tensorial.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://KeitaNakamura.github.io/Tensorial.jl/dev)
[![Build Status](https://github.com/KeitaNakamura/Tensorial.jl/workflows/CI/badge.svg)](https://github.com/KeitaNakamura/Tensorial.jl/actions)
[![codecov](https://codecov.io/gh/KeitaNakamura/Tensorial.jl/branch/main/graph/badge.svg?token=V58DXDI1R5)](https://codecov.io/gh/KeitaNakamura/Tensorial.jl)


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

## Cheat Sheet

### Constructors

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

# from arrays
Tensor{Tuple{2,2}}([1 2; 3 4]) == [1 2; 3 4]
Tensor{Tuple{@Symmetry{2,2}}}([1 2; 3 4]) == [1 3; 3 4] # lower triangular part is used

# from functions
Tensor{Tuple{2,2}}((i,j) -> i == j ? 1 : 0) == one(Tensor{Tuple{2,2}})
Tensor{Tuple{@Symmetry{2,2}}}((i,j) -> i == j ? 1 : 0) == one(Tensor{Tuple{@Symmetry{2,2}}})
```

### Tensor Operations

```julia
# 2nd-order vs. 2nd-order
x = rand(Tensor{Tuple{2,2}})
y = rand(Tensor{Tuple{@Symmetry{2,2}}})
x ⊗ y isa Tensor{Tuple{2,2,@Symmetry{2,2}}} # tensor product
x ⋅ y isa Tensor{Tuple{2,2}}                # single contraction
x ⊡ y isa Real                              # double contraction

# 3rd-order vs. 1st-order
A = rand(Tensor{Tuple{@Symmetry{2,2},2}})
v = rand(Vec{2})
A ⊗ v isa Tensor{Tuple{@Symmetry{2,2},2,2}}
A ⋅ v isa Tensor{Tuple{@Symmetry{2,2}}}
A ⊡ v # error

# 4th-order vs. 2nd-order
II = one(SymmetricFourthOrderTensor{2}) # equal to one(Tensor{Tuple{@Symmetry{2,2}, @Symmetry{2,2}}})
A = rand(Tensor{Tuple{2,2}})
S = rand(Tensor{Tuple{@Symmetry{2,2}}})
II ⊡ A == (A + A') / 2 == symmetric(A) # symmetrizing A, resulting in Tensor{Tuple{@Symmetry{2,2}}}
II ⊡ S == S

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

### Automatic differentiation

```julia
# Real -> Real
gradient(x -> 2x^2 + x + 3, 3) == (x = 3; 4x + 1)
gradient(x -> 2.0, 3) == 0.0

# Real -> Tensor
gradient(x -> Tensor{Tuple{2,2}}((i,j) -> i*x^2), 3) == (x = 3; Tensor{Tuple{2,2}}((i,j) -> 2i*x))
gradient(x -> one(Tensor{Tuple{2,2}}), 3) == zero(Tensor{Tuple{2,2}})

# Tensor -> Real
gradient(tr, rand(Tensor{Tuple{3,3}})) == one(Tensor{Tuple{3,3}})

# Tensor -> Tensor
A = rand(Tensor{Tuple{3,3}})
D  = gradient(dev, A)            # deviatoric projection tensor
Ds = gradient(dev, symmetric(A)) # symmetric deviatoric projection tensor
A ⊡ D  ≈ dev(A)
A ⊡ Ds ≈ symmetric(dev(A))
gradient(identity, A)  == one(FourthOrderTensor{3})          # 4th-order identity tensor
gradient(symmetric, A) == one(SymmetricFourthOrderTensor{3}) # symmetric 4th-order identity tensor
```

### Aliases

```julia
const SecondOrderTensor{dim, T, L} = Tensor{NTuple{2, dim}, T, 2, L}
const FourthOrderTensor{dim, T, L} = Tensor{NTuple{4, dim}, T, 4, L}
const SymmetricSecondOrderTensor{dim, T, L} = Tensor{Tuple{@Symmetry{dim, dim}}, T, 2, L}
const SymmetricFourthOrderTensor{dim, T, L} = Tensor{NTuple{2, @Symmetry{dim, dim}}, T, 4, L}
const Mat{m, n, T, L} = Tensor{Tuple{m, n}, T, 2, L}
const Vec{dim, T} = Tensor{Tuple{dim}, T, 1, dim}
```

## Other tensor packages

- [Einsum.jl](https://github.com/ahwillia/Einsum.jl)
- [TensorOprations.jl](https://github.com/Jutho/TensorOperations.jl)
- [Tensors.jl](https://github.com/KristofferC/Tensors.jl)

## Inspiration

- [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl)
- [Tensors.jl](https://github.com/KristofferC/Tensors.jl)
