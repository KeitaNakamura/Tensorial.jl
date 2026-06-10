```@meta
CurrentModule = Tensorial
DocTestSetup = :(using Tensorial; using LinearAlgebra)
```

# Operations

Tensorial operations act on tensor values directly. When an operation returns a
tensor, the result keeps the corresponding tensor space.

This page shows the common operations used in formulas. Detailed signatures are
collected in the [API Reference](@ref).

```@setup operations
using Tensorial
using LinearAlgebra
```

## Basic array-like operations

Tensorial values are `AbstractArray`s, so ordinary Julia and `LinearAlgebra`
functions work as expected:

```@repl operations
x = @Vec [1.0, 2.0, 3.0]
A = @Mat [1.0 2.0 0.0; 0.0 3.0 4.0; 5.0 0.0 6.0]
norm(x)
tr(A)
```

Julia's uniform-scaling identity `I` can be used in tensor formulas. For a
symmetric tensor, adding `I` keeps the result symmetric:

```@repl operations
S = symmetric(@Mat [1.0 2.0; 2.0 4.0])
S + I
S ⊡₂ I ≈ tr(S)
```

## Inverse of tensor operators

Even-order tensors can be inverted when they represent square linear operators:

```@repl operations
C = rand(SymmetricFourthOrderTensor{3});
inv(C) ⊡₂ C ≈ one(SymmetricFourthOrderTensor{3})
```

## Contractions and tensor products

Use `⊡`, `⊡₂`, and `⊗` to write contractions and tensor products in tensor
notation:

```@repl operations
A ⊡ x
A ⊡ x ≈ A * x
A ⊡₂ A
x ⊗ x
contract(A, x, Val(1)) ≈ A ⊡ x
```

- `⊡` contracts one neighboring index.
- `⊡₂` contracts two neighboring indices.
- `⊗` forms a tensor product.

The ASCII aliases are `contract1`, `contract2`, and `tensor`.

## Indexed formulas

Use [`@einsum`](@ref) for formulas that are easier to read with indices.
Repeated indices are contracted:

```@repl operations
(@einsum A[i,k] * A[k,j]) ≈ A * A
(@einsum A[i,j] * A[i,j]) ≈ A ⋅ A
```

Free indices can also be written explicitly when the output order matters:

```@repl operations
(@einsum (j,i) -> A[i,k] * A[k,j]) ≈ transpose(A * A)
```

For a named result, put the free indices on the left-hand side:

```@repl operations
@einsum B[i,j] := A[i,k] * A[k,j];
B ≈ A * A
```

## Symmetric and skew parts

[`symmetric`](@ref) computes the symmetric part of a second-order tensor.
[`skew`](@ref) computes the skew part:

```@repl operations
B = @Mat [1.0 3.0; 2.0 4.0]
symmetric(B)
skew(B)
symmetric(B) + skew(B) ≈ B
```

For the type-level symmetry notation, see [Tensor Types and Spaces](@ref).

## Continuum mechanics helpers

The `vol` and `dev` helpers split a 3D tensor into volumetric and deviatoric
parts. Fourth-order projector tensors are also available:

```@repl operations
σ = SymmetricSecondOrderTensor{3}((2.0, 0.4, 0.2, 1.2, 0.1, 0.9))
vol(σ) + dev(σ) ≈ σ
tr(dev(σ))
vonmises(σ)
Ivol = vol(SymmetricFourthOrderTensor{3});
Idev = dev(SymmetricFourthOrderTensor{3});
(Ivol + Idev) ⊡₂ σ ≈ σ
```

## Rotations and spectral functions

Rotation helpers work with Tensorial vectors and tensors:

```@repl operations
R = rotmatz(π/4)
R * (@Vec [1.0, 0.0, 0.0])
rotate(σ, R) isa SymmetricSecondOrderTensor{3}
```

For symmetric second-order tensors, spectral functions such as `sqrt`, `exp`,
and `log` return symmetric tensor types:

```@repl operations
sqrt(one(SymmetricSecondOrderTensor{3})) ≈ one(SymmetricSecondOrderTensor{3})
exp(zero(SymmetricSecondOrderTensor{3})) ≈ one(SymmetricSecondOrderTensor{3})
```

Use [`spectral`](@ref) to apply another scalar function to the eigenvalues:

```@repl operations
S = symmetric(A' * A)
spectral(x -> x^2 + 2x, S) ≈ S^2 + 2S
```

For operation docstrings, see [Core operations](@ref),
[Continuum mechanics API](@ref), [Rotations and quaternions](@ref), and
[Spectral functions](@ref).
