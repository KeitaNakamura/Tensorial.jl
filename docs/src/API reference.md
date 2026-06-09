```@meta
CurrentModule = Tensorial
DocTestSetup = :(using Tensorial; using LinearAlgebra)
```

# API Reference

This page collects the public Tensorial API. Use the manual pages for examples
and workflows, and use this page to look up signatures and detailed behavior.

- [Constructors and special tensors](@ref)
- [Symmetry API](@ref)
- [Core operations](@ref)
- [Continuum mechanics API](@ref)
- [Rotations and quaternions](@ref)
- [Spectral functions](@ref)
- [Automatic differentiation API](@ref)
- [Direct sum API](@ref)
- [Voigt and Mandel API](@ref)

## Tensor types

Tensorial values are statically sized `AbstractArray`s whose type records the
tensor space.

- `Tensor{S, T, N, L}` is the general tensor type.
- `Vec{dim, T}` and `Mat{m, n, T, L}` are vector and matrix aliases.
- `SecondOrderTensor{dim, T, L}` and `FourthOrderTensor{dim, T, L}` are square
  tensor aliases.
- `SymmetricSecondOrderTensor{dim, T, L}` and
  `SymmetricFourthOrderTensor{dim, T, L}` use symmetric tensor spaces.

See [Tensor Types and Spaces](@ref) and [Constructing Tensors](@ref) for
examples.

```@docs
Tensor
Vec
Mat
SecondOrderTensor
FourthOrderTensor
SymmetricSecondOrderTensor
SymmetricFourthOrderTensor
```

## Constructors and special tensors

```@docs
@Vec
@Mat
@Tensor
one
zero
levicivita
```

## Symmetry API

```@docs
Symmetry
@Symmetry
symmetric
skew
minorsymmetric
```

## Core operations

```@docs
cross
norm
normalize
tr
inv
contract
tensor
^
@einsum
```

## Continuum mechanics API

```@docs
vol
dev
vonmises
stress_invariants
deviatoric_stress_invariants
```

## Rotations and quaternions

```@docs
rotmatx
rotmaty
rotmatz
rotmat
rotate
angleaxis
Quaternion
quaternion
Base.exp(::Quaternion)
Base.log(::Quaternion)
```

## Spectral functions

```@docs
Base.sqrt(::Tensorial.AbstractSymmetricSecondOrderTensor)
Base.exp(::Tensorial.AbstractSymmetricSecondOrderTensor)
Base.log(::Tensorial.AbstractSymmetricSecondOrderTensor)
```

## Automatic differentiation API

```@docs
gradient
hessian
∂
```

## Direct sum API

```@docs
DirectSumArray
DirectSumVector
DirectSumMatrix
pack
unpack
flatview
flatsize
```

## Voigt and Mandel API

```@docs
tovoigt
tomandel
fromvoigt
frommandel
```
