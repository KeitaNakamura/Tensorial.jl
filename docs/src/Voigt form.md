```@meta
CurrentModule = Tensorial
DocTestSetup = :(using Tensorial)
```

# Voigt Form

Tensorial code usually does not need Voigt or Mandel arrays: tensor operations
can be written directly with tensor types. The conversion functions are useful
at boundaries, for example when exchanging data with a solver or file format
that expects vectors and matrices.

```@setup voigt-form
using Tensorial
```

## Second-order tensors

[`tovoigt`](@ref) converts a tensor to its Voigt vector. For symmetric 3D
second-order tensors, the default order is `(11, 22, 33, 23, 13, 12)`:

```@repl voigt-form
σ = SymmetricSecondOrderTensor{3}((2.0, 0.4, 0.2, 1.2, 0.1, 0.9))
v = tovoigt(σ)
fromvoigt(SymmetricSecondOrderTensor{3}, v) ≈ σ
```

The inverse conversion must use the same order and scaling that were used to
create the Voigt vector.

## Mandel scaling

For symmetric tensors, Mandel form scales off-diagonal components by `√2`.
This makes the Euclidean dot product of the vector representation match the
tensor inner product:

```@repl voigt-form
m = tomandel(σ)
m ≈ tovoigt(σ; offdiagscale = sqrt(2.0))
frommandel(SymmetricSecondOrderTensor{3}, m) ≈ σ
```

Use `tovoigt(...; offdiagscale = 2)` for engineering-strain style scaling, and
use the same `offdiagscale` in [`fromvoigt`](@ref) to reconstruct the tensor.

## Fourth-order tensors

Fourth-order tensors are converted to matrices in Voigt or Mandel form:

```@repl voigt-form
I = one(SymmetricFourthOrderTensor{3});
M = tomandel(I);
size(M)
frommandel(SymmetricFourthOrderTensor{3}, M) ≈ I
```

For Voigt and Mandel conversion docstrings, see [Voigt and Mandel API](@ref).
