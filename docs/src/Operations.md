```@meta
DocTestSetup = :(using Tensorial)
```

# Operations

```@index
Pages = ["Operations.md"]
```

## Basic operations

```@docs
cross
norm
normalize
tr
inv
```

## Tensor operations

```@docs
contract
tensor
^
@einsum
```

## Symmetry

```@docs
symmetric
skew
minorsymmetric
```

## Rotation

```@docs
rotmatx
rotmaty
rotmatz
rotmat
rotate
```

## Spectral functions

```@docs
sqrt(::Tensorial.AbstractSymmetricSecondOrderTensor)
exp(::Tensorial.AbstractSymmetricSecondOrderTensor)
log(::Tensorial.AbstractSymmetricSecondOrderTensor)
```

## Continuum mechanics

```@docs
vol(::Tensorial.AbstractSquareTensor{3})
dev(::Tensorial.AbstractSquareTensor{3})
vonmises
stress_invariants
deviatoric_stress_invariants
```
