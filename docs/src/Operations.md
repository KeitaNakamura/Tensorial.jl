```@meta
DocTestSetup = :(using Tensorial)
```

# Operations

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

## Continuum mechanics

```@docs
vol(::Tensorial.AbstractSquareTensor{3})
dev(::Tensorial.AbstractSquareTensor{3})
vonmises
stress_invariants
deviatoric_stress_invariants
```
