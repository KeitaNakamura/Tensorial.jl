```@meta
DocTestSetup = :(using Tensorial)
```

# Continuum Mechanics

Tensor operations for continuum mechanics.

```@docs
mean(::Tensorial.AbstractSquareTensor{3})
```

## Deviatoric--volumetric additive split

```@docs
vol
dev
```

## Stress invariants

```@docs
stress_invariants
deviatoric_stress_invariants
```
