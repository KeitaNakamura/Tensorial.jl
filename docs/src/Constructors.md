```@meta
DocTestSetup = :(using Tensorial)
```

# Constructors

## Identity tensors

### Second order tensor

```jldoctest
julia> one(SecondOrderTensor{3})
3×3 Tensor{Tuple{3, 3}, Float64, 2, 9}:
 1.0  0.0  0.0
 0.0  1.0  0.0
 0.0  0.0  1.0

julia> one(SymmetricSecondOrderTensor{3})
3×3 SymmetricSecondOrderTensor{3, Float64, 6}:
 1.0  0.0  0.0
 0.0  1.0  0.0
 0.0  0.0  1.0

julia> one(Mat{2,2})
2×2 Tensor{Tuple{2, 2}, Float64, 2, 4}:
 1.0  0.0
 0.0  1.0
```

### Fourth order tensor

```jldoctest
julia> one(FourthOrderTensor{3})
3×3×3×3 FourthOrderTensor{3, Float64, 81}:
[:, :, 1, 1] =
 1.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0

[:, :, 2, 1] =
 0.0  0.0  0.0
 1.0  0.0  0.0
 0.0  0.0  0.0

[:, :, 3, 1] =
 0.0  0.0  0.0
 0.0  0.0  0.0
 1.0  0.0  0.0

[:, :, 1, 2] =
 0.0  1.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0

[:, :, 2, 2] =
 0.0  0.0  0.0
 0.0  1.0  0.0
 0.0  0.0  0.0

[:, :, 3, 2] =
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  1.0  0.0

[:, :, 1, 3] =
 0.0  0.0  1.0
 0.0  0.0  0.0
 0.0  0.0  0.0

[:, :, 2, 3] =
 0.0  0.0  0.0
 0.0  0.0  1.0
 0.0  0.0  0.0

[:, :, 3, 3] =
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  1.0

julia> one(SymmetricFourthOrderTensor{3})
3×3×3×3 SymmetricFourthOrderTensor{3, Float64, 36}:
[:, :, 1, 1] =
 1.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0

[:, :, 2, 1] =
 0.0  0.5  0.0
 0.5  0.0  0.0
 0.0  0.0  0.0

[:, :, 3, 1] =
 0.0  0.0  0.5
 0.0  0.0  0.0
 0.5  0.0  0.0

[:, :, 1, 2] =
 0.0  0.5  0.0
 0.5  0.0  0.0
 0.0  0.0  0.0

[:, :, 2, 2] =
 0.0  0.0  0.0
 0.0  1.0  0.0
 0.0  0.0  0.0

[:, :, 3, 2] =
 0.0  0.0  0.0
 0.0  0.0  0.5
 0.0  0.5  0.0

[:, :, 1, 3] =
 0.0  0.0  0.5
 0.0  0.0  0.0
 0.5  0.0  0.0

[:, :, 2, 3] =
 0.0  0.0  0.0
 0.0  0.0  0.5
 0.0  0.5  0.0

[:, :, 3, 3] =
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  1.0
```

## Other special tensors

### Levi-Civita

```@docs
levicivita
```
