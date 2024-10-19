```@meta
DocTestSetup = :(using Tensorial)
```

# Constructing tensors

## From an `AbstractArray`

```@setup construct-from-abstractarray
using Tensorial
```

```@repl construct-from-abstractarray
Vec{2}([1,2])
Vec{2,Float64}([1,2])
Mat{2,2}([1 2; 3 4])
Mat{2,2,Float64}([1 2; 3 4])
SymmetricSecondOrderTensor{2}([1 2; 3 4]) # InexactError
SymmetricSecondOrderTensor{2}([1 2; 2 4])
```

## From a function

```@setup construct-from-function
using Tensorial
```

```@repl construct-from-function
δ = one(Mat{2,2})
I = SymmetricFourthOrderTensor{2}((i,j,k,l) -> (δ[i,k]*δ[j,l] + δ[i,l]*δ[j,k])/2)
I == one(SymmetricFourthOrderTensor{2})
```

## Identity tensors

```@docs
one
```

## Zero tensors

```@docs
zero
```

## Macros

```@docs
@Vec
@Mat
@Tensor
```

## Other special tensors

### Levi-Civita

```@docs
levicivita
```
