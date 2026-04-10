# Getting started

## Quick start

```@setup quick-start
using Tensorial
```

```@repl quick-start
using Tensorial
x = Vec{3}(rand(3)); # constructor analogous to SArray.jl
A = @Mat rand(3,3); # @Vec, @Mat, and @Tensor, analogous to @SVector, @SMatrix, and @SArray
A ⊡ x ≈ A * x # single contraction (⊡)
A ⊡₂ A ≈ A ⋅ A # double contraction (⊡₂)
x ⊗ x ≈ x * x' # tensor product (⊗)
(@einsum x[i] * A[j,i] * x[j]) ≈ x ⋅ (A' * x) # Einstein summation with @einsum
S = rand(Tensor{Tuple{@Symmetry{3,3}}}); # symmetric tensor S₍ᵢⱼ₎
SS = rand(Tensor{Tuple{@Symmetry{3,3}, @Symmetry{3,3}}}); # symmetric tensor SS₍ᵢⱼ₎₍ₖₗ₎
inv(SS) ⊡₂ S ≈ @einsum inv(SS)[i,j,k,l] * S[k,l] # works as expected
δ = one(Mat{3,3}) # identity tensor
gradient(identity, S) ≈ one(SS) # ∂Sᵢⱼ/∂Sₖₗ = (δᵢₖδⱼₗ + δᵢₗδⱼₖ) / 2
```

## Defining tensors

### 1. `Tensor`

All tensors in Tensorial.jl are represented by the type `Tensor{S, T, N, L}`, where the type parameters have the following meanings:

- `S`: the tensor size, specified as a `Tuple` (for example, a 3×2 tensor is written as `Tensor{Tuple{3,2}}`)
- `T`: the element type, where `T <: Real`
- `N`: the tensor order
- `L`: the number of independent components

The type parameters `N` and `L` do not need to be specified when [Constructing tensors](@ref), since they can be inferred from the size parameter `S`.
When using `Tensor` fields in a `struct`, however, it is necessary to declare all type parameters explicitly to ensure type stability, as shown below:

```@setup tensors-in-struct
using Tensorial
```

```@example tensors-in-struct
struct MyBadType{T} # all bad
    A::Tensor{Tuple{3,3}, Float64}
    B::Tensor{Tuple{3,3}, T}
    C::Tensor{Tuple{@Symmetry{3,3}}, T, 2}
end

struct MyGoodType{T, dim, L, TT <: Tensor} # all good
    A::Tensor{Tuple{3,3}, Float64, 2, 9}
    B::Tensor{Tuple{3,3}, T, 2, 9}
    C::Tensor{Tuple{@Symmetry{dim,dim}}, T, 2, L}
    D::TT
end
```

!!! tip
    The type parameters `N` and `L` can be checked using the `@Tensor` macro:

    ```@repl tensors-in-struct
    @Tensor{Tuple{@Symmetry{3,3,3}}}
    ```

### 2. `Symmetry`

Specifying tensor symmetry can improve performance, since Tensorial.jl eliminates duplicate computations.
Symmetry can be encoded in the size parameter `S` using `Symmetry{...}`.
The `@Symmetry` macro simplifies this by letting you omit `Tuple`, as in `@Symmetry{3,3}`.

Below are some examples:

- ``A_{(ij)}`` with 3×3: `Tensor{Tuple{@Symmetry{3,3}}}`
- ``A_{(ij)k}`` with 3×3×3: `Tensor{Tuple{@Symmetry{3,3}, 3}}`
- ``A_{(ijk)}`` with 3×3×3: `Tensor{Tuple{@Symmetry{3,3,3}}}`
- ``A_{(ij)(kl)}`` with 3×3×3×3: `Tensor{Tuple{@Symmetry{3,3}, @Symmetry{3,3}}}`

where parentheses in the indices denote symmetry.

## Aliases

```julia
const Vec{dim, T} = Tensor{Tuple{dim}, T, 1, dim}
const Mat{m, n, T, L} = Tensor{Tuple{m, n}, T, 2, L}
const SecondOrderTensor{dim, T, L} = Tensor{NTuple{2, dim}, T, 2, L}
const FourthOrderTensor{dim, T, L} = Tensor{NTuple{4, dim}, T, 4, L}
const SymmetricSecondOrderTensor{dim, T, L} = Tensor{Tuple{@Symmetry{dim, dim}}, T, 2, L}
const SymmetricFourthOrderTensor{dim, T, L} = Tensor{NTuple{2, @Symmetry{dim, dim}}, T, 4, L}
```
