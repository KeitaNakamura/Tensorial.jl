# Getting started

## Quick start

```@setup quick-start
using Tensorial
```

```@repl quick-start
using Tensorial
x = Vec{3}(rand(3)); # constructor similar to SArray.jl
A = @Mat rand(3,3); # @Vec, @Mat and @Tensor, analogous to @SVector, @SMatrix and @SArray
A ⋅ x ≈ A * x # single contraction (⋅)
A ⊡ A ≈ tr(A'A) # double contraction (⊡)
x ⊗ x ≈ x * x' # tensor product (⊗)
(@einsum x[i] * A[j,i] * x[j]) ≈ x ⋅ A' ⋅ x # Einstein summation (@einsum)
S = rand(Tensor{Tuple{@Symmetry{3,3}}}); # specify symmetry S₍ᵢⱼ₎
SS = rand(Tensor{Tuple{@Symmetry{3,3}, @Symmetry{3,3}}}); # SS₍ᵢⱼ₎₍ₖₗ₎
inv(SS) ⊡ S ≈ @einsum inv(SS)[i,j,k,l] * S[k,l] # it just works
δ = one(Mat{3,3}) # identity tensor
gradient(identity, S) ≈ one(SS) # ∂Sᵢⱼ/∂Sₖₗ = (δᵢₖδⱼₗ + δᵢₗδⱼₖ) / 2
```

## Defining tensors

### 1. `Tensor`

All tensors in Tensorial.jl are represented by the type `Tensor{S, T, N, L}`, where each type parameter represents the following:

- `S`: The size of `Tensor`s which is specified by using `Tuple` (e.g., 3×2 tensor becomes `Tensor{Tuple{3,2}}`).
- `T`: The type of element which must be `T <: Real`.
- `N`: The number of dimensions (the order of tensor).
- `L`: The number of independent components.

The type parameters `N` and T do not need to be specified when [Constructing tensors](@ref), as they can be inferred from the size of tensor `S`.
However, when defining `Tensor`s in a `struct`, it is necessary to declare all type parameters to avoid type instability, as follows:

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
    The type parameters `N` and `L` can be checked using the `@Tensor` macro as follows:
    ```@repl tensors-in-struct
    @Tensor{Tuple{@Symmetry{3,3,3}}}
    ```

### 2. `Symmetry`

Specifying the symmetry of a tensor can improve performance, as Tensorial.jl eliminates duplicate computations. Symmetries can be applied using `Symmetry` in the type parameter `S` (e.g., `Symmetry{Tuple{3,3}}`). The `@Symmetry` macro simplifies this process by allowing you to omit `Tuple`, as in `@Symmetry{3,3}`. Below are some examples of how to specify symmetries:

* ``A_{(ij)}`` with 3x3: `Tensor{Tuple{@Symmetry{3,3}}}`
* ``A_{(ij)k}`` with 3x3x3: `Tensor{Tuple{@Symmetry{3,3}, 3}}`
* ``A_{(ijk)}`` with 3x3x3: `Tensor{Tuple{@Symmetry{3,3,3}}}`
* ``A_{(ij)(kl)}`` with 3x3x3x3: `Tensor{Tuple{@Symmetry{3,3}, @Symmetry{3,3}}}`

where the bracket ``()`` in the indices denotes the symmetry.

## Aliases

```julia
const Vec{dim, T} = Tensor{Tuple{dim}, T, 1, dim}
const Mat{m, n, T, L} = Tensor{Tuple{m, n}, T, 2, L}
const SecondOrderTensor{dim, T, L} = Tensor{NTuple{2, dim}, T, 2, L}
const FourthOrderTensor{dim, T, L} = Tensor{NTuple{4, dim}, T, 4, L}
const SymmetricSecondOrderTensor{dim, T, L} = Tensor{Tuple{@Symmetry{dim, dim}}, T, 2, L}
const SymmetricFourthOrderTensor{dim, T, L} = Tensor{NTuple{2, @Symmetry{dim, dim}}, T, 4, L}
```
