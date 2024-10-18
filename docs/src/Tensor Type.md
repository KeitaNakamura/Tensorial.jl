# `Tensor` type

## Type parameters

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

## `Symmetry`

Specifying the symmetry of a tensor can improve performance, as Tensorial.jl eliminates duplicate computations. Symmetries can be applied using `Symmetry` in the type parameter `S` (e.g., `Symmetry{Tuple{3,3}}`). The `@Symmetry` macro simplifies this process by allowing you to omit `Tuple`, as in `@Symmetry{2,2}`. Below are some examples of how to specify symmetries:

* ``A_{(ij)}`` with 3x3: `Tensor{Tuple{@Symmetry{3,3}}}`
* ``A_{(ij)k}`` with 3x3x2: `Tensor{Tuple{@Symmetry{3,3}, 2}}`
* ``A_{(ijk)}`` with 3x3x3: `Tensor{Tuple{@Symmetry{3,3,3}}}`
* ``A_{(ij)(kl)}`` with 3x3x3x3: `Tensor{Tuple{@Symmetry{3,3}, @Symmetry{3,3}}}`

where the bracket ``()`` in indices denotes the symmetry.
