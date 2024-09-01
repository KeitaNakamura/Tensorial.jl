# `Tensor` type

## Type parameters

All tensors in Tensorial.jl are represented by the type `Tensor{S, T, N, L}`, where each type parameter represents the following:

- `S`: The size of `Tensor`s which is specified by using `Tuple` (e.g., 3x2 tensor becomes `Tensor{Tuple{3,2}}`).
- `T`: The type of element which must be `T <: Real`.
- `N`: The number of dimensions (the order of tensor).
- `L`: The number of independent components.

Basically, the type parameters `N` and `L` do not need to be specified for constructing tensors because it can be inferred from the size of tensor `S`.

## `Symmetry`

Specifying the symmetry of a tensor can improve performance, as Tensorial.jl provides optimized computations for symmetric tensors. Symmetries can be applied using `Symmetry` in the type parameter `S` (e.g., `Symmetry{Tuple{3,3}}`). The `@Symmetry` macro simplifies this process by allowing you to omit `Tuple`, as in `@Symmetry{2,2}`. Below are some examples of how to specify symmetries:

- ``A_{(ij)}`` with 3x3: `Tensor{Tuple{@Symmetry{3,3}}}`
- ``A_{(ij)k}`` with 3x3x2: `Tensor{Tuple{@Symmetry{3,3}, 2}}`
- ``A_{(ijk)}`` with 3x3x3: `Tensor{Tuple{@Symmetry{3,3,3}}}`
- ``A_{(ij)(kl)}`` with 3x3x3x3: `Tensor{Tuple{@Symmetry{3,3}, @Symmetry{3,3}}}`

where the bracket ``()`` in indices denotes the symmetry.
