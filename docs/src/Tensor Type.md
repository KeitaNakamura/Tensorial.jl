# `Tensor` type

## Type parameters

All tensors are represented by a type `Tensor{S, T, N, L}` where each type parameter represents following:

- `S`: The size of `Tensor`s which is specified by using `Tuple` (e.g., 3x2 tensor becomes `Tensor{Tuple{3,2}}`).
- `T`: The type of element which must be `T <: Real`.
- `N`: The number of dimensions (the order of tensor).
- `L`: The number of independent components.

Basically, the type parameters `N` and `L` do not need to be specified for constructing tensors because it can be inferred from the size of tensor `S`.

## `Symmetry`

If possible, specifying the symmetry of the tensor is good for performance since Tensorial.jl provides the optimal computations.
The symmetries can be applied using `Symmetry` in type parameter `S` (e.g., `Symmetry{Tuple{3,3}}`).
`@Symmetry` macro can omit `Tuple` like `@Symmetry{2,2}`.
The following are examples to specify symmetries:

- ``A_{(ij)}`` with 3x3: `Tensor{Tuple{@Symmetry{3,3}}}`
- ``A_{(ij)k}`` with 3x3x2: `Tensor{Tuple{@Symmetry{3,3}, 2}}`
- ``A_{(ijk)}`` with 3x3x3: `Tensor{Tuple{@Symmetry{3,3,3}}}`
- ``A_{(ij)(kl)}`` with 3x3x3x3: `Tensor{Tuple{@Symmetry{3,3}, @Symmetry{3,3}}}`

where the bracket ``()`` in indices denotes the symmetry.
