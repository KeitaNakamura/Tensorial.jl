```@meta
CurrentModule = Tensorial
DocTestSetup = :(using Tensorial)
```

# Tensor Types and Spaces

Tensorial values are statically sized `AbstractArray`s. The extra information is
in the type: it records the tensor space, including order, dimension, and
symmetry. This page explains how to read those types.

## Reading tensor types

Start with the kinds of values you will see in ordinary code. The values print
like arrays, but their types carry more information:

```@repl spaces
using Tensorial
x = @Vec [1.0, 2.0, 3.0];
A = @Mat [1.0 2.0 3.0; 4.0 5.0 6.0];
typeof(x)
typeof(A)
```

The first type says that `x` is a 3-dimensional vector with `Float64`
components. The second says that `A` is a `2×3` matrix tensor. `Vec` and `Mat`
are convenient aliases; the fully general form is `Tensor{S, T, N, L}`:

- `S` describes the tensor space.
- `T` is the element type.
- `N` is the tensor order.
- `L` is the number of stored independent components.

For ordinary use, `S` and `T` are the important parts to read. The last two
parameters, `N` and `L`, are derived from the tensor space: they record the
tensor order and the number of stored components.

## Tensor spaces

The first parameter, `S`, is a tuple describing the index spaces. Each bare
integer represents one ordinary index space:

- `Tensor{Tuple{3}}` represents a vector ``a_i`` with `i = 1,2,3`.
- `Tensor{Tuple{3,3}}` represents a second-order tensor ``A_{ij}``.
- `Tensor{Tuple{3,3,3}}` represents a third-order tensor ``A_{ijk}``.

This is why `S` is written as a tuple: each entry contributes one index space.
For ordinary dense tensor spaces, the entries are just dimensions.

The common aliases follow this pattern:

- `Vec{dim}` for `Tensor{Tuple{dim}}`.
- `Mat{m,n}` for `Tensor{Tuple{m,n}}`.
- `SecondOrderTensor{dim}` for `Tensor{Tuple{dim,dim}}`.
- `FourthOrderTensor{dim}` for `Tensor{Tuple{dim,dim,dim,dim}}`.

## Symmetric spaces

Symmetric tensors use the same idea, but one entry of `S` can represent a whole
group of symmetric indices. Write that group with `@Symmetry`. For example,
`@Symmetry{dim,dim}` represents two symmetric indices of the same dimension, so
`Tensor{Tuple{@Symmetry{3,3}}}` means a tensor ``A_{(ij)}`` with
`i,j = 1,2,3`.

The constructor below takes the six stored independent components. Their order
is shown later in [Inspecting tensor spaces](@ref).

```@repl spaces
S = Tensor{Tuple{@Symmetry{3,3}}}((1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
Tuple(S)
```

The displayed value still behaves like a `3×3` tensor, but the tuple stores only
the independent components. The type is equivalent to the common alias
`SymmetricSecondOrderTensor{3}`.

`@Symmetry{3,3}` is shorthand for `Symmetry{Tuple{3,3}}`. Parentheses in the
index notation below indicate one symmetric group:

- `Tensor{Tuple{3, 3}}` represents a general ``A_{ij}`` and stores 9
  components.
- `Tensor{Tuple{@Symmetry{3,3}}}` represents ``A_{(ij)}`` and stores 6
  components.
- `Tensor{Tuple{@Symmetry{3,3}, 3}}` represents ``A_{(ij)k}`` and stores 18
  components.
- `Tensor{Tuple{@Symmetry{3,3,3}}}` represents ``A_{(ijk)}`` and stores 10
  components.
- `Tensor{Tuple{@Symmetry{3,3}, @Symmetry{3,3}}}` represents
  ``A_{(ij)(kl)}`` and stores 36 components.

The common aliases are:

- `SymmetricSecondOrderTensor{dim}` for `Tensor{Tuple{@Symmetry{dim,dim}}}`.
- `SymmetricFourthOrderTensor{dim}` for
  `Tensor{Tuple{@Symmetry{dim,dim}, @Symmetry{dim,dim}}}`.

## Why it matters

Once symmetry is part of the tensor space, later operations use that space. For
example, automatic differentiation returns derivatives in matching tensor
spaces:

```@repl spaces
ε = SymmetricSecondOrderTensor{3}((0.02, 0.01, 0.0, 0.0, 0.0, -0.01));
ψ(ε) = tr(ε)^2;
gradient(ψ, ε) isa SymmetricSecondOrderTensor{3}
hessian(ψ, ε) isa SymmetricFourthOrderTensor{3}
```

The result is not just numerically symmetric; the returned type records the
symmetric tensor space as well.

## Inspecting tensor spaces

The main interface is the tensor type itself, but a few helpers are useful when
you want to inspect what the type means.

`@Tensor{Tuple{...}}` turns a tensor-space description into the corresponding
Tensorial type. This is a compact way to see the inferred order `N` and stored
component count `L`:

```@repl spaces
@Tensor{Tuple{2,3}}
@Tensor{Tuple{2,3},Float64}
@Tensor{Tuple{@Symmetry{3,3},2}}
```

In the last type, `N = 3` because the space represents ``A_{(ij)k}``, and
`L = 12` because the symmetric `3×3` group stores 6 independent components and
the last index has dimension 2.

The same tensor-space information can be shown as a `Tensorial.Space` value:

```@repl spaces
Tensorial.Space(Tuple{3,3})
Tensorial.Space(Tuple{@Symmetry{3,3}})
```

For a symmetric tensor, `Tensorial.component_to_independent_map` shows which
entry of the stored tuple is used at each displayed index position:

```@repl spaces
S = SymmetricSecondOrderTensor{3}((11, 21, 31, 22, 32, 33))
Tuple(S)
Tensorial.component_to_independent_map(S)
```

The map tells you, for example, that display positions `(1,2)` and `(2,1)` both
use the second stored component, while `(3,3)` uses the sixth stored component.

For tensor type docstrings, see [Tensor types](@ref). For symmetry-related
docstrings, see [Symmetry API](@ref).
