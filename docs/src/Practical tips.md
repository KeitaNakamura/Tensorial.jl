```@meta
CurrentModule = Tensorial
DocTestSetup = :(using Tensorial; using LinearAlgebra)
```

# Practical Tips

This page collects small practical notes for writing Tensorial code clearly and
efficiently.

- [Use concrete tensor field types](#Use-concrete-tensor-field-types)
- [Put known symmetry in the type](#Put-known-symmetry-in-the-type)
- [Specify `@einsum` result types when needed](#Specify-@einsum-result-types-when-needed)
- [Flatten at boundaries](#Flatten-at-boundaries)
- [Type inference when unpacking](#Type-inference-when-unpacking)

```@setup practical-tips
using Tensorial
using LinearAlgebra
```

## Use concrete tensor field types

Tensor constructors can infer the tensor order `N` and the number of independent
components `L`:

```@repl practical-tips
@Tensor{Tuple{@Symmetry{3,3}}}
```

When a tensor type appears as a `struct` field, include all type parameters so
the field type is concrete:

```@repl practical-tips
isconcretetype(Tensor{Tuple{@Symmetry{3,3}}, Float64})
isconcretetype(Tensor{Tuple{@Symmetry{3,3}}, Float64, 2, 6})
```

For example, prefer the fully specified field type in code that stores tensor
values:

```@example practical-tips
struct MaterialState{T}
    σ::Tensor{Tuple{@Symmetry{3,3}}, T, 2, 6} # same as SymmetricSecondOrderTensor{3,T,6}
end
nothing # hide
```

For fixed-size aliases, the same rule applies. `SymmetricSecondOrderTensor{3,T}`
is convenient in method signatures, but a concrete field type also needs the
component-count parameter `L`, as in `SymmetricSecondOrderTensor{3,T,6}`.
For the Julia background, see
[Avoid fields with abstract type](https://docs.julialang.org/en/v1/manual/performance-tips/#Avoid-fields-with-abstract-type)
in the Julia manual.

## Put known symmetry in the type

If a value is symmetric, represent that in the tensor type. The tensor still
indexes and displays like the full tensor, but only the independent components
are stored:

```@repl practical-tips
A = @Mat [1.0 2.0 3.0; 2.0 4.0 5.0; 3.0 5.0 6.0]
S = SymmetricSecondOrderTensor{3}(A)
Tuple(S)
```

This also tells later operations which tensor space to use:

```@repl practical-tips
gradient(identity, S) isa SymmetricFourthOrderTensor{3}
```

Use [`symmetric`](@ref) when you want to compute the symmetric part of a general
second-order tensor. For the full notation, see [Tensor Types and Spaces](@ref).

## Specify `@einsum` result types when needed

[`@einsum`](@ref) infers free indices, but it cannot prove every symmetry of the
result. If you know the result belongs to a symmetric tensor space, give that
space explicitly:

```@repl practical-tips
A = rand(Mat{3,3});
S1 = @einsum A[k,i] * A[k,j]
S1 isa SymmetricSecondOrderTensor{3}
S2 = @einsum SymmetricSecondOrderTensor{3} A[k,i] * A[k,j]
S2 isa SymmetricSecondOrderTensor{3}
S1 ≈ S2
```

Tensorial uses the annotated type as the result space. Use it only when the
formula really has that symmetry.

## Flatten at boundaries

Tensorial code usually reads best when tensor values stay as tensors. Use
[`flatview`](@ref), [`tovoigt`](@ref), [`tomandel`](@ref), and the corresponding
inverse conversions at boundaries, such as solver interfaces, file formats, or
code that specifically expects vectors and matrices:

```@repl practical-tips
σ = SymmetricSecondOrderTensor{3}((2.0, 0.4, 0.2, 1.2, 0.1, 0.9))
v = tovoigt(σ)
fromvoigt(SymmetricSecondOrderTensor{3}, v) ≈ σ
```

For symmetric tensor blocks inside direct sums, [`flatview`](@ref) uses Mandel
scaling. See [Voigt Form](@ref) for the conversion rules.

## Type inference when unpacking

[`unpack(x)`](@ref unpack) returns all blocks of a direct-sum value as a tuple.
The return type is concrete because the full block layout is known. The
`@code_warntype` excerpts below omit the long inferred-code body and keep the
parts relevant to type inference. In each excerpt, focus on the `Body::...`
line: it shows the return type Julia inferred for the call.

For `unpack(x)`, Julia knows the complete block layout and infers the concrete
tuple type:

```julia-repl
julia> x = pack(σ, 0.2)
2-element DirectSumVector with storage Float64:
 Space(Symmetry(3, 3),)
 Space()

julia> @code_warntype unpack(x)
MethodInstance for unpack(::DirectSumVector{...})
Arguments
  #self#::Core.Const(Tensorial.unpack)
  A::DirectSumVector{...}
Body::Tuple{SymmetricSecondOrderTensor{3, Float64, 6}, Float64}
```

For indexed access, the method receives the block index as an `Int`. Since the
blocks can have different types, Julia infers a small `Union`:

```julia-repl
julia> @code_warntype unpack(x, 1)
MethodInstance for unpack(::DirectSumVector{...}, ::Int64)
Arguments
  #self#::Core.Const(Tensorial.unpack)
  A::DirectSumVector{...}
  i::Int64
Body::Union{Float64, SymmetricSecondOrderTensor{3, Float64, 6}}
```

If the selected block is written as a constant inside a function, Julia can
propagate that constant and infer the concrete block type:

```julia-repl
julia> first_block(x) = unpack(x, 1);

julia> @code_warntype first_block(x)
MethodInstance for first_block(::DirectSumVector{...})
Arguments
  #self#::Core.Const(first_block)
  x::DirectSumVector{...}
Body::SymmetricSecondOrderTensor{3, Float64, 6}
```

Use indexed `unpack` for inspection or when the selected block is known to the
compiler. Otherwise, prefer `unpack(x)` and destructure the returned tuple. For
the full output, see the
[`@code_warntype`](https://docs.julialang.org/en/v1/stdlib/InteractiveUtils/#InteractiveUtils.@code_warntype)
documentation.
