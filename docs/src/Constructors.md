```@meta
CurrentModule = Tensorial
DocTestSetup = :(using Tensorial)
```

# Constructing Tensors

Tensorial values can be constructed from literal values, ordinary Julia arrays,
or index formulas. For most code, start with the literal macros. Use typed
constructors when the tensor shape is part of the API you want to express.

```@setup constructors
using Tensorial
```

## Literal macros

`@Vec`, `@Mat`, and `@Tensor` build Tensorial values from Julia literals or
array expressions. They are analogous to StaticArrays.jl's `@SVector`,
`@SMatrix`, and `@SArray`, but the result is a Tensorial tensor type.

```@repl constructors
using Tensorial
x = @Vec [1.0, 2.0, 3.0]
A = @Mat [1.0 2.0; 3.0 4.0; 5.0 6.0]
C = @Tensor rand(2, 2, 2);
typeof(C)
```

## From arrays

Use explicit constructors when the tensor size should be visible in the type:

```@repl constructors
Vec{3}([1.0, 2.0, 3.0])
Mat{3,2}([1.0 2.0; 3.0 4.0; 5.0 6.0])
Mat{3,2,Float64}([1 2; 3 4; 5 6])
```

The element type can be supplied explicitly, as in `Mat{3,2,Float64}`. If it is
omitted, Tensorial uses the element type of the input.

## Construct symmetric tensors

Use a symmetric tensor constructor when the input data is already symmetric:

```@repl constructors
SymmetricSecondOrderTensor{2}([1.0 2.0; 2.0 4.0])
```

If the input is not symmetric and you want its symmetric part, use
[`symmetric`](@ref):

```@repl constructors
B = @Mat [1.0 3.0; 2.0 4.0]
symmetric(B)
```

For the general `@Symmetry` size notation, see [Tensor Types and Spaces](@ref).

## From functions

For tensors defined by an index formula, pass a function whose arguments are the
tensor indices:

```@repl constructors
δ = one(Mat{2,2})
I = SymmetricFourthOrderTensor{2}((i, j, k, l) ->
    (δ[i,k] * δ[j,l] + δ[i,l] * δ[j,k]) / 2)
I == one(SymmetricFourthOrderTensor{2})
```

This form is useful for identity tensors, projectors, and other tensors that are
clearer as indexed formulas than as stored component tuples.

## Special tensors

Use `zero` and `one` for typed zero and identity tensors:

```@repl constructors
zero(Vec{3})
one(Mat{2,2})
one(SymmetricFourthOrderTensor{2})
```

The Levi-Civita tensor is available with [`levicivita`](@ref):

```@repl constructors
E = levicivita();
E[1,2,3]
E[1,3,2]
```

For constructor docstrings, see [Constructors and special tensors](@ref).
