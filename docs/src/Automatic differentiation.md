# Automatic differentiation

```@setup automatic-differentiation
using Tensorial
```

Automatic differentiation is provided by the callable operator [`∂`](@ref).

For a function `f` and arguments `args...`, `∂{N}(f, args...)` computes the
`N`th-order partial derivative of `f` with respect to `args...`.
`∂(f, args...)` is equivalent to `∂{1}(f, args...)`.

The basic usage is:

- `∂{N}(f, args...)` returns only the highest-order derivative.
- `∂{N}(f, args..., :all)` returns all derivatives up to order `N`,
  together with the function value.

When pseudo keyword `:all` is given, the return value is ordered from higher to lower order,
ending with the function value:

```julia
(∂{N}(f, args...), ..., ∂{2}(f, args...), ∂(f, args...), f(args...))
```

For multiple inputs, mixed partial derivatives are grouped by input blocks.
If `f` returns a tuple, each component is differentiated separately.

!!! warning
    The user must provide the appropriate tensor symmetry information;
    otherwise, automatic differentiation may not preserve the expected tensor symmetry.
    In the following example, even with identical tensor values,
    the results differ depending on the `Tensor` type.

    ```@repl automatic-differentiation
    A = rand(Mat{3,3})
    S = A * A' # symmetric in value, but not typed as `SymmetricSecondOrderTensor`
    ∂(identity, S) ≈ one(FourthOrderTensor{3})
    ∂(identity, symmetric(S)) ≈ one(SymmetricFourthOrderTensor{3})
    ```

```@docs
∂
```

## Basic examples

We begin with the simplest case: a scalar-valued function of a single scalar variable.

```@repl automatic-differentiation
x = 2.0
∂(x -> x^3, x)
∂(x -> x^3, x, :all)
```

Here, `∂(x -> x^3, x)` returns only the first derivative, while
`∂(x -> x^3, x, :all)` returns

```julia
(∂f/∂x, f(x))
```

Higher-order derivatives are obtained by specifying the order in braces.

```@repl automatic-differentiation
∂{2}(x -> x^3, x)
∂{2}(x -> x^3, x, :all)
```

In this case,

```julia
∂{2}(x -> x^3, x, :all)
```

returns

```julia
(∂²f/∂x², ∂f/∂x, f(x))
```

The same interface also works for tensor inputs.

```@repl automatic-differentiation
a = rand(Vec{2})
∂(norm, a)
∂(norm, a, :all)
∂{2}(norm, a)
∂{2}(norm, a, :all)
```

## Multiple inputs

For multiple inputs, the first-order derivative is returned as a tuple whose
entries follow the order of the inputs.

```@repl automatic-differentiation
∂((x, y) -> x^2 + 3x*y + y^2, 2.0, 4.0)
∂((x, y) -> x^2 + 3x*y + y^2, 2.0, 4.0, :all)
```

The first result is interpreted as

```julia
(∂f/∂x, ∂f/∂y)
```

and the second as

```julia
((∂f/∂x, ∂f/∂y), f(x, y))
```

Second-order derivatives for multiple inputs are returned as a block Hessian.

```@repl automatic-differentiation
∂{2}((x, y) -> x^2 + x*y + y^3, 2.0, 3.0)
∂{2}((x, y) -> x^2 + x*y + y^3, 2.0, 3.0, :all)
```

The first result is interpreted as

```julia
(
    (∂²f/∂x², ∂²f/∂x∂y),
    (∂²f/∂y∂x, ∂²f/∂y²),
)
```

and the second as

```julia
(
    (
        (∂²f/∂x², ∂²f/∂x∂y),
        (∂²f/∂y∂x, ∂²f/∂y²),
    ),
    (∂f/∂x, ∂f/∂y),
    f(x, y),
)
```

The same block structure is used even when the input types differ.

```@repl automatic-differentiation
x = 2.0
A = rand(SymmetricSecondOrderTensor{2})

∂((x, A) -> x * tr(A), x, A)
∂((x, A) -> x * tr(A), x, A, :all)

∂{2}((x, A) -> x * tr(A), x, A)
∂{2}((x, A) -> x * tr(A), x, A, :all)
```

## Multiple outputs

If `f` returns a tuple, each component is differentiated separately.
The outer tuple follows the outputs.

```@repl automatic-differentiation
∂(x -> (x^2, x^3), 2.0)
∂(x -> (x^2, x^3), 2.0, :all)
```

The first result is interpreted as

```julia
(∂f₁/∂x, ∂f₂/∂x)
```

and the second as

```julia
((∂f₁/∂x, ∂f₂/∂x), (f₁(x), f₂(x)))
```

Second-order derivatives are handled in the same way.

```@repl automatic-differentiation
∂{2}(x -> (x^2, x^3), 2.0)
∂{2}(x -> (x^2, x^3), 2.0, :all)
```

The first result is interpreted as

```julia
(∂²f₁/∂x², ∂²f₂/∂x²)
```

and the second as

```julia
(
    (∂²f₁/∂x², ∂²f₂/∂x²),
    (∂f₁/∂x, ∂f₂/∂x),
    (f₁(x), f₂(x)),
)
```

## Multiple inputs and multiple outputs

When there are both multiple inputs and multiple outputs, the outer tuple
follows the outputs, and the inner tuple follows the inputs.

```@repl automatic-differentiation
∂((x, y) -> (x + y, x * y), 2.0, 3.0)
∂((x, y) -> (x + y, x * y), 2.0, 3.0, :all)
```

The first result is interpreted as

```julia
(
    (∂f₁/∂x, ∂f₁/∂y),
    (∂f₂/∂x, ∂f₂/∂y),
)
```

and the second as

```julia
(
    (
        (∂f₁/∂x, ∂f₁/∂y),
        (∂f₂/∂x, ∂f₂/∂y),
    ),
    (f₁(x, y), f₂(x, y)),
)
```

For second-order derivatives, each output carries its own block Hessian.

```@repl automatic-differentiation
∂{2}((x, y) -> (x + y, x * y), 2.0, 3.0)
∂{2}((x, y) -> (x + y, x * y), 2.0, 3.0, :all)
```

The first result is interpreted as

```julia
(
    (
        (∂²f₁/∂x², ∂²f₁/∂x∂y),
        (∂²f₁/∂y∂x, ∂²f₁/∂y²),
    ),
    (
        (∂²f₂/∂x², ∂²f₂/∂x∂y),
        (∂²f₂/∂y∂x, ∂²f₂/∂y²),
    ),
)
```

and the second as

```julia
(
    (
        (
            (∂²f₁/∂x², ∂²f₁/∂x∂y),
            (∂²f₁/∂y∂x, ∂²f₁/∂y²),
        ),
        (
            (∂²f₂/∂x², ∂²f₂/∂x∂y),
            (∂²f₂/∂y∂x, ∂²f₂/∂y²),
        ),
    ),
    (
        (∂f₁/∂x, ∂f₁/∂y),
        (∂f₂/∂x, ∂f₂/∂y),
    ),
    (f₁(x, y), f₂(x, y)),
)
```

## Aliases

`gradient` and `hessian` are aliases for first- and second-order partial
derivatives:

```@docs
gradient
hessian
```
