```@meta
CurrentModule = Tensorial
DocTestSetup = :(using Tensorial; using LinearAlgebra)
```

# Automatic Differentiation

Tensorial differentiates scalar, vector, tensor, and packed-state functions in
the tensor spaces of their arguments. For most code, start with
[`gradient`](@ref) and [`hessian`](@ref). The callable operator [`∂`](@ref) is
the general form for higher derivatives.

```@setup automatic-differentiation
using Tensorial
using LinearAlgebra
```

## Gradients and hessians

For a scalar-valued function of a vector input, `gradient` returns a vector and
`hessian` returns a second-order tensor:

```@repl automatic-differentiation
a = @Vec [3.0, 4.0]
φ(a) = 0.5 * (a ⊡ a)
gradient(φ, a)
hessian(φ, a)
```

When an argument is a symmetric tensor, AD is performed on that symmetric tensor
space:

```@repl automatic-differentiation
ε = symmetric(@Mat [0.02 0.01 0.0; 0.01 0.00 0.0; 0.0 0.0 -0.01])
K = 10.0; # bulk modulus
G = 5.0;  # shear modulus
ψ(ε) = K/2 * tr(ε)^2 + G * (dev(ε) ⊡₂ dev(ε))
σ = gradient(ψ, ε);
σ isa SymmetricSecondOrderTensor{3}
ℂ = hessian(ψ, ε);
ℂ isa SymmetricFourthOrderTensor{3}
```

The tensor space of the argument matters. If a value is symmetric only
numerically, but its type is a general matrix tensor, the derivative is taken in
the general matrix space:

```@repl automatic-differentiation
A = rand(Mat{3,3})
S = A * A'; # symmetric in value, but not typed as a symmetric tensor
gradient(identity, S) ≈ one(FourthOrderTensor{3})
gradient(identity, symmetric(S)) ≈ one(SymmetricFourthOrderTensor{3})
```

For more on symmetric tensor spaces, see [Tensor Types and Spaces](@ref).

## Returning the function value

Pass `:all` when you want derivatives and the function value from one call.
For `gradient`, the return value is

```julia
(gradient(f, x), f(x))
```

and for `hessian`, it is

```julia
(hessian(f, x), gradient(f, x), f(x))
```

```@repl automatic-differentiation
gradient(φ, a, :all)
hessian(φ, a, :all)
```

For the general operator `∂{N}`, `:all` returns derivatives from higher to lower
order, ending with the function value:

```julia
(∂{N}(f, x), ..., ∂{2}(f, x), ∂(f, x), f(x))
```

## The general operator `∂`

For scalar input and scalar output, `∂` is the most direct spelling.
`∂(f, args...)` is equivalent to `∂{1}(f, args...)`. Use braces for higher
orders:

```@repl automatic-differentiation
f(x) = x^4 + x
x = 2.0
f(x)
∂(f, x)
∂{2}(f, x)
∂{2}(f, x, :all)
∂{3}(x -> x^5, x)
```

The general operator is useful when the derivative order is part of the formula
you want to write down. `gradient(f, x)` is `∂{1}(f, x)`, and
`hessian(f, x)` is `∂{2}(f, x)`.

## Multiple inputs

For multiple inputs, the first derivative is returned as a tuple whose entries
follow the order of the inputs:

```@repl automatic-differentiation
gradient((x, y) -> x^2 + 3x*y + y^2, 2.0, 4.0)
gradient((x, y) -> x^2 + 3x*y + y^2, 2.0, 4.0, :all)
```

The result represents:

```julia
(∂f/∂x, ∂f/∂y)
```

With `:all`, it returns:

```julia
((∂f/∂x, ∂f/∂y), f(x, y))
```

Second derivatives for multiple inputs are returned as a block Hessian:

```@repl automatic-differentiation
hessian((x, y) -> x^2 + x*y + y^3, 2.0, 3.0)
hessian((x, y) -> x^2 + x*y + y^3, 2.0, 3.0, :all)
```

The Hessian block structure is

```julia
(
    (∂²f/∂x², ∂²f/∂x∂y),
    (∂²f/∂y∂x, ∂²f/∂y²),
)
```

The same block structure is used when the input types differ:

```@repl automatic-differentiation
A = symmetric(@Mat [1.0 0.2; 0.2 2.0])
d = gradient((x, A) -> x * tr(A), x, A)
d[1]
d[2] isa SymmetricSecondOrderTensor{2}
H = hessian((x, A) -> x * tr(A), x, A);
H[1][2] isa SymmetricSecondOrderTensor{2}
H[2][2] isa SymmetricFourthOrderTensor{2}
```

## Multiple outputs

If `f` returns a tuple, each output is differentiated separately. The outer
tuple follows the outputs:

```@repl automatic-differentiation
gradient(x -> (x^2, x^3), 2.0)
gradient(x -> (x^2, x^3), 2.0, :all)
```

The result represents:

```julia
(∂f₁/∂x, ∂f₂/∂x)
```

With `:all`, it returns:

```julia
((∂f₁/∂x, ∂f₂/∂x), (f₁(x), f₂(x)))
```

Second derivatives are handled in the same way:

```@repl automatic-differentiation
hessian(x -> (x^2, x^3), 2.0)
hessian(x -> (x^2, x^3), 2.0, :all)
```

The result represents:

```julia
(∂²f₁/∂x², ∂²f₂/∂x²)
```

and `:all` returns

```julia
(
    (∂²f₁/∂x², ∂²f₂/∂x²),
    (∂f₁/∂x, ∂f₂/∂x),
    (f₁(x), f₂(x)),
)
```

## Multiple inputs and multiple outputs

When there are both multiple inputs and multiple outputs, the outer tuple
follows the outputs, and the inner tuple follows the inputs:

```@repl automatic-differentiation
gradient((x, y) -> (x + y, x * y), 2.0, 3.0)
gradient((x, y) -> (x + y, x * y), 2.0, 3.0, :all)
```

The result represents:

```julia
(
    (∂f₁/∂x, ∂f₁/∂y),
    (∂f₂/∂x, ∂f₂/∂y),
)
```

For second derivatives, each output carries its own block Hessian:

```@repl automatic-differentiation
hessian((x, y) -> (x + y, x * y), 2.0, 3.0)
hessian((x, y) -> (x + y, x * y), 2.0, 3.0, :all)
```

The result represents:

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

With `:all`, the return value is

```julia
(
    second_derivatives,
    first_derivatives,
    function_value,
)
```

where `second_derivatives` has the block-Hessian structure shown above and
`first_derivatives` has the output/input structure of `gradient`.

For automatic-differentiation docstrings, see
[Automatic differentiation API](@ref).
