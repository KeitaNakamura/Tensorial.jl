```@meta
CurrentModule = Tensorial
```

# Getting Started

Tensorial.jl lets you write small tensor calculations directly in Julia. This
page starts with vectors and matrices, then moves on to symmetry, automatic
differentiation, and block-structured states.

On this page:

- [Small tensors](@ref "Create small tensors")
- [Tensor operations](@ref "Write tensor operations")
- [`@einsum` index notation](@ref "Write indexed formulas with `@einsum`")
- [Symmetric tensors](@ref "Symmetric tensors")
- [Automatic differentiation](@ref "Differentiate tensor formulas")
- [Direct sum](@ref "Direct sum (pack and unpack)")
- [Advanced example](@ref "Advanced example: block-structured return mapping")

## Create small tensors

`Vec` and `Mat` are aliases for Tensorial vector and matrix types. If you know
StaticArrays.jl, `@Vec` and `@Mat` are analogous to `@SVector` and `@SMatrix`,
but they create Tensorial tensor types. Use `@Tensor` for higher-order tensors.

```@setup quick-start
using Tensorial
```

```@repl quick-start
using Tensorial
x = @Vec [1.0, 2.0, 3.0]
A = @Mat [1.0 2.0 0.0; 0.0 3.0 4.0; 5.0 0.0 6.0]
A isa AbstractArray
C = @Tensor rand(3, 3, 3);
typeof(C)
```

These tensors work with ordinary Julia array code and support tensor-specific
operations:

## Write tensor operations

```@repl quick-start
A ⊡ x
A ⊡ x ≈ A * x
A ⊡₂ A ≈ A ⋅ A
x ⊗ x
```

The comparisons show the familiar array equivalents. The operators themselves
express tensor notation directly:

- `⊡` contracts one index. In this example, it is the matrix-vector product.
- `⊡₂` contracts two indices. For matrices, it is the Frobenius inner product.
- `⊗` forms a tensor product.

!!! info "Typing Unicode operators"
    In the Julia REPL or editors with Julia tab completion, type the LaTeX-like
    name and press `<TAB>`. ASCII aliases are also available if you prefer plain
    names.

    | Operator | Type | Alias |
    |:---------|:-----|:------|
    | `⊡` | `\boxdot<TAB>` | `contract1` |
    | `⊡₂` | `\boxdot<TAB>` then `\_2<TAB>` | `contract2` |
    | `⊗` | `\otimes<TAB>` | `tensor` |

## Write indexed formulas with `@einsum`

Use [`@einsum`](@ref) when an expression is clearer in index notation. Repeated
indices are summed, and indices that appear once become the free indices of the
result:

```@repl quick-start
(@einsum A[i,k] * A[k,j]) ≈ A * A
(@einsum A[i,j] * A[i,j]) ≈ A ⋅ A
```

In the first expression, `k` appears twice, so it is contracted. The indices `i`
and `j` appear once, so they are the free indices of the result. When free
indices are not written explicitly, `@einsum` infers them in the order they
appear from left to right.

You can also give the free indices explicitly. This is useful when you want a
particular output order:

```@repl quick-start
(@einsum (j,i) -> A[i,k] * A[k,j]) ≈ transpose(A * A)
```

For a named result, put the free indices on the left-hand side:

```@repl quick-start
@einsum B[i,j] := A[i,k] * A[k,j];
B ≈ A * A
```

## Symmetric tensors

Use `symmetric(...)` to compute the symmetric part of a second-order tensor and
return a symmetric tensor type:

```@repl quick-start
ε = symmetric(@Mat [0.02 0.01 0.0; 0.01 0.00 0.0; 0.0 0.0 -0.01])
ε isa SymmetricSecondOrderTensor{3}
Tuple(ε)
```

The displayed matrix is `3×3`, but the stored tuple contains only the six
independent components. The symmetry is part of the tensor type, not just a
property of the displayed values.

For the general `@Symmetry` notation and tensor-space aliases, see
[Tensor Types and Spaces](@ref).

## Differentiate tensor formulas

Automatic differentiation is performed in the tensor space of the input. For a
scalar strain-energy function, the gradient is a stress tensor and the Hessian
is a tangent stiffness tensor:

```@repl quick-start
K = 10.0; # bulk modulus
G = 5.0;  # shear modulus
ψ(ε) = K/2 * tr(ε)^2 + G * (dev(ε) ⊡₂ dev(ε))
σ = gradient(ψ, ε);
σ isa SymmetricSecondOrderTensor{3}
ℂ = hessian(ψ, ε);
ℂ isa SymmetricFourthOrderTensor{3}
```

The derivatives stay in the corresponding tensor spaces. No Voigt conversion is
needed in this example.

!!! info "gradient(..., :all)"
    Passing `:all` returns both the derivative and the function value. For
    example, `g, y = gradient(f, x, :all)` gives `g = gradient(f, x)` and
    `y = f(x)`.

## Direct sum (pack and unpack)

A direct sum lets you treat several blocks as one state while keeping the block
structure explicit. [`pack`](@ref) builds the state, and [`unpack`](@ref)
retrieves its blocks:

```@repl quick-start
state = pack(σ, 0.0)
unpack(state, 1)
unpack(state, 2)
```

!!! info "Infix form"
    `⊕` is an infix spelling of `pack`: `a ⊕ b` is equivalent to `pack(a, b)`.
    Type it as `\oplus<TAB>`.

## Advanced example: block-structured return mapping

As a final, slightly more advanced example, combine the tools above in the
active plastic branch of a small isotropic J2 return-mapping update. The state
contains an updated stress tensor `σ` and a plastic multiplier increment `Δγ`.

The residual keeps the radial-return form explicit. Tensorial differentiates a
residual with tensor and scalar blocks directly, so the Newton correction can be
written in block form.

```@example quick-start
σᵗʳ = SymmetricSecondOrderTensor{3}((2.0, 0.4, 0.2, 1.2, 0.1, 0.9))
G = 5.0   # shear modulus
σy0 = 0.6 # initial yield stress
H = 2.0   # isotropic hardening modulus

q(σ) = sqrt(3/2) * norm(dev(σ)) # von Mises stress
yield_function(σ, Δγ) = q(σ) - (σy0 + H * Δγ)

function residual(x)
    σ, Δγ = unpack(x)
    # flow direction and yield-function value
    n, f = gradient(σ -> yield_function(σ, Δγ), σ, :all)
    R_σ = σ - σᵗʳ + 2G * Δγ * n
    R_γ = f
    pack(R_σ, R_γ)
end
nothing # hide
```

With the residual defined, one Newton update gives a new packed state. Unpack it
immediately to recover the updated variables:

```@example quick-start
x = pack(σᵗʳ, 0.0)
J = gradient(residual, x)
δx = -J \ residual(x)
xnew = x + δx
σ, Δγ = unpack(xnew)
nothing # hide
```

Here `x` is the current Newton state, `J` is the Jacobian of the packed residual,
`δx` is the Newton correction, and `xnew` is the updated state.

The updated stress and plastic multiplier are:

```@example quick-start
σ
```

```@example quick-start
Δγ
```

Packing those updated blocks again gives a direct-sum state whose residual is
zero for this return-mapping update:

```@example quick-start
norm(residual(σ ⊕ Δγ))
```

For the full direct-sum explanation, including the Jacobian blocks, see
[Direct Sum](@ref).

## Where to go next

- [Tensor Types and Spaces](@ref): `Tensor{S, T, N, L}`, aliases, and symmetry
  in tensor types.
- [Constructing Tensors](@ref): constructors, macros, and explicit
  `Tensor{S, T, N, L}` types.
- [Operations](@ref): contractions, tensor products, `@einsum`, and related
  operations.
- [Automatic Differentiation](@ref): gradients, Hessians, and derivatives with
  multiple inputs or outputs.
- [Direct Sum](@ref): packed states, block residuals, and Newton updates.
- [Practical Tips](@ref): type annotations, `@einsum` result types, and
  representation boundaries.
- [API Reference](@ref): signatures and detailed behavior.
- [Benchmarks](@ref "Benchmarks"): small-tensor performance.
