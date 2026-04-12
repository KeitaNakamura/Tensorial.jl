# Direct sum

```@setup direct-sum
using Tensorial
using LinearAlgebra
```

A direct sum is useful when a single unknown consists of several blocks with
different meanings, for example

- a tensor and a scalar,
- a symmetric tensor and a vector,
- several tensors with different symmetries.

In such cases, it is often convenient to treat the whole collection as a single object while still keeping the block structure explicit. In Tensorial.jl, this is represented by [`DirectSumArray`](@ref). A direct-sum value is constructed with [`pack`](@ref), or equivalently with `⊕`.

## Basic usage

Direct sums are particularly convenient when a problem naturally has several coupled unknowns of different types.

The basic constructors are [`pack`](@ref) and its alias `⊕`.

```@repl direct-sum
A = @Mat[1.0 2.0; 3.0 4.0]
x = pack(A, 3.0)
y = A ⊕ 3.0
x == y
```

The stored blocks are recovered with [`unpack`](@ref):

```@repl direct-sum
unpack(x)
unpack(x, 1)
unpack(x, 2)
```

For symmetric tensor blocks, the internal storage uses Mandel coordinates. This is visible through [`flatview`](@ref):

```@repl direct-sum
As = symmetric(A)
z = As ⊕ 3.0
flatview(z)
```

## Direct sums and automatic differentiation

One of the main advantages of direct sums is that several coupled variables can be treated as one state while preserving the block structure in derivatives.

As a first example, consider a function of a symmetric tensor block `A` and a scalar block `s`:

```@repl direct-sum
x = symmetric(@Mat[1.0 2.0; 3.0 4.0]) ⊕ 2.0

function f(z)
    A, s = unpack(z)
    dot(A, A) + s * tr(A) + s^2
end

f(x)
```

The gradient is again a direct-sum object with the same blocks:

```@repl direct-sum
g = gradient(f, x)
unpack(g)
```

The Hessian is returned as a block matrix:

```@repl direct-sum
H = hessian(f, x)
unpack(H, 1, 1)
unpack(H, 1, 2)
unpack(H, 2, 1)
unpack(H, 2, 2)
```

Thus the derivative already reflects the natural block structure of the problem:

- `unpack(H, 1, 1)` is the tensor--tensor block,
- `unpack(H, 1, 2)` and `unpack(H, 2, 1)` are the mixed couplings,
- `unpack(H, 2, 2)` is the scalar--scalar block.

The same idea applies to vector-valued maps.

```@repl direct-sum
x = symmetric(@Mat[1.0 2.0; 3.0 4.0]) ⊕ 2.0
C = symmetric(@Mat[2.0 1.0; 1.0 3.0])

function F(z)
    A, s = unpack(z)
    B = A + s * C
    t = tr(A) + s^2
    B ⊕ t
end

y = F(x)
unpack(y)
```

Its Jacobian is again block-structured:

```@repl direct-sum
J = gradient(F, x)
unpack(J, 1, 1)
unpack(J, 1, 2)
unpack(J, 2, 1)
unpack(J, 2, 2)
```

This is precisely the structure needed in many coupled problems.

## Linear algebra in direct-sum form

A `DirectSumArray` also has a flat coordinate representation, which is useful for low-level inspection and for linear algebra operations.

```@repl direct-sum
flatview(x)
flatview(J)
```

For symmetric tensor blocks, the flat coordinates use Mandel scaling. This makes the Euclidean inner product of the flat representation consistent with the natural tensor inner product.

In particular, linear solves can be written directly in block form. If `J` is a `DirectSumMatrix` and `r` is a `DirectSumVector`, one may solve

```julia
δx = -J \ r
```

without manually converting the system to flat coordinates.

## Example: return mapping solved by Newton's method

A natural use of direct sums is a local Newton solve in return mapping, where the unknown consists of multiple blocks with different meanings.

As a simple model, consider a small-strain von Mises update written as a nonlinear system for

- the updated symmetric stress `σ`, and
- the plastic multiplier increment `Δγ`.

We collect them into one direct-sum variable,

```@repl direct-sum
σ₀ = SymmetricSecondOrderTensor{3}((1.0, 0.2, 0.1, 0.8, 0.05, 0.5))
x = σ₀ ⊕ 0.0
unpack(x)
```

Suppose we want to solve the local system

```math
R(\bm{\sigma}, \Delta\gamma) =
\begin{Bmatrix}
\bm{\sigma} - \bm{\sigma}^{\mathrm{tr}} + \Delta\gamma \bm{n} \\
\|\operatorname{dev}(\bm{\sigma})\| - (\sigma_y + H\,\Delta\gamma)
\end{Bmatrix}
= \bm{0},
```

where `σᵗʳ` is the trial stress, `σy` is the yield stress, and `H` is the
hardening modulus. The flow direction is given by the associative flow rule,

```math
n = \frac{\partial f}{\partial \bm{\sigma}},
```

with `f` the yield function.

The residual can be written directly in terms of a packed state:

```@repl direct-sum
σᵗʳ = SymmetricSecondOrderTensor{3}((2.0, 0.4, 0.2, 1.2, 0.1, 0.9))
σy = 0.6
H = 2.0

yield_function(σ, Δγ) = norm(dev(σ)) - (σy + H * Δγ)

function R(x)
    σ, Δγ = unpack(x)
    f = yield_function(σ, Δγ)
    n = gradient(σ -> yield_function(σ, Δγ), σ)
    R_σ = σ - σᵗʳ + Δγ * n
    R_γ = f
    R_σ ⊕ R_γ
end

x = σᵗʳ ⊕ 0.0
unpack(R(x))
```

Its Jacobian is obtained automatically:

```@repl direct-sum
J = gradient(R, x)
n = gradient(σ -> yield_function(σ, 0.0), σᵗʳ)

unpack(J, 1, 1) ≈ one(SymmetricFourthOrderTensor{3})
unpack(J, 1, 2) ≈ n
unpack(J, 2, 1) ≈ n
unpack(J, 2, 2) == -H
```

A Newton step can then be written directly in direct-sum form:

```@repl direct-sum
r = R(x)
δx = -J \ r
xnew = x + δx
norm(R(xnew))
```

For the present von Mises example, the Newton update reaches the solution in a
single step. This reflects the classical radial-return structure of the problem.

For more general return-mapping problems, however, the local residual is
genuinely nonlinear, and several Newton iterations are typically required.

## APIs

```@index
Pages = ["Direct sum.md"]
```

```@docs
DirectSumArray
pack
unpack
flatview
```
