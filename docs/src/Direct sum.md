```@meta
CurrentModule = Tensorial
DocTestSetup = :(using Tensorial; using LinearAlgebra)
```

# Direct Sum

A direct sum collects several blocks into one state. This is useful when the
unknowns have different meanings or different tensor spaces, such as a
symmetric stress tensor and a scalar internal variable.

In Tensorial, direct-sum values are built with [`pack`](@ref). The infix
operator `⊕` is equivalent to `pack`.

```@setup direct-sum
using Tensorial
using LinearAlgebra
```

## Pack and unpack blocks

```@repl direct-sum
A = @Mat [1.0 2.0; 3.0 4.0]
x = pack(A, 3.0)
unpack(x)
unpack(x, 1)
unpack(x, 2)
```

The infix form is convenient for short expressions:

```@repl direct-sum
y = A ⊕ 3.0
x == y
```

!!! info "Typing ⊕"
    In the Julia REPL or editors with Julia tab completion, type
    `\oplus<TAB>`.

For symmetric tensor blocks, the flat coordinate representation uses Mandel
scaling. This is visible with [`flatview`](@ref):

```@repl direct-sum
As = symmetric(A)
z = pack(As, 3.0)
flatview(z)
```

## Differentiating packed states

Several coupled variables can be treated as one state while derivatives keep the
block layout. Here the state contains a symmetric tensor block `A` and a scalar
block `s`:

```@repl direct-sum
x = pack(symmetric(@Mat [1.0 2.0; 2.0 4.0]), 2.0)

function f(z)
    A, s = unpack(z)
    dot(A, A) + s * tr(A) + s^2
end

f(x)
```

The gradient has the same block structure as `x`:

```@repl direct-sum
g = gradient(f, x)
unpack(g)
```

The Hessian is a direct-sum matrix. The tensor--tensor block is fourth-order,
so here we check its type and print the smaller coupling blocks:

```@repl direct-sum
H = hessian(f, x)
unpack(H, 1, 1) isa SymmetricFourthOrderTensor{2}
unpack(H, 1, 2)
unpack(H, 2, 1)
unpack(H, 2, 2)
```

## Residuals and Jacobian blocks

Packed states also work for residual maps, where the output is packed as well:

```@repl direct-sum
C = symmetric(@Mat [2.0 1.0; 1.0 3.0])

function F(z)
    A, s = unpack(z)
    B = A + s * C
    t = tr(A) + s^2
    pack(B, t)
end

r = F(x)
unpack(r)
```

With

```math
\bm{z} =
\begin{bmatrix}
\bm{A} \\
s
\end{bmatrix},
\qquad
\bm{F}(\bm{z}) =
\begin{bmatrix}
\bm{B}(\bm{A},s) \\
t(\bm{A},s)
\end{bmatrix},
```

the Jacobian has the corresponding block structure:

```math
\bm{J} =
\frac{\partial \bm{F}}{\partial \bm{z}}
=
\begin{bmatrix}
\dfrac{\partial \bm{B}}{\partial \bm{A}} & \dfrac{\partial \bm{B}}{\partial s} \\
\dfrac{\partial t}{\partial \bm{A}} & \dfrac{\partial t}{\partial s}
\end{bmatrix}.
```

The calls below inspect the same four Jacobian blocks. Again, the first block is
a fourth-order tensor, while the remaining blocks are small enough to print:

```@repl direct-sum
J = gradient(F, x)
unpack(J, 1, 1) isa SymmetricFourthOrderTensor{2}
unpack(J, 1, 2)
unpack(J, 2, 1)
unpack(J, 2, 2)
```

This is the structure needed by Newton updates and other coupled local solves.
If `J` is a `DirectSumMatrix` and `r` is a `DirectSumVector`, the correction can
be written directly:

```@repl direct-sum
δx = -J \ r
unpack(δx)
```

No manual flattening is needed.

!!! note "Mandel form"
    When you do inspect the flat coordinates, symmetric blocks are shown in
    Mandel form: off-diagonal components are scaled so the Euclidean inner
    product of the flat coordinates matches the tensor inner product. See
    Wikipedia's
    [Mandel notation](https://en.wikipedia.org/wiki/Voigt_notation#Mandel_notation)
    summary for the convention.

```@repl direct-sum
flatview(J)
```

## Example: return mapping residual

As a larger example, consider the active plastic branch of a small-strain
isotropic J2 return-mapping update. The unknown state contains

- the updated symmetric stress `σ`, and
- the plastic multiplier increment `Δγ`.

We solve the local residual

```math
\bm{R}(\bm{\sigma}, \Delta\gamma) =
\begin{Bmatrix}
\bm{\sigma} - \bm{\sigma}^{\mathrm{tr}}
    + \Delta\gamma\,\bm{\mathbb{C}}^{\mathrm{e}} : \bm{n} \\
q(\bm{\sigma}) - (\sigma_{y0} + H\,\Delta\gamma)
\end{Bmatrix}
= \bm{0}.
```

Here `σᵗʳ` is the trial stress, `ℂᵉ` is the elastic stiffness, `q` is the von
Mises stress, `σy0` is the initial yield stress, and `H` is the isotropic
hardening modulus. The flow direction `n` is the stress derivative of the yield
function.

The chosen trial stress is in the plastic branch. If `q(σᵗʳ) ≤ σy0`, the update
is elastic: `σ = σᵗʳ` and `Δγ = 0`.

```@example direct-sum
σᵗʳ = SymmetricSecondOrderTensor{3}((2.0, 0.4, 0.2, 1.2, 0.1, 0.9))
K = 10.0  # bulk modulus
G = 5.0   # shear modulus
σy0 = 0.6 # initial yield stress
H = 2.0   # isotropic hardening modulus

Ivol = vol(SymmetricFourthOrderTensor{3}) # volumetric projector
Idev = dev(SymmetricFourthOrderTensor{3}) # deviatoric projector
ℂᵉ = 3K * Ivol + 2G * Idev

q(σ) = sqrt(3/2) * norm(dev(σ)) # von Mises stress
yield_function(σ, Δγ) = q(σ) - (σy0 + H * Δγ)

function R(x)
    σ, Δγ = unpack(x)
    # flow direction and yield-function value
    n, f = gradient(σ -> yield_function(σ, Δγ), σ, :all)
    R_σ = σ - σᵗʳ + Δγ * (ℂᵉ ⊡₂ n)
    pack(R_σ, f)
end

x = pack(σᵗʳ, 0.0)
unpack(R(x))
```

The Jacobian is obtained directly from the packed residual:

```@example direct-sum
J = gradient(R, x)
n = gradient(σ -> yield_function(σ, 0.0), σᵗʳ)

(
    unpack(J, 1, 1) ≈ one(SymmetricFourthOrderTensor{3}),
    unpack(J, 1, 2) ≈ ℂᵉ ⊡₂ n,
    unpack(J, 2, 1) ≈ n,
    unpack(J, 2, 2) ≈ -H,
)
```

A Newton correction can then be written in direct-sum form:

```@example direct-sum
δx = -J \ R(x)
xnew = x + δx
σ, Δγ = unpack(xnew)
norm(R(σ ⊕ Δγ))
```

For this radial-return example, one Newton update gives the return-mapping
solution, so the residual is zero for the updated state. More general
return-mapping problems may need several Newton iterations.

For direct-sum docstrings, see [Direct sum API](@ref).
