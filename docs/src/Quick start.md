# Quick start

```@setup quick-start
using Tensorial
```

```@repl quick-start
using Tensorial
x = Vec{3}(rand(3)); # constructor similar to SArray.jl
A = @Mat rand(3,3); # @Vec, @Mat and @Tensor, analogous to @SVector, @SMatrix and @SArray
A ⋅ x ≈ A * x   # single contraction (⋅)
A ⊡ A ≈ tr(A'A) # double contraction (⊡)
x ⊗ x ≈ x * x'  # tensor product (⊗)
(@einsum x[i] * A[j,i] * x[j]) ≈ x ⋅ A' ⋅ x # Einstein summation (@einsum)
S = rand(Tensor{Tuple{@Symmetry{3,3}}}); # specify symmetry S₍ᵢⱼ₎
SS = rand(Tensor{Tuple{@Symmetry{3,3}, @Symmetry{3,3}}}); # SS₍ᵢⱼ₎₍ₖₗ₎
inv(SS) ⊡ S ≈ @einsum inv(SS)[i,j,k,l] * S[k,l] # it just works
δ = one(Mat{3,3}) # identity tensor
gradient(identity, S) ≈ one(SS) # ∂Sᵢⱼ/∂Sₖₗ = (δᵢₖδⱼₗ + δᵢₗδⱼₖ) / 2
```

## Aliases

```julia
const Vec{dim, T} = Tensor{Tuple{dim}, T, 1, dim}
const Mat{m, n, T, L} = Tensor{Tuple{m, n}, T, 2, L}
const SecondOrderTensor{dim, T, L} = Tensor{NTuple{2, dim}, T, 2, L}
const FourthOrderTensor{dim, T, L} = Tensor{NTuple{4, dim}, T, 4, L}
const SymmetricSecondOrderTensor{dim, T, L} = Tensor{Tuple{@Symmetry{dim, dim}}, T, 2, L}
const SymmetricFourthOrderTensor{dim, T, L} = Tensor{NTuple{2, @Symmetry{dim, dim}}, T, 4, L}
```
