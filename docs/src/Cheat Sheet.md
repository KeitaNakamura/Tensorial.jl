# Cheat Sheet

## Constructors

```julia
# identity tensors
one(Tensor{Tuple{3,3}})            == Matrix(1I,3,3) # second-order identity tensor
one(Tensor{Tuple{@Symmetry{3,3}}}) == Matrix(1I,3,3) # symmetric second-order identity tensor
I  = one(Tensor{NTuple{4,3}})               # fourth-order identity tensor
Is = one(Tensor{NTuple{2, @Symmetry{3,3}}}) # symmetric fourth-order identity tensor

# zero tensors
zero(Mat{2,3}) == zeros(2, 3)
zero(Tensor{Tuple{@Symmetry{3,3}}}) == zeros(3, 3)

# random tensors
rand(Mat{2,3})
randn(Mat{2,3})

# from arrays
Mat{2,2}([1 2; 3 4]) == [1 2; 3 4]
Tensor{Tuple{@Symmetry{2,2}}}([1 2; 3 4]) == [1 3; 3 4] # lower triangular part is used

# from functions
Mat{2,2}((i,j) -> i == j ? 1 : 0) == one(Mat{2,2})
Tensor{Tuple{@Symmetry{2,2}}}((i,j) -> i == j ? 1 : 0) == one(Tensor{Tuple{@Symmetry{2,2}}})

# macros (same interface as StaticArrays.jl)
@Vec [1,2,3]
@Vec rand(4)
@Mat [1 2
      3 4]
@Mat rand(4,4)
@Tensor rand(2,2,2)
```

## Tensor Operations

```julia
# 2nd-order vs. 2nd-order
x = rand(Mat{2,2})
y = rand(Tensor{Tuple{@Symmetry{2,2}}})
x ⊗ y isa Tensor{Tuple{2,2,@Symmetry{2,2}}} # tensor product
x ⋅ y isa Tensor{Tuple{2,2}}                # single contraction (x_ij * y_jk)
x ⊡ y isa Real                              # double contraction (x_ij * y_ij)

# 3rd-order vs. 1st-order
A = rand(Tensor{Tuple{@Symmetry{2,2},2}})
v = rand(Vec{2})
A ⊗ v isa Tensor{Tuple{@Symmetry{2,2},2,2}} # A_ijk * v_l
A ⋅ v isa Tensor{Tuple{@Symmetry{2,2}}}     # A_ijk * v_k
A ⊡ v # error

# 4th-order vs. 2nd-order
II = one(SymmetricFourthOrderTensor{2}) # equal to one(Tensor{Tuple{@Symmetry{2,2}, @Symmetry{2,2}}})
A = rand(Mat{2,2})
S = rand(Tensor{Tuple{@Symmetry{2,2}}})
II ⊡ A == (A + A') / 2 == symmetric(A) # symmetrizing A, resulting in Tensor{Tuple{@Symmetry{2,2}}}
II ⊡ S == S

# contraction
x = rand(Tensor{Tuple{2,2,2}})
y = rand(Tensor{Tuple{2,@Symmetry{2,2}}})
contraction(x, y, Val(1)) isa Tensor{Tuple{2,2, @Symmetry{2,2}}}     # single contraction (== ⋅)
contraction(x, y, Val(2)) isa Tensor{Tuple{2,2}}                     # double contraction (== ⊡)
contraction(x, y, Val(3)) isa Real                                   # triple contraction (x_ijk * y_ijk)
contraction(x, y, Val(0)) isa Tensor{Tuple{2,2,2,2, @Symmetry{2,2}}} # tensor product (== ⊗)

# norm/tr/mean/vol/dev
x = rand(SecondOrderTensor{3}) # equal to rand(Tensor{Tuple{3,3}})
v = rand(Vec{3})
norm(v)
tr(x)
mean(x) == tr(x) / 3 # useful for computing mean stress
vol(x) + dev(x) == x # decomposition into volumetric part and deviatoric part

# det/inv for 2nd-order tensor
A = rand(SecondOrderTensor{3})          # equal to one(Tensor{Tuple{3,3}})
S = rand(SymmetricSecondOrderTensor{3}) # equal to one(Tensor{Tuple{@Symmetry{3,3}}})
det(A); det(S)
inv(A) ⋅ A ≈ one(A)
inv(S) ⋅ S ≈ one(S)

# inv for 4th-order tensor
AA = rand(FourthOrderTensor{3})          # equal to one(Tensor{Tuple{3,3,3,3}})
SS = rand(SymmetricFourthOrderTensor{3}) # equal to one(Tensor{Tuple{@Symmetry{3,3}, @Symmetry{3,3}}})
inv(AA) ⊡ AA ≈ one(AA)
inv(SS) ⊡ SS ≈ one(SS)

# Einstein summation convention (experimental)
A = rand(Mat{3,3})
B = rand(Mat{3,3})
@einsum (i,j) -> A[i,k] * B[k,j]
@einsum A[i,j] * B[i,j]
```

## Automatic differentiation

```julia
# Real -> Real
gradient(x -> 2x^2 + x + 3, 3) == (x = 3; 4x + 1)
gradient(x -> 2.0, 3) == 0.0

# Real -> Tensor
gradient(x -> Mat{2,2}((i,j) -> i*x^2), 3) == (x = 3; Mat{2,2}((i,j) -> 2i*x))
gradient(x -> one(Mat{2,2}), 3) == zero(Mat{2,2})

# Tensor -> Real
gradient(tr, rand(Mat{3,3})) == one(Mat{3,3})

# Tensor -> Tensor
A = rand(Mat{3,3})
D  = gradient(dev, A)            # deviatoric projection tensor
Ds = gradient(dev, symmetric(A)) # symmetric deviatoric projection tensor
A ⊡ D  ≈ dev(A)
A ⊡ Ds ≈ symmetric(dev(A))
gradient(identity, A)  == one(FourthOrderTensor{3})          # 4th-order identity tensor
gradient(symmetric, A) == one(SymmetricFourthOrderTensor{3}) # symmetric 4th-order identity tensor
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
