# simd version is defined in simd.jl
@generated function _map(f, xs::Vararg{AbstractTensor, N}) where {N}
    S = promote_space(map(Space, xs)...)
    exps = map(tensorindices_tuple(S)) do i
        vals = [:(xs[$j][$i]) for j in 1:N]
        :(f($(vals...)))
    end
    TT = tensortype(S)
    return quote
        @_inline_meta
        data = tuple($(exps...))
        T = promote_ntuple_eltype(data)
        @inbounds $TT{T}(data)
    end
end

@inline Base.:+(x::AbstractTensor, y::AbstractTensor) = _map(+, x, y)
@inline Base.:-(x::AbstractTensor, y::AbstractTensor) = _map(-, x, y)
@inline Base.:*(y::Number, x::AbstractTensor) = _map(x -> x*y, x)
@inline Base.:*(x::AbstractTensor, y::Number) = _map(x -> x*y, x)
@inline Base.:/(x::AbstractTensor, y::Number) = _map(x -> x/y, x)
@inline Base.:+(x::AbstractTensor) = x
@inline Base.:-(x::AbstractTensor) = _map(-, x)

# with AbstractArray
@inline Base.:+(x::AbstractTensor, y::AbstractArray) = Tensor(SArray(x) + y)
@inline Base.:+(x::AbstractArray, y::AbstractTensor) = Tensor(x + SArray(y))
@inline Base.:-(x::AbstractTensor, y::AbstractArray) = Tensor(SArray(x) - y)
@inline Base.:-(x::AbstractArray, y::AbstractTensor) = Tensor(x - SArray(y))
# with StaticArray
@inline Base.:+(x::AbstractTensor, y::StaticArray) = Tensor(SArray(x) + y)
@inline Base.:+(x::StaticArray, y::AbstractTensor) = Tensor(x + SArray(y))
@inline Base.:-(x::AbstractTensor, y::StaticArray) = Tensor(SArray(x) - y)
@inline Base.:-(x::StaticArray, y::AbstractTensor) = Tensor(x - SArray(y))

@inline _add_uniform(x::AbstractSquareTensor, λ::Number) = x + λ * one(x)
@inline Base.:+(x::AbstractTensor, y::UniformScaling) = _add_uniform( x,  y.λ)
@inline Base.:-(x::AbstractTensor, y::UniformScaling) = _add_uniform( x, -y.λ)
@inline Base.:+(x::UniformScaling, y::AbstractTensor) = _add_uniform( y,  x.λ)
@inline Base.:-(x::UniformScaling, y::AbstractTensor) = _add_uniform(-y,  x.λ)
@inline contract1(x::Union{AbstractVec, AbstractMat, AbstractSquareTensor}, y::UniformScaling) = x * y.λ
@inline contract1(x::UniformScaling, y::Union{AbstractVec, AbstractMat, AbstractSquareTensor}) = x.λ * y
@inline contract2(x::AbstractSquareTensor, y::UniformScaling) = tr(x) * y.λ
@inline contract2(x::UniformScaling, y::AbstractSquareTensor) = x.λ * tr(y)

# multiplication
@inline Base.:*(x::AbstractVecOrMatLike, y::AbstractVecOrMatLike) = Tensor(SArray(x) * SArray(y))
@inline Base.:*(x::LinearAlgebra.Transpose{T, <: AbstractVec{<: Any, T}}, y::AbstractVec{<: Any, U}) where {T <: Real, U <: Real} = parent(x) ⋅ y
@inline Base.:*(x::AbstractVecOrMatLike, I::UniformScaling) = x * I.λ
@inline Base.:*(I::UniformScaling, x::AbstractVecOrMatLike) = I.λ * x

"""
    contract(x, y, ::Val{N})

Conduct contraction of `N` inner indices.
For example, `N=2` contraction for third-order tensors ``A_{ij} = B_{ikl} C_{klj}``
can be computed as follows:

# Examples
```jldoctest
julia> B = rand(Tensor{Tuple{3,3,3}});

julia> C = rand(Tensor{Tuple{3,3,3}});

julia> A = contract(B, C, Val(2))
3×3 Tensor{Tuple{3, 3}, Float64, 2, 9}:
 3.70978  2.47156  3.91807
 2.90966  2.30881  3.25965
 1.78391  1.38714  2.2079
```

The following infix operators are also available for specific contractions:

- `x ⊡ y` (where `⊡` can be typed by `\\boxdot<tab>` ): `contract(x, y, Val(1))`
- `x ⊡₂ y` (where `⊡₂` can be typed by `\\boxdot<tab>\\_2<tab>` ): `contract(x, y, Val(2))`
- `x ⊗ y` (where `⊗` can be typed by `\\otimes<tab>` ): `contract(x, y, Val(0))`
"""
@generated function contract(t1::AbstractTensor, t2::AbstractTensor, ::Val{N}) where {N}
    _check_contract(t1, t2, Val(N))
    S1, S2 = Space(t1), Space(t2)
    Scon = promote_space(dropfirst(S1, Val(ndims(t1)-N)),
                          droplast(S2, Val(ndims(t2)-N)))
    S1′, S2′ = ⊗(droplast(S1, Val(N)), Scon), ⊗(Scon, dropfirst(S2, Val(N)))
    K = ncomponents(Scon)
    I, J = ncomponents(S1′)÷K, ncomponents(S2′)÷K
    TT = tensortype(contract(S1′, S2′, Val(N)))
    dups = nduplicates_tuple(Scon)
    if all(isone, dups)
        vec2′ = :(vec(arr2))
        arr2′ = :(arr2)
    else
        vec2′ = :($dups .* vec(arr2))
        arr2′ = :($dups .* arr2)
    end
    quote
        @_inline_meta
        arr1 = SMatrix{$I,$K}(Tuple(convert($(tensortype(S1′)), t1)))
        arr2 = SMatrix{$K,$J}(Tuple(convert($(tensortype(S2′)), t2)))
        $(ndims(TT) == 0) && return dot_unrolled(vec(arr1), $vec2′)
        data = Tuple(mul_unrolled(arr1, $arr2′))
        $TT(data)
    end
end
@inline contract(t::AbstractTensor, a::Number, ::Val{0}) = t * a
@inline contract(a::Number, t::AbstractTensor, ::Val{0}) = a * t
@inline contract(a::Number, b::Number, ::Val{0}) = a * b
@inline contract(a, b, nth) = contract(a, b, Val(nth))

@noinline _throw_contract_dmm(axA, axB) = throw(DimensionMismatch("neighbouring axes of `A` and `B` must match, got $axA and $axB"))
@noinline _throw_contract_ndims(ndims, n) = throw(ArgumentError("contraction order should be ≤ ndims(A) = $ndims, got $n"))
@noinline _throw_contract_nth(n) = throw(ArgumentError("contraction order should be ≥ 0, got $n"))

@generated function _check_contract(::Type{A}, ::Type{B}, ::Val{K}) where {A<:AbstractTensor,B<:AbstractTensor,K}
    K::Int
    N, M = ndims(A), ndims(B)
    K ≥ 0 || return :(_throw_contract_nth($K))
    N ≥ K || return :(_throw_contract_ndims($N, $K))
    M ≥ K || return :(_throw_contract_ndims($M, $K))
    for i in 1:K
        axA, axB = axes(A)[N-K+i], axes(B)[i]
        axA == axB || return :(_throw_contract_dmm($axA, $axB))
    end
end

@generated function dot_unrolled(x::SVector{N}, y::SVector{N}) where {N}
    ex = :(x[1] * y[1])
    for i in 2:N
        ex = :(muladd(x[$i], y[$i], $ex))
    end
    quote
        @_inline_meta
        @inbounds $ex
    end
end
@generated function mul_unrolled(x::SMatrix{m,l}, y::SMatrix{l,n}) where {l,m,n}
    exps = map(CartesianIndices((m,n))) do I
        i, j = Tuple(I)
        :(dot_unrolled(x[$i,:], y[:,$j]))
    end
    quote
        @_inline_meta
        @inbounds SMatrix{m,n}($(exps...))
    end
end

"""
    contract(x, y, Val(xdims), Val(ydims))

Perform contraction over the given dimensions.

# Examples
```jldoctest
julia> A = rand(Mat{3,3})
3×3 Tensor{Tuple{3, 3}, Float64, 2, 9}:
 0.325977  0.894245  0.953125
 0.549051  0.353112  0.795547
 0.218587  0.394255  0.49425

julia> B = rand(Mat{3,3})
3×3 Tensor{Tuple{3, 3}, Float64, 2, 9}:
 0.748415  0.00744801  0.682533
 0.578232  0.199377    0.956741
 0.727935  0.439243    0.647855

julia> contract(A, B, Val(1), Val(2)) ≈ @einsum A[k,i] * B[j,k]
true

julia> contract(A, B, Val((1,2)), Val((2,1))) ≈ @einsum A[i,j] * B[j,i]
true
```
"""
@generated function contract(x::AbstractTensor, y::AbstractTensor, ::Val{xdims}, ::Val{ydims}) where {xdims, ydims}
    xdims = check_contract_dims(xdims)
    ydims = check_contract_dims(ydims)
    @assert length(xdims) == length(ydims)
    xperm = [setdiff(1:ndims(x), xdims); xdims]
    yperm = [ydims; setdiff(1:ndims(y), ydims)]
    quote
        @_inline_meta
        contract(permutedims(x, $(ValTuple(xperm...))),
                 permutedims(y, $(ValTuple(yperm...))),
                 $(Val(length(xdims))))
    end
end

@generated function contract(::Type{TT}, x::AbstractTensor, y::AbstractTensor, ::Val{xdims}, ::Val{ydims}) where {TT, xdims, ydims}
    xdims = check_contract_dims(xdims)
    ydims = check_contract_dims(ydims)
    @assert length(xdims) == length(ydims)
    I = [length(xdims), ndims(x)-length(xdims), ndims(y)-length(ydims)]
    dummy_indices, x_free_indices, y_free_indices = UnitRange.(cumsum(I) - I .+ 1, cumsum(I))
    function create_indices(t, dims, free_indices)
        indices = collect(1:ndims(t))
        indices[dims] = dummy_indices
        indices[setdiff(1:ndims(t), dims)] .= free_indices
        Tuple(indices)
    end
    xindices = create_indices(x, xdims, x_free_indices)
    yindices = create_indices(y, ydims, y_free_indices)
    quote
        @_inline_meta
        contract_einsum(TT, (x,y), ($(Val(xindices)),$(Val(yindices))))
    end
end

function check_contract_dims(dims)
    @assert dims isa Union{Int, Tuple{Vararg{Int}}}
    dims = dims isa Int ? [dims] : collect(dims)
    @assert allunique(dims)
    dims
end

@inline contract1(x1::AbstractTensor, x2::AbstractTensor) = contract(x1, x2, Val(1))
@inline contract2(x1::AbstractTensor, x2::AbstractTensor) = contract(x1, x2, Val(2))
@inline contract3(x1::AbstractTensor, x2::AbstractTensor) = contract(x1, x2, Val(3))
@inline contract4(x1::AbstractTensor, x2::AbstractTensor) = contract(x1, x2, Val(4))
@inline contract5(x1::AbstractTensor, x2::AbstractTensor) = contract(x1, x2, Val(5))
@inline contract6(x1::AbstractTensor, x2::AbstractTensor) = contract(x1, x2, Val(6))
@inline contract7(x1::AbstractTensor, x2::AbstractTensor) = contract(x1, x2, Val(7))
@inline contract8(x1::AbstractTensor, x2::AbstractTensor) = contract(x1, x2, Val(8))
@inline contract9(x1::AbstractTensor, x2::AbstractTensor) = contract(x1, x2, Val(9))

@inline dot(x1::AbstractTensor{<: Any, <: Any, N}, x2::AbstractTensor{<: Any, <: Any, N}) where {N} = contract(x1, x2, Val(N))

"""
    tensor(x::AbstractTensor, y::AbstractTensor)
    x ⊗ y

Compute tensor product such as ``A_{ij} = x_i y_j``.
`x ⊗ y` (where `⊗` can be typed by `\\otimes<tab>`) is a synonym for `tensor(x, y)`.

# Examples
```jldoctest
julia> x = rand(Vec{3})
3-element Vec{3, Float64}:
 0.32597672886359486
 0.5490511363155669
 0.21858665481883066

julia> y = rand(Vec{3})
3-element Vec{3, Float64}:
 0.8942454282009883
 0.35311164439921205
 0.39425536741585077

julia> A = x ⊗ y
3×3 Tensor{Tuple{3, 3}, Float64, 2, 9}:
 0.291503  0.115106   0.128518
 0.490986  0.193876   0.216466
 0.19547   0.0771855  0.086179
```
"""
@inline tensor(x1::Union{AbstractTensor, Number}, x2::Union{AbstractTensor, Number}) = contract(x1, x2, Val(0))
@inline tensor(x1::Union{AbstractTensor, Number}, x2::Union{AbstractTensor, Number}, others...) = tensor(tensor(x1, x2), others...)
@inline tensor(x::Union{AbstractTensor, Number}) = x

struct OTimes{N} end
tensor(N::Int) = OTimes{N}()

@inline Base.:^(x::AbstractVec, ::OTimes{0}) = one(eltype(x))
@inline Base.:^(x::AbstractVec, ::OTimes{1}) = x

"""
    x^⊗(n)

`n`-fold tensor product of a tensor `x`.

# Examples
```jldoctest
julia> x = rand(Vec{2})
2-element Vec{2, Float64}:
 0.32597672886359486
 0.5490511363155669

julia> x^⊗(3)
2×2×2 Tensor{Tuple{Symmetry{Tuple{2, 2, 2}}}, Float64, 3, 4}:
[:, :, 1] =
 0.0346386  0.0583426
 0.0583426  0.098268

[:, :, 2] =
 0.0583426  0.098268
 0.098268   0.165515
```
"""
@generated function Base.:^(x::AbstractTensor, ::OTimes{N}) where {N}
    ex = :x
    for i in 1:N-1
        ex = :(⊗($ex, x))
    end
    quote
        @_inline_meta
        $ex
    end
end
@generated function Base.:^(x::AbstractVec{dim}, ::OTimes{N}) where {dim, N}
    ex = :x
    for i in 1:N-1
        ex = :(_pow_otimes($ex, x))
    end
    quote
        @_inline_meta
        $ex
    end
end
@inline function _pow_otimes(x::Tensor{Tuple{Symmetry{NTuple{N, dim}}}}, y::Vec{dim}) where {N, dim}
    contract(Tensor{Tuple{Symmetry{NTuple{N+1,dim}}}}, x, y, Val(()), Val(()))
end
@inline function _pow_otimes(x::Vec{dim}, y::Vec{dim}) where {dim}
    contract(Tensor{Tuple{@Symmetry{dim,dim}}}, x, y, Val(()), Val(()))
end

"""
    norm(::AbstractTensor)

Compute norm of a tensor.

# Examples
```jldoctest
julia> x = rand(Mat{3, 3})
3×3 Tensor{Tuple{3, 3}, Float64, 2, 9}:
 0.325977  0.894245  0.953125
 0.549051  0.353112  0.795547
 0.218587  0.394255  0.49425

julia> norm(x)
1.8223398556552728
```
"""
@inline norm(x::AbstractTensor) = sqrt(contract(x, x, Val(ndims(x))))

"""
    normalize(x)

Compute `x / norm(x)`.
"""
@inline normalize(x::AbstractTensor) = x / norm(x)

"""
    tr(A)

Compute the trace of a square tensor `A`.

# Examples
```jldoctest
julia> A = rand(Mat{3,3})
3×3 Tensor{Tuple{3, 3}, Float64, 2, 9}:
 0.325977  0.894245  0.953125
 0.549051  0.353112  0.795547
 0.218587  0.394255  0.49425

julia> tr(A)
1.1733382401532275
```
"""
@generated function tr(x::AbstractSquareTensor{dim}) where {dim}
    exps = [getindex_expr(x,:x,i,i) for i in 1:dim]
    quote
        @_inline_meta
        @inbounds +($(exps...))
    end
end

"""
    symmetric(::AbstractSecondOrderTensor)
    symmetric(::AbstractSecondOrderTensor, uplo)

Compute the symmetric part of a second order tensor.

# Examples
```jldoctest
julia> x = rand(Mat{3,3})
3×3 Tensor{Tuple{3, 3}, Float64, 2, 9}:
 0.325977  0.894245  0.953125
 0.549051  0.353112  0.795547
 0.218587  0.394255  0.49425

julia> symmetric(x)
3×3 SymmetricSecondOrderTensor{3, Float64, 6}:
 0.325977  0.721648  0.585856
 0.721648  0.353112  0.594901
 0.585856  0.594901  0.49425

julia> symmetric(x, :U)
3×3 SymmetricSecondOrderTensor{3, Float64, 6}:
 0.325977  0.894245  0.953125
 0.894245  0.353112  0.795547
 0.953125  0.795547  0.49425
```
"""
@inline symmetric(x::AbstractSymmetricSecondOrderTensor) = x
@inline function symmetric(x::AbstractSymmetricSecondOrderTensor, uplo::Symbol)
    if uplo == :U || uplo == :L
        return x
    end
    throw(ArgumentError("uplo argument must be either :U (upper) or :L (lower)"))
end
## :U or :L
for (uplo, filter) in ((:U, (inds,i,j) -> i ≤ j ? inds[i,j] : inds[j,i]),
                       (:L, (inds,i,j) -> i ≥ j ? inds[i,j] : inds[j,i]))
    @eval @generated function $(Symbol(:symmetric, uplo))(x::AbstractSecondOrderTensor{dim}) where {dim}
        exps = [:(Tuple(x)[$($filter(tupleindices_tensor(x), Tuple(I)...))]) for I in CartesianIndices(x)]
        TT = SymmetricSecondOrderTensor{dim}
        quote
            @_inline_meta
            @inbounds $TT($(exps[tensorindices_tuple(TT)]...))
        end
    end
end
@inline function symmetric(x::AbstractSecondOrderTensor{dim}, uplo::Symbol) where {dim}
    uplo == :U && return symmetricU(x)
    uplo == :L && return symmetricL(x)
    throw(ArgumentError("uplo argument must be either :U (upper) or :L (lower)"))
end
## no uplo
@inline symmetric(x::AbstractSecondOrderTensor{dim}) where {dim} =
    SymmetricSecondOrderTensor{dim}((i,j) -> @inbounds i == j ? x[i,j] : (x[i,j] + x[j,i]) / 2)
@inline symmetric(x::AbstractSecondOrderTensor{1}) =
    @inbounds SymmetricSecondOrderTensor{1}(x[1])
@inline symmetric(x::AbstractSecondOrderTensor{2}) =
    @inbounds SymmetricSecondOrderTensor{2}(x[1], (x[2]+x[3])/2, x[4])
@inline symmetric(x::AbstractSecondOrderTensor{3}) =
    @inbounds SymmetricSecondOrderTensor{3}(x[1], (x[2]+x[4])/2, (x[3]+x[7])/2, x[5], (x[6]+x[8])/2, x[9])

"""
    minorsymmetric(::AbstractFourthOrderTensor) -> SymmetricFourthOrderTensor

Compute the minor symmetric part of a fourth order tensor.

# Examples
```jldoctest
julia> x = rand(Tensor{Tuple{3,3,3,3}});

julia> minorsymmetric(x) ≈ @einsum (i,j,k,l) -> (x[i,j,k,l] + x[j,i,k,l] + x[i,j,l,k] + x[j,i,l,k]) / 4
true
```
"""
@inline function minorsymmetric(x::AbstractFourthOrderTensor{dim}) where {dim}
    SymmetricFourthOrderTensor{dim}(
        @inline function(i,j,k,l)
            @inbounds i==j && k==l ? x[i,j,k,l] : (x[i,j,k,l]+x[j,i,k,l]+x[i,j,l,k]+x[j,i,l,k])/4
        end
    )
end
@inline minorsymmetric(x::AbstractSymmetricFourthOrderTensor) = x

"""
    skew(A)

Compute skew-symmetric (anti-symmetric) part of a second order tensor.

# Examples
```jldoctest
julia> x = rand(Mat{3,3})
3×3 Tensor{Tuple{3, 3}, Float64, 2, 9}:
 0.325977  0.894245  0.953125
 0.549051  0.353112  0.795547
 0.218587  0.394255  0.49425

julia> symmetric(x) + skew(x) ≈ x
true
```
"""
@inline skew(x::AbstractSecondOrderTensor) = (x - x') / 2
@inline skew(x::AbstractSymmetricSecondOrderTensor{dim, T}) where {dim, T} = zero(SecondOrderTensor{dim, T})

"""
    skew(ω::Vec{3})

Construct a skew-symmetric (anti-symmetric) tensor `W` from a vector `ω` as

```math
\\bm{\\omega} = \\begin{Bmatrix}
    \\omega_1 \\\\
    \\omega_2 \\\\
    \\omega_3
\\end{Bmatrix}, \\quad
\\bm{W} = \\begin{bmatrix}
     0         & -\\omega_3 &  \\omega_2 \\\\
     \\omega_3 & 0          & -\\omega_1 \\\\
    -\\omega_2 &  \\omega_1 &  0
\\end{bmatrix}
```

# Examples
```jldoctest
julia> skew(Vec(1,2,3))
3×3 Tensor{Tuple{3, 3}, Int64, 2, 9}:
  0  -3   2
  3   0  -1
 -2   1   0
```
"""
@inline function skew(ω::Vec{3})
    z = zero(eltype(ω))
    @inbounds @Mat [ z    -ω[3]  ω[2]
                     ω[3]     z -ω[1]
                    -ω[2]  ω[1]     z]
end

# transpose/adjoint
@inline transpose(x::AbstractTensor{Tuple{@Symmetry{dim, dim}}}) where {dim} = x
@inline transpose(x::AbstractTensor{Tuple{m, n}}) where {m, n} = Tensor{Tuple{n, m}}((i,j) -> @inbounds x[j,i])
@inline adjoint(x::AbstractTensor) = transpose(x)

# det
@generated function extract_vecs(x::AbstractSquareTensor{dim}) where {dim}
    exps = map(1:dim) do j
        :(Vec($([getindex_expr(x, :x, i, j) for i in 1:dim]...)))
    end
    quote
        @_inline_meta
        @inbounds tuple($(exps...))
    end
end
@inline function det(x::AbstractSquareTensor{1})
    @inbounds x[1,1]
end
@inline function det(x::AbstractSquareTensor{2})
    @inbounds x[1,1] * x[2,2] - x[1,2] * x[2,1]
end
@inline function det(A::AbstractSquareTensor{3})
    a1, a2, a3 = extract_vecs(A)
    (a1 × a2) ⋅ a3
end
@inline function det(x::AbstractSquareTensor{dim}) where {dim}
    det(SMatrix{dim, dim}(x))
end

# AD insertion
@inline function det(g::AbstractSquareTensor{dim, <: Dual{Tg}}) where {dim, F, V, Tg <: Tag{F,V}}
    x = extract_value(g)
    f = det(x)
    dfdg = adj(x)'
    dgdx = extract_gradient(g, zero(V))
    create_dual(Tg(), f, dfdg ⊡₂ dgdx)
end

"""
    cross(x::Vec, y::Vec)
    x × y

Compute the cross product between two vectors.
The infix operation `x × y` (where `×` can be typed by `\\times<tab>`) is a synonym for `cross(x, y)`.

# Examples
```jldoctest
julia> x = rand(Vec{3})
3-element Vec{3, Float64}:
 0.32597672886359486
 0.5490511363155669
 0.21858665481883066

julia> y = rand(Vec{3})
3-element Vec{3, Float64}:
 0.8942454282009883
 0.35311164439921205
 0.39425536741585077

julia> x × y
3-element Vec{3, Float64}:
  0.13928086435138393
  0.0669520417303531
 -0.37588028973385323
```
"""
@inline function cross(x::Vec{3}, y::Vec{3})
    @inbounds Vec(x[2]*y[3] - x[3]*y[2],
                  x[3]*y[1] - x[1]*y[3],
                  x[1]*y[2] - x[2]*y[1])
end
@inline function cross(x::Vec{2}, y::Vec{2})
    @inbounds x[1]*y[2] - x[2]*y[1]
end

# power
@inline Base.literal_pow(::typeof(^), x::AbstractSquareTensor, ::Val{-1}) = inv(x)
@inline Base.literal_pow(::typeof(^), x::AbstractSquareTensor, ::Val{0})  = one(x)
@inline Base.literal_pow(::typeof(^), x::AbstractSquareTensor, ::Val{1})  = x
@inline function Base.literal_pow(::typeof(^), x::AbstractSquareTensor, ::Val{p}) where {p}
    p > 0 ? (y = x; q = p) : (y = inv(x); q = -p)
    z = y
    for i in 2:q
        y = _powdot(y, z)
    end
    y
end
## helper functions
@inline _powdot(x::AbstractSecondOrderTensor, y::AbstractSecondOrderTensor) = contract1(x, y)
@inline function _powdot(x::AbstractSymmetricSecondOrderTensor{dim}, y::AbstractSymmetricSecondOrderTensor{dim}) where {dim}
    contract(SymmetricSecondOrderTensor{dim}, x, y, Val(2), Val(1))
end

# rotate
"""
    rotmat(θ::Number)

Construct 2D rotation matrix.

```math
\\bm{R} = \\begin{bmatrix}
\\cos{\\theta} & -\\sin{\\theta} \\\\
\\sin{\\theta} &  \\cos{\\theta}
\\end{bmatrix}
```

# Examples
```jldoctest
julia> rotmat(deg2rad(30))
2×2 Tensor{Tuple{2, 2}, Float64, 2, 4}:
 0.866025  -0.5
 0.5        0.866025
```
"""
@inline function rotmat(θ::Number)
    sinθ, cosθ = sincos(θ)
    @Mat [cosθ -sinθ
          sinθ  cosθ]
end

"""
    rotmat(θ::Vec{3}; sequence::Symbol)

Convert Euler angles to rotation matrix.
Use 3 characters belonging to the set (X, Y, Z) for intrinsic rotations,
or (x, y, z) for extrinsic rotations.

# Examples
```jldoctest
julia> α, β, γ = map(deg2rad, rand(3));

julia> rotmat(Vec(α,β,γ), sequence = :XYZ) ≈ rotmatx(α) * rotmaty(β) * rotmatz(γ)
true

julia> rotmat(Vec(α,β,γ), sequence = :xyz) ≈ rotmatz(γ) * rotmaty(β) * rotmatx(α)
true

julia> rotmat(Vec(α,β,γ), sequence = :XYZ) ≈ rotmat(Vec(γ,β,α), sequence = :zyx)
true
```
"""
function rotmat(θ::Vec{3}; sequence::Symbol)
    @inbounds α, β, γ = θ[1], θ[2], θ[3]
    # intrinsic
    sequence == :XZX && return rotmatx(α) * rotmaty(β) * rotmatz(γ)
    sequence == :XYX && return rotmatx(α) * rotmaty(β) * rotmatx(γ)
    sequence == :YXY && return rotmaty(α) * rotmatx(β) * rotmaty(γ)
    sequence == :YZY && return rotmaty(α) * rotmatz(β) * rotmaty(γ)
    sequence == :ZYZ && return rotmatz(α) * rotmaty(β) * rotmatz(γ)
    sequence == :ZXZ && return rotmatz(α) * rotmatx(β) * rotmatz(γ)
    sequence == :XZY && return rotmatx(α) * rotmatz(β) * rotmaty(γ)
    sequence == :XYZ && return rotmatx(α) * rotmaty(β) * rotmatz(γ)
    sequence == :YXZ && return rotmaty(α) * rotmatx(β) * rotmatz(γ)
    sequence == :YZX && return rotmaty(α) * rotmatz(β) * rotmatx(γ)
    sequence == :ZYX && return rotmatz(α) * rotmaty(β) * rotmatx(γ)
    sequence == :ZXY && return rotmatz(α) * rotmatx(β) * rotmaty(γ)
    # extrinsic
    sequence == :xzx && return rotmatx(γ) * rotmaty(β) * rotmatz(α)
    sequence == :xyx && return rotmatx(γ) * rotmaty(β) * rotmatx(α)
    sequence == :yxy && return rotmaty(γ) * rotmatx(β) * rotmaty(α)
    sequence == :yzy && return rotmaty(γ) * rotmatz(β) * rotmaty(α)
    sequence == :zyz && return rotmatz(γ) * rotmaty(β) * rotmatz(α)
    sequence == :zxz && return rotmatz(γ) * rotmatx(β) * rotmatz(α)
    sequence == :xzy && return rotmaty(γ) * rotmatz(β) * rotmatx(α)
    sequence == :xyz && return rotmatz(γ) * rotmaty(β) * rotmatx(α)
    sequence == :yxz && return rotmatz(γ) * rotmatx(β) * rotmaty(α)
    sequence == :yzx && return rotmatx(γ) * rotmatz(β) * rotmaty(α)
    sequence == :zyx && return rotmatx(γ) * rotmaty(β) * rotmatz(α)
    sequence == :zxy && return rotmaty(γ) * rotmatx(β) * rotmatz(α)
    throw(ArgumentError("sequence $sequence is not supported"))
end

"""
    rotmatx(θ::Number)

Construct rotation matrix around `x` axis.

```math
\\bm{R}_x = \\begin{bmatrix}
1 & 0 & 0 \\\\
0 & \\cos{\\theta} & -\\sin{\\theta} \\\\
0 & \\sin{\\theta} &  \\cos{\\theta}
\\end{bmatrix}
```
"""
@inline function rotmatx(θ::Number)
    o = one(θ)
    z = zero(θ)
    sinθ, cosθ = sincos(θ)
    @Mat [o z     z
          z cosθ -sinθ
          z sinθ  cosθ]
end

"""
    rotmaty(θ::Number)

Construct rotation matrix around `y` axis.

```math
\\bm{R}_y = \\begin{bmatrix}
\\cos{\\theta} & 0 & \\sin{\\theta} \\\\
0 & 1 & 0 \\\\
-\\sin{\\theta} & 0 & \\cos{\\theta}
\\end{bmatrix}
```
"""
@inline function rotmaty(θ::Number)
    o = one(θ)
    z = zero(θ)
    sinθ, cosθ = sincos(θ)
    @Mat [ cosθ z sinθ
           z    o z
          -sinθ z cosθ]
end

"""
    rotmatz(θ::Number)

Construct rotation matrix around `z` axis.

```math
\\bm{R}_z = \\begin{bmatrix}
\\cos{\\theta} & -\\sin{\\theta} & 0 \\\\
\\sin{\\theta} &  \\cos{\\theta} & 0 \\\\
0 & 0 & 1
\\end{bmatrix}
```
"""
@inline function rotmatz(θ::Number)
    o = one(θ)
    z = zero(θ)
    sinθ, cosθ = sincos(θ)
    @Mat [cosθ -sinθ z
          sinθ  cosθ z
          z     z    o]
end

"""
    rotmat(a => b)

Construct rotation matrix rotating vector `a` to `b`.
The norms of two vectors must be the same.

# Examples
```jldoctest
julia> a = normalize(rand(Vec{3}))
3-element Vec{3, Float64}:
 0.4829957515506539
 0.8135223859352438
 0.3238771859304809

julia> b = normalize(rand(Vec{3}))
3-element Vec{3, Float64}:
 0.8605677447967596
 0.3398133016944055
 0.3794075336718636

julia> R = rotmat(a => b)
3×3 Tensor{Tuple{3, 3}, Float64, 2, 9}:
 -0.00540771   0.853773   0.520617
  0.853773    -0.267108   0.446905
  0.520617     0.446905  -0.727485

julia> R * a ≈ b
true
```
"""
function rotmat(pair::Pair{Vec{dim, T}, Vec{dim, T}})::Mat{dim, dim, T} where {dim, T}
    # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d/2672702#2672702
    a = pair.first
    b = pair.second
    contract1(a, a) ≈ contract1(b, b) || throw(ArgumentError("the norms of two vectors must be the same"))
    a ==  b && return  one(Mat{dim, dim, T})
    a == -b && return -one(Mat{dim, dim, T})
    c = a + b
    2 * (c ⊗ c) / (c ⋅ c) - one(Mat{dim, dim, T})
end

"""
    rotmat(θ, n::Vec)

Construct rotation matrix from angle `θ` and axis `n`.

# Examples
```jldoctest
julia> x = Vec(1.0, 0.0, 0.0)
3-element Vec{3, Float64}:
 1.0
 0.0
 0.0

julia> n = Vec(0.0, 0.0, 1.0)
3-element Vec{3, Float64}:
 0.0
 0.0
 1.0

julia> rotmat(π/2, n) * x
3-element Vec{3, Float64}:
 6.123233995736766e-17
 1.0
 0.0
```
"""
function rotmat(θ::Number, n::Vec{3})
    n′ = normalize(n)
    sinθ, cosθ = sincos(θ)
    cosθ*I + sinθ*skew(n′) + (1-cosθ)*(n′⊗n′)
end

"""
    rotate(x, R::SecondOrderTensor)

Rotate `x` using the rotation matrix `R`.
This function preserves the symmetry of the matrix.

# Examples
```jldoctest
julia> R = rotmatz(π/4)
3×3 Tensor{Tuple{3, 3}, Float64, 2, 9}:
 0.707107  -0.707107  0.0
 0.707107   0.707107  0.0
 0.0        0.0       1.0

julia> rotate(Vec(1,0,0), R)
3-element Vec{3, Float64}:
 0.7071067811865476
 0.7071067811865475
 0.0

julia> A = rand(SymmetricSecondOrderTensor{3})
3×3 SymmetricSecondOrderTensor{3, Float64, 6}:
 0.325977  0.549051  0.218587
 0.549051  0.894245  0.353112
 0.218587  0.353112  0.394255

julia> rotate(A, R) ≈ R * A * R'
true
```
"""
@inline rotate(v::Vec, R::SecondOrderTensor) = R * v
@inline rotate(v::Vec{2}, R::SecondOrderTensor{3}) = rotate(vcat(v,0), R) # extend to 3d vector, then rotate it
@inline rotate(A::SecondOrderTensor, R::SecondOrderTensor) = R * A * R'
@inline function rotate(A::SymmetricSecondOrderTensor{dim}, R::SecondOrderTensor{dim}) where {dim}
    ARᵀ = contract(A, R, Val(2), Val(2))
    contract(SymmetricSecondOrderTensor{dim}, R, ARᵀ, Val(2), Val(1))
end

function angleaxis(R::SecondOrderTensor{3})
    # https://math.stackexchange.com/questions/893984/conversion-of-rotation-matrix-to-quaternion
    θ = acos((tr(R)-1) / 2)
    n = Tensor(Real.(eigvecs(SArray(R))[:,3]))
    θ, n
end

# exp/log
@inline function Base.exp(x::AbstractSymmetricSecondOrderTensor)
    F = eigen(x)
    symmetric(F.vectors * diagm(exp.(F.values)) * F.vectors')
end
@inline function Base.log(x::AbstractSymmetricSecondOrderTensor)
    F = eigen(x)
    symmetric(F.vectors * diagm(log.(F.values)) * F.vectors')
end

# ----------------------------------------------#
# operations calling methods in StaticArrays.jl #
# ----------------------------------------------#

# exp
@inline Base.exp(x::AbstractSecondOrderTensor) = typeof(x)(exp(SArray(x)))

# diag/diagm
@inline diag(x::Union{AbstractMat, AbstractSymmetricSecondOrderTensor}, ::Val{k} = Val(0)) where {k} = Tensor(diag(SArray(x), Val{k}))
@inline diagm(kvs::Pair{<: Val, <: AbstractVec}...) = Tensor(diagm(map(kv -> kv.first => SArray(kv.second), kvs)...))
@inline diagm(x::Vec) = diagm(Val(0) => x)

# qr
struct QR{Q, R, P}
    Q::Q
    R::R
    p::P
end

# iteration for destructuring into components
Base.iterate(S::QR) = (S.Q, Val(:R))
Base.iterate(S::QR, ::Val{:R}) = (S.R, Val(:p))
Base.iterate(S::QR, ::Val{:p}) = (S.p, Val(:done))
Base.iterate(S::QR, ::Val{:done}) = nothing

for pv in (:true, :false)
    @eval function qr(A::AbstractMat, pivot::Val{$pv})
        F = qr(SArray(A), pivot)
        QR(Tensor(F.Q), Tensor(F.R), Tensor(F.p))
    end
end
qr(A::AbstractMat) = qr(A, Val(false))

# lu
struct LU{L, U, p}
    L::L
    U::U
    p::p
end

# iteration for destructuring into components
Base.iterate(S::LU) = (S.L, Val(:U))
Base.iterate(S::LU, ::Val{:U}) = (S.U, Val(:p))
Base.iterate(S::LU, ::Val{:p}) = (S.p, Val(:done))
Base.iterate(S::LU, ::Val{:done}) = nothing

for pv in (:true, :false)
    @eval function lu(A::AbstractMat, pivot::Val{$pv}; check = true)
        F = lu(SArray(A), pivot; check = check)
        LU(LowerTriangular(Tensor(parent(F.L))), UpperTriangular(Tensor(parent(F.U))), Tensor(F.p))
    end
end
lu(A::AbstractMat; check = true) = lu(A, Val(true); check = check)

# eigen
@inline function eigvals(x::AbstractSymmetricSecondOrderTensor; permute::Bool=true, scale::Bool=true)
    Tensor(eigvals(Symmetric(SArray(x)); permute=permute, scale = scale))
end

@inline function eigen(x::AbstractSymmetricSecondOrderTensor; permute::Bool=true, scale::Bool=true)
    eig = eigen(Symmetric(SArray(x)); permute=permute, scale=scale)
    Eigen(Tensor(eig.values), Tensor(eig.vectors))
end

# svd
struct SVD{T, TU, TS, TVt} <: Factorization{T}
    U::TU
    S::TS
    Vt::TVt
end
SVD(U::AbstractArray{T}, S::AbstractVector, Vt::AbstractArray{T}) where {T} = SVD{T, typeof(U), typeof(S), typeof(Vt)}(U, S, Vt)

@inline function Base.getproperty(F::SVD, s::Symbol)
    if s === :V
        return getfield(F, :Vt)'
    else
        return getfield(F, s)
    end
end
Base.propertynames(::SVD) = (:U, :S, :V, :Vt)

# iteration for destructuring into components
Base.iterate(S::SVD) = (S.U, Val(:S))
Base.iterate(S::SVD, ::Val{:S}) = (S.S, Val(:V))
Base.iterate(S::SVD, ::Val{:V}) = (S.V, Val(:done))
Base.iterate(S::SVD, ::Val{:done}) = nothing

function svd(A::AbstractMat; full=Val(false))
    F = svd(SArray(A); full = full)
    SVD(Tensor(F.U), Tensor(F.S), Tensor(F.Vt))
end
