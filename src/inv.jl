"""
    inv(::AbstractSecondOrderTensor)
    inv(::AbstractSymmetricSecondOrderTensor)
    inv(::AbstractFourthOrderTensor)
    inv(::AbstractSymmetricFourthOrderTensor)

Compute the inverse of a tensor.

# Examples
```jldoctest
julia> x = rand(SecondOrderTensor{3})
3×3 Tensor{Tuple{3,3},Float64,2,9}:
 0.590845  0.460085  0.200586
 0.766797  0.794026  0.298614
 0.566237  0.854147  0.246837

julia> inv(x)
3×3 Tensor{Tuple{3,3},Float64,2,9}:
  19.7146   -19.2802    7.30384
   6.73809  -10.7687    7.55198
 -68.541     81.4917  -38.8361

julia> x ⋅ inv(x) ≈ one(I)
true
```
"""
function inv end

@inline function _inv(x::AbstractSquareTensor{1})
    typeof(x)(1/det(x))
end

@generated function _inv(x::AbstractSquareTensor{2})
    x_11 = getindex_expr(:x, x, 1, 1)
    x_21 = getindex_expr(:x, x, 2, 1)
    x_12 = getindex_expr(:x, x, 1, 2)
    x_22 = getindex_expr(:x, x, 2, 2)
    exps = [x_22, :(-$x_21), :(-$x_12), x_11]
    quote
        @_inline_meta
        @inbounds typeof(x)($(exps[indices(x)]...)) / det(x)
    end
end

@generated function _inv(x::AbstractSquareTensor{3})
    x_11 = getindex_expr(:x, x, 1, 1)
    x_21 = getindex_expr(:x, x, 2, 1)
    x_31 = getindex_expr(:x, x, 3, 1)
    x_12 = getindex_expr(:x, x, 1, 2)
    x_22 = getindex_expr(:x, x, 2, 2)
    x_32 = getindex_expr(:x, x, 3, 2)
    x_13 = getindex_expr(:x, x, 1, 3)
    x_23 = getindex_expr(:x, x, 2, 3)
    x_33 = getindex_expr(:x, x, 3, 3)
    exps = [:(($x_22*$x_33 - $x_23*$x_32)),
            :(($x_23*$x_31 - $x_21*$x_33)),
            :(($x_21*$x_32 - $x_22*$x_31)),
            :(($x_13*$x_32 - $x_12*$x_33)),
            :(($x_11*$x_33 - $x_13*$x_31)),
            :(($x_12*$x_31 - $x_11*$x_32)),
            :(($x_12*$x_23 - $x_13*$x_22)),
            :(($x_13*$x_21 - $x_11*$x_23)),
            :(($x_11*$x_22 - $x_12*$x_21))]
    quote
        @_inline_meta
        @inbounds typeof(x)($(exps[indices(x)]...)) / det(x)
    end
end

@inline function _inv(x::AbstractSquareTensor{dim}) where {dim}
    typeof(x)(inv(SMatrix{dim, dim}(x)))
end

@generated function toblocks(x::Mat{dim, dim}) where {dim}
    m = dim ÷ 2
    n = dim - m
    inds = independent_indices(x)
    a = [:(Tuple(x)[$(inds[I])]) for I in CartesianIndices((1:m, 1:m))]
    b = [:(Tuple(x)[$(inds[I])]) for I in CartesianIndices((1:m, m+1:dim))]
    c = [:(Tuple(x)[$(inds[I])]) for I in CartesianIndices((m+1:dim, 1:m))]
    d = [:(Tuple(x)[$(inds[I])]) for I in CartesianIndices((m+1:dim, m+1:dim))]
    quote
        @_inline_meta
        A = Mat{$m,$m}($(a...))
        B = Mat{$m,$n}($(b...))
        C = Mat{$n,$m}($(c...))
        D = Mat{$n,$n}($(d...))
        A, B, C, D
    end
end

@generated function fromblocks(A::Mat{m,m}, B::Mat{m,n}, C::Mat{n,m}, D::Mat{n,n}) where {m, n}
    exps = Expr[]
    for j in 1:m
        for i in 1:m
            push!(exps, getindex_expr(:A, A, i, j))
        end
        for i in 1:n
            push!(exps, getindex_expr(:C, C, i, j))
        end
    end
    for j in 1:n
        for i in 1:m
            push!(exps, getindex_expr(:B, B, i, j))
        end
        for i in 1:n
            push!(exps, getindex_expr(:D, D, i, j))
        end
    end
    quote
        @_inline_meta
        @inbounds Mat{$(m+n), $(m+n)}($(exps...))
    end
end

# https://en.wikipedia.org/wiki/Block_matrix#Block_matrix_inversion
function _inv_with_blocks(x::Mat{dim, dim}) where {dim}
    A, B, C, D = toblocks(x)

    A⁻¹ = inv(A)
    A⁻¹B = A⁻¹ ⋅ B
    E = inv(D - C ⋅ A⁻¹B)
    A⁻¹BE = A⁻¹B ⋅ E
    CA⁻¹ = C ⋅ A⁻¹

    X = A⁻¹ + A⁻¹BE ⋅ CA⁻¹
    Y = -A⁻¹BE
    Z = -E ⋅ CA⁻¹
    W = E

    fromblocks(X, Y, Z, W)
end

@inline function _inv_with_blocks(x::Tensor{Tuple{@Symmetry{dim, dim}}}) where {dim}
    typeof(x)(_inv_with_blocks(convert(Mat{dim, dim}, x)))
end

# use faster inv for dim ≤ 10
@inline inv(x::AbstractSquareTensor{4}) = _inv_with_blocks(x)
@inline inv(x::AbstractSquareTensor{5}) = _inv_with_blocks(x)
@inline inv(x::AbstractSquareTensor{6}) = _inv_with_blocks(x)
@inline inv(x::AbstractSquareTensor{7}) = _inv_with_blocks(x)
@inline inv(x::AbstractSquareTensor{8}) = _inv_with_blocks(x)
@inline inv(x::AbstractSquareTensor{9}) = _inv_with_blocks(x)
@inline inv(x::AbstractSquareTensor{10}) = _inv_with_blocks(x)
@inline inv(x::AbstractSquareTensor) = _inv(x)

# don't use `voigt` or `mandel` for fast computations
@generated function inv(x::FourthOrderTensor{dim}) where {dim}
    L = dim * dim
    quote
        @_inline_meta
        M = Mat{$L, $L}(Tuple(x))
        FourthOrderTensor{dim}(Tuple(inv(M)))
    end
end

@generated function inv(x::SymmetricFourthOrderTensor{dim, T}) where {dim, T}
    S = Space(Symmetry(dim, dim))
    L = ncomponents(S)
    c = Vec{L, T}([i == j ? 1 : √2 for j in 1:dim for i in j:dim])
    coef = c ⊗ c
    quote
        @_inline_meta
        M = _map(*, Mat{$L, $L}(Tuple(x)), $coef)
        M⁻¹ = inv(M)
        SymmetricFourthOrderTensor{dim}(Tuple(_map(/, M⁻¹, $coef)))
    end
end

@generated function _solve(A::AbstractSquareTensor{dim}, b::AbstractVec{dim}) where {dim}
    exps_A = [:(Tuple(A)[$i]) for i in independent_indices(A)]
    exps_b = [:(Tuple(b)[$i]) for i in independent_indices(b)]
    quote
        @_inline_meta
        @inbounds begin
            SA = SMatrix{dim, dim}($(exps_A...))
            Sb = SVector{dim}($(exps_b...))
        end
        Vec{dim}(Tuple(SA \ Sb))
    end
end

# use faster inv for dim ≤ 10
@inline Base.:\(A::AbstractSquareTensor{1}, b::AbstractVec{1}) = inv(A) ⋅ b
@inline Base.:\(A::AbstractSquareTensor{2}, b::AbstractVec{2}) = inv(A) ⋅ b
@inline Base.:\(A::AbstractSquareTensor{3}, b::AbstractVec{3}) = inv(A) ⋅ b
@inline Base.:\(A::AbstractSquareTensor{4}, b::AbstractVec{4}) = inv(A) ⋅ b
@inline Base.:\(A::AbstractSquareTensor{5}, b::AbstractVec{5}) = inv(A) ⋅ b
@inline Base.:\(A::AbstractSquareTensor{6}, b::AbstractVec{6}) = inv(A) ⋅ b
@inline Base.:\(A::AbstractSquareTensor{7}, b::AbstractVec{7}) = inv(A) ⋅ b
@inline Base.:\(A::AbstractSquareTensor{8}, b::AbstractVec{8}) = inv(A) ⋅ b
@inline Base.:\(A::AbstractSquareTensor{9}, b::AbstractVec{9}) = inv(A) ⋅ b
@inline Base.:\(A::AbstractSquareTensor{10}, b::AbstractVec{10}) = inv(A) ⋅ b
@inline Base.:\(A::AbstractSquareTensor, b::AbstractVec) = _solve(A, b)
