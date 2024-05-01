"""
    adj(::AbstractSecondOrderTensor)
    adj(::AbstractSymmetricSecondOrderTensor)

Compute the adjugate matrix.

# Examples
```jldoctest
julia> x = rand(Mat{3,3});

julia> Tensorial.adj(x) / det(x) ≈ inv(x)
true
```
"""
function adj end

@inline adj(x::AbstractSquareTensor{1}) = one(x)
@generated function adj(x::AbstractSquareTensor{2})
    x_11 = getindex_expr(x, :x, 1, 1)
    x_21 = getindex_expr(x, :x, 2, 1)
    x_12 = getindex_expr(x, :x, 1, 2)
    x_22 = getindex_expr(x, :x, 2, 2)
    exps = [x_22, :(-$x_21), :(-$x_12), x_11]
    quote
        @_inline_meta
        @inbounds typeof(x)(tuple($(exps[indices_unique(x)]...)))
    end
end
@generated function adj(x::AbstractSquareTensor{3})
    x_11 = getindex_expr(x, :x, 1, 1)
    x_21 = getindex_expr(x, :x, 2, 1)
    x_31 = getindex_expr(x, :x, 3, 1)
    x_12 = getindex_expr(x, :x, 1, 2)
    x_22 = getindex_expr(x, :x, 2, 2)
    x_32 = getindex_expr(x, :x, 3, 2)
    x_13 = getindex_expr(x, :x, 1, 3)
    x_23 = getindex_expr(x, :x, 2, 3)
    x_33 = getindex_expr(x, :x, 3, 3)
    exps = [:( ($x_22*$x_33 - $x_23*$x_32)),
            :(-($x_21*$x_33 - $x_23*$x_31)),
            :( ($x_21*$x_32 - $x_22*$x_31)),
            :(-($x_12*$x_33 - $x_13*$x_32)),
            :( ($x_11*$x_33 - $x_13*$x_31)),
            :(-($x_11*$x_32 - $x_12*$x_31)),
            :( ($x_12*$x_23 - $x_13*$x_22)),
            :(-($x_11*$x_23 - $x_13*$x_21)),
            :( ($x_11*$x_22 - $x_12*$x_21))]
    quote
        @_inline_meta
        @inbounds typeof(x)(tuple($(exps[indices_unique(x)]...)))
    end
end
@generated function adj(x::AbstractSquareTensor)
    exps = map(CartesianIndices(x)) do I
        j, i = Tuple(I)
        :($((-1)^(i+j)) * det(adj_ij(x, Val($i), Val($j))))
    end
    quote
        @_inline_meta
        @inbounds typeof(x)(tuple($(exps[indices_unique(x)]...)))
    end
end

@generated function adj_ij(x::AbstractSquareTensor, ::Val{i}, ::Val{j}) where {i, j}
    quote
        @_inline_meta
        vcat(
            hcat((@Tensor x[1:i-1,   1:j-1]), (@Tensor x[1:i-1,   j+1:end])),
            hcat((@Tensor x[i+1:end, 1:j-1]), (@Tensor x[i+1:end, j+1:end])),
        )
    end
end

"""
    inv(::AbstractSecondOrderTensor)
    inv(::AbstractSymmetricSecondOrderTensor)
    inv(::AbstractFourthOrderTensor)
    inv(::AbstractSymmetricFourthOrderTensor)

Compute the inverse of a tensor.

# Examples
```jldoctest
julia> x = rand(SecondOrderTensor{3})
3×3 Tensor{Tuple{3, 3}, Float64, 2, 9}:
 0.325977  0.894245  0.953125
 0.549051  0.353112  0.795547
 0.218587  0.394255  0.49425

julia> inv(x)
3×3 Tensor{Tuple{3, 3}, Float64, 2, 9}:
 -587.685  -279.668   1583.46
 -411.743  -199.494   1115.12
  588.35    282.819  -1587.79

julia> x ⋅ inv(x) ≈ one(I)
true
```
"""
function inv end

@inline _inv(x::AbstractSquareTensor{1}) = adj(x) * inv(det(x))
@inline _inv(x::AbstractSquareTensor{2}) = adj(x) * inv(det(x))
@inline _inv(x::AbstractSquareTensor{3}) = adj(x) * inv(det(x))

@inline function _inv(x::AbstractSecondOrderTensor{dim}) where {dim}
    typeof(x)(inv(SMatrix{dim, dim}(x)))
end
@inline function _inv(x::AbstractSymmetricSecondOrderTensor{dim}) where {dim}
    # `InexactError` occurs without `symmetric`
    sa = inv(SMatrix{dim, dim}(x))
    typeof(x)(symmetric(Tensor(sa), :U))
end

@generated function toblocks(x::Mat{dim, dim}) where {dim}
    m = dim ÷ 2
    n = dim - m
    inds = indices_all(x)
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
            push!(exps, getindex_expr(A, :A, i, j))
        end
        for i in 1:n
            push!(exps, getindex_expr(C, :C, i, j))
        end
    end
    for j in 1:n
        for i in 1:m
            push!(exps, getindex_expr(B, :B, i, j))
        end
        for i in 1:n
            push!(exps, getindex_expr(D, :D, i, j))
        end
    end
    quote
        @_inline_meta
        @inbounds Mat{$(m+n), $(m+n)}($(exps...))
    end
end

# https://en.wikipedia.org/wiki/Block_matrix#Block_matrix_inversion
# https://core.ac.uk/download/pdf/193046446.pdf
# https://math.stackexchange.com/questions/411492/inverse-of-a-block-matrix-with-singular-diagonal-blocks
function _inv_with_blocks(x::Mat{dim, dim}) where {dim}
    xᵀ = x
    M = xᵀ ⋅ x

    A, B,
    C, D = toblocks(M)

    A⁻¹ = inv(A)
    A⁻¹B = A⁻¹ ⋅ B
    E = inv(D - C ⋅ A⁻¹B)
    A⁻¹BE = A⁻¹B ⋅ E
    CA⁻¹ = C ⋅ A⁻¹

    X = A⁻¹ + A⁻¹BE ⋅ CA⁻¹
    Y = -A⁻¹BE
    Z = -E ⋅ CA⁻¹
    W = E

    fromblocks(X, Y,
               Z, W) ⋅ xᵀ
end

@inline function _inv_with_blocks(x::Tensor{Tuple{@Symmetry({dim, dim})}}) where {dim}
    # `InexactError` occurs without `symmetric`
    typeof(x)(symmetric(_inv_with_blocks(convert(Mat{dim, dim}, x)), :U))
end

@inline inv(x::AbstractSquareTensor) = _inv(x)

# fast inv for dim ≤ 10
@inline fastinv(x::AbstractSquareTensor{1}) = _inv(x)
@inline fastinv(x::AbstractSquareTensor{2}) = _inv(x)
@inline fastinv(x::AbstractSquareTensor{3}) = _inv(x)
@inline fastinv(x::AbstractSquareTensor{4}) = _inv(x)
@inline fastinv(x::AbstractSquareTensor{5}) = _inv_with_blocks(x)
@inline fastinv(x::AbstractSquareTensor{6}) = _inv_with_blocks(x)
@inline fastinv(x::AbstractSquareTensor{7}) = _inv_with_blocks(x)
@inline fastinv(x::AbstractSquareTensor{8}) = _inv_with_blocks(x)
@inline fastinv(x::AbstractSquareTensor{9}) = _inv_with_blocks(x)
@inline fastinv(x::AbstractSquareTensor{10}) = _inv_with_blocks(x)
@inline fastinv(x::AbstractSquareTensor{<: Any}) = _inv(x)

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

@inline function _solve(A::AbstractSquareTensor, b::AbstractVec)
    SA = SArray(A)
    Sb = SArray(b)
    Vec(Tuple(SA \ Sb))
end

@inline Base.:\(A::AbstractSquareTensor, b::AbstractVec) = _solve(A, b)

# fast solve for dim ≤ 10
@inline fastsolve(A::AbstractSquareTensor{1, Float64}, b::AbstractVec{1, Float64}) = inv(A) ⋅ b
@inline fastsolve(A::AbstractSquareTensor{2, Float64}, b::AbstractVec{2, Float64}) = inv(A) ⋅ b
@inline fastsolve(A::AbstractSquareTensor{3, Float64}, b::AbstractVec{3, Float64}) = inv(A) ⋅ b
@inline fastsolve(A::AbstractSquareTensor{4, Float64}, b::AbstractVec{4, Float64}) = inv(A) ⋅ b
@inline fastsolve(A::AbstractSquareTensor{5, Float64}, b::AbstractVec{5, Float64}) = inv(A) ⋅ b
@inline fastsolve(A::AbstractSquareTensor{6, Float64}, b::AbstractVec{6, Float64}) = inv(A) ⋅ b
@inline fastsolve(A::AbstractSquareTensor{7, Float64}, b::AbstractVec{7, Float64}) = inv(A) ⋅ b
@inline fastsolve(A::AbstractSquareTensor{8, Float64}, b::AbstractVec{8, Float64}) = inv(A) ⋅ b
@inline fastsolve(A::AbstractSquareTensor{9, Float64}, b::AbstractVec{9, Float64}) = inv(A) ⋅ b
@inline fastsolve(A::AbstractSquareTensor{10, Float64}, b::AbstractVec{10, Float64}) = inv(A) ⋅ b
@inline fastsolve(A::AbstractSquareTensor{dim, Float64}, b::AbstractVec{dim, Float64}) where {dim} = inv(A) ⋅ b
