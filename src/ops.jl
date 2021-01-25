@generated function _map(f, xs::Vararg{Tensor, N}) where {N}
    S = promote_size(map(Size, xs)...)
    exps = map(indices(S)) do i
        vals = [:(xs[$j][$i]) for j in 1:N]
        :(f($(vals...)))
    end
    TT = tensortype(S)
    return quote
        @_inline_meta
        @inbounds $TT($(exps...))
    end
end

@inline Base.:+(x::Tensor, y::Tensor) = _map(+, x, y)
@inline Base.:-(x::Tensor, y::Tensor) = _map(-, x, y)
@inline Base.:*(y::Real, x::Tensor) = _map(x -> x*y, x)
@inline Base.:*(x::Tensor, y::Real) = _map(x -> x*y, x)
@inline Base.:/(x::Tensor, y::Real) = _map(x -> x/y, x)
@inline Base.:+(x::Tensor) = x
@inline Base.:-(x::Tensor) = _map(-, x)

# error for standard multiplications
function Base.:*(::Tensor, ::Tensor)
    error("use `⋅` (`\\cdot`) for single contraction and `⊡` (`\\boxdot`) for double contraction instead of `*`")
end

function contraction_exprs(S1::Size, S2::Size, ::Val{N}) where {N}
    S = contraction(S1, S2, Val(N))
    s1 = map(i -> EinsumIndex(:(Tuple(x)), i), independent_indices(S1))
    s2 = map(i -> EinsumIndex(:(Tuple(y)), i), independent_indices(S2))
    J = prod(size(s2)[1:N])
    I = length(s1) ÷ J
    K = length(s2) ÷ J
    s1′ = reshape(s1, I, J)
    s2′ = reshape(s2, J, K)
    s′ = @einsum (i,k) -> s1′[i,j] * s2′[j,k]
    s = reshape(s′, size(s1)[1:end-N]..., size(s2)[N+1:end]...)
    map(construct_expr, s[indices(S)])
end

"""
    contraction(::Tensor, ::Tensor, ::Val{N})

Conduct contraction of `N` inner indices.
For example, `N=2` contraction with third-order tensors ``B_{ijk}`` and ``C_{ijk}`` becomes

``
A_{ij} = B_{ikl} C_{klj}
``

In Tensorial.jl, above computation is the same as

```jldoctest
julia> B = rand(Tensor{Tuple{3,3,3}});

julia> C = rand(Tensor{Tuple{3,3,3}});

julia> A = contraction(B, C, Val(2))
3×3 Tensor{Tuple{3,3},Float64,2,9}:
 1.36912   1.86751  1.32531
 1.61744   2.34426  1.94101
 0.929252  1.89656  1.79015
```

Following aliases are also available:

- `⊗` (typed by `\\otimes<tab>`): `contraction(::Tensor, ::Tensor, Val(0))`
- `⋅` (typed by `\\cdot<tab>`): `contraction(::Tensor, ::Tensor, Val(1))`
- `⊡` (typed by `\\boxdot<tab>`): `contraction(::Tensor, ::Tensor, Val(2))`
"""
@generated function contraction(x::Tensor, y::Tensor, ::Val{N}) where {N}
    S = contraction(Size(x), Size(y), Val(N))
    exps = contraction_exprs(Size(x), Size(y), Val(N))
    T = promote_type(eltype(x), eltype(y))
    if length(S) == 0
        TT = T
    else
        TT = tensortype(S){T}
    end
    quote
        @_inline_meta
        @inbounds $TT($(exps...))
    end
end

"""
    otimes(x::Tensor, y::Tensor)
    x ⊗ y

Compute tensor product such as ``A_{ij} = x_i y_j``.
`x ⊗ y` (where `⊗` can be typed by `\\otimes<tab>`) is a synonym for `otimes(x, y)`.

# Examples
```jldoctest
julia> x = rand(Vec{3});

julia> y = rand(Vec{3});

julia> A = x ⊗ y
3×3 Tensor{Tuple{3,3},Float64,2,9}:
 0.271839  0.469146  0.504668
 0.352792  0.608857  0.654957
 0.260518  0.449607  0.48365
```
"""
@inline otimes(x1::Tensor, x2::Tensor) = contraction(x1, x2, Val(0))

"""
    dot(x::Tensor, y::Tensor)
    x ⋅ y

Compute dot product such as ``a = x_i y_i``.
This is equivalent to [`contraction(::Tensor, ::Tensor, Val(1))`](@ref).
`x ⋅ y` (where `⋅` can be typed by `\\cdot<tab>`) is a synonym for `cdot(x, y)`.

# Examples
```jldoctest
julia> x = rand(Vec{3});

julia> y = rand(Vec{3});

julia> a = x ⋅ y
1.3643452781654775
```
"""
@inline dot(x1::Tensor, x2::Tensor) = contraction(x1, x2, Val(1))
@inline dcontraction(x1::Tensor, x2::Tensor) = contraction(x1, x2, Val(2))
@inline norm(x::Tensor) = sqrt(contraction(x, x, Val(ndims(x))))

# v_k * S_ikjl * u_l
@generated function dotdot(v1::Vec{dim}, S::SymmetricFourthOrderTensor{dim}, v2::Vec{dim}) where {dim}
    v1inds = map(i -> EinsumIndex(:(Tuple(v1)), i), independent_indices(v1))
    v2inds = map(i -> EinsumIndex(:(Tuple(v2)), i), independent_indices(v2))
    Sinds = map(i -> EinsumIndex(:(Tuple(S)), i), independent_indices(S))
    exps = @einsum (i,j) -> v1inds[k] * Sinds[i,k,j,l] * v2inds[l]
    TT = Tensor{Tuple{dim, dim}}
    quote
        @_inline_meta
        @inbounds $TT($(map(construct_expr, exps)...))
    end
end

const SquareTensor{dim, T} = Union{SecondOrderTensor{dim, T}, SymmetricSecondOrderTensor{dim, T}}

# tr/mean
@inline function tr(x::SquareTensor{dim}) where {dim}
    sum(i -> @inbounds(x[i,i]), 1:dim)
end
@inline tr(x::SquareTensor{1}) = @inbounds x[1,1]
@inline tr(x::SquareTensor{2}) = @inbounds x[1,1] + x[2,2]
@inline tr(x::SquareTensor{3}) = @inbounds x[1,1] + x[2,2] + x[3,3]
@inline mean(x::SquareTensor{dim}) where {dim} = tr(x) / dim

# vol/dev
@inline function vol(x::SquareTensor{3})
    v = mean(x)
    z = zero(v)
    typeof(x)((i,j) -> i == j ? v : z)
end
@inline dev(x::SquareTensor{3}) = x - vol(x)

# transpose/adjoint
@inline transpose(x::SymmetricSecondOrderTensor) = x
@inline transpose(x::Tensor{Tuple{m, n}}) where {m, n} = Tensor{Tuple{n, m}}((i,j) -> @inbounds x[j,i])
@inline adjoint(x::Tensor) = transpose(x)

# symmetric
@inline symmetric(x::SymmetricSecondOrderTensor) = x
@inline symmetric(x::SecondOrderTensor{dim}) where {dim} =
    SymmetricSecondOrderTensor{dim}((i,j) -> @inbounds i == j ? x[i,j] : (x[i,j] + x[j,i]) / 2)
@inline symmetric(x::SecondOrderTensor{1}) =
    @inbounds SymmetricSecondOrderTensor{1}(x[1,1])
@inline symmetric(x::SecondOrderTensor{2}) =
    @inbounds SymmetricSecondOrderTensor{2}(x[1,1], (x[2,1]+x[1,2])/2, x[2,2])
@inline symmetric(x::SecondOrderTensor{3}) =
    @inbounds SymmetricSecondOrderTensor{3}(x[1,1], (x[2,1]+x[1,2])/2, (x[3,1]+x[1,3])/2, x[2,2], (x[3,2]+x[2,3])/2, x[3,3])

# det
@inline function det(x::SquareTensor{1})
    @inbounds x[1,1]
end
@inline function det(x::SquareTensor{2})
    @inbounds x[1,1] * x[2,2] - x[1,2] * x[2,1]
end
@inline function det(x::SquareTensor{3})
    @inbounds (x[1,1] * (x[2,2]*x[3,3] - x[2,3]*x[3,2]) -
               x[1,2] * (x[2,1]*x[3,3] - x[2,3]*x[3,1]) +
               x[1,3] * (x[2,1]*x[3,2] - x[2,2]*x[3,1]))
end
@inline function det(x::SquareTensor)
    det(Matrix(x))
end

# inv
@inline function inv(x::SquareTensor{1})
    typeof(x)(1/det(x))
end
@generated function inv(x::SquareTensor{2})
    x_11 = getindex_expr(:x, x, 1, 1)
    x_21 = getindex_expr(:x, x, 2, 1)
    x_12 = getindex_expr(:x, x, 1, 2)
    x_22 = getindex_expr(:x, x, 2, 2)
    exps = [:($x_22 * detinv), :(-$x_21 * detinv), :(-$x_12 * detinv), :($x_11 * detinv)]
    return quote
        @_inline_meta
        detinv = 1 / det(x)
        @inbounds typeof(x)($(exps[indices(x)]...))
    end
end
@generated function inv(x::SquareTensor{3})
    x_11 = getindex_expr(:x, x, 1, 1)
    x_21 = getindex_expr(:x, x, 2, 1)
    x_31 = getindex_expr(:x, x, 3, 1)
    x_12 = getindex_expr(:x, x, 1, 2)
    x_22 = getindex_expr(:x, x, 2, 2)
    x_32 = getindex_expr(:x, x, 3, 2)
    x_13 = getindex_expr(:x, x, 1, 3)
    x_23 = getindex_expr(:x, x, 2, 3)
    x_33 = getindex_expr(:x, x, 3, 3)
    exps = [:(($x_22*$x_33 - $x_23*$x_32) * detinv),
            :(($x_23*$x_31 - $x_21*$x_33) * detinv),
            :(($x_21*$x_32 - $x_22*$x_31) * detinv),
            :(($x_13*$x_32 - $x_12*$x_33) * detinv),
            :(($x_11*$x_33 - $x_13*$x_31) * detinv),
            :(($x_12*$x_31 - $x_11*$x_32) * detinv),
            :(($x_12*$x_23 - $x_13*$x_22) * detinv),
            :(($x_13*$x_21 - $x_11*$x_23) * detinv),
            :(($x_11*$x_22 - $x_12*$x_21) * detinv)]
    return quote
        @_inline_meta
        detinv = 1 / det(x)
        @inbounds typeof(x)($(exps[indices(x)]...))
    end
end
@inline function inv(x::SquareTensor)
    typeof(x)(inv(Matrix(x)))
end
@inline function inv(x::FourthOrderTensor{dim}) where {dim}
    @assert dim < 4
    fromvoigt(FourthOrderTensor{dim}, inv(tovoigt(x)))
end
@inline function inv(x::SymmetricFourthOrderTensor{dim}) where {dim}
    @assert dim < 4
    frommandel(SymmetricFourthOrderTensor{dim}, inv(tomandel(x)))
end

# cross
@inline cross(x::Vec{1}, y::Vec{1}) =
    zero(Vec{3, promote_type(eltype(x), eltype(y))})
@inline cross(x::Vec{2, T1}, y::Vec{2, T2}) where {T1, T2} =
    @inbounds Vec{3}((zero(T1)*zero(T2), zero(T1)*zero(T2), x[1]*y[2] - x[2]*y[1]))
@inline cross(x::Vec{3}, y::Vec{3}) =
    @inbounds Vec{3}((x[2]*y[3] - x[3]*y[2], x[3]*y[1] - x[1]*y[3], x[1]*y[2] - x[2]*y[1]))

# power
@inline Base.literal_pow(::typeof(^), x::SquareTensor, ::Val{-1}) = inv(x)
@inline Base.literal_pow(::typeof(^), x::SquareTensor, ::Val{0})  = one(x)
@inline Base.literal_pow(::typeof(^), x::SquareTensor, ::Val{1})  = x
@inline function Base.literal_pow(::typeof(^), x::SquareTensor, ::Val{p}) where {p}
    p > 0 ? (y = x; q = p) : (y = inv(x); q = -p)
    z = y
    for i in 2:q
        y = _powdot(y, z)
    end
    y
end
## helper functions
@inline _powdot(x::SecondOrderTensor, y::SecondOrderTensor) = dot(x, y)
@generated function _powdot(x::SymmetricSecondOrderTensor{dim}, y::SymmetricSecondOrderTensor{dim}) where {dim}
    S = Size(x)
    exps = contraction_exprs(Size(x), Size(y), Val(1))
    quote
        @_inline_meta
        @inbounds SymmetricSecondOrderTensor{dim}($(exps[indices(S)]...))
    end
end
