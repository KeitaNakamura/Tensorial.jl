@generated function _map(f, xs::Vararg{Tensor, N}) where {N}
    inds = promote_indices(map(TensorIndices, xs)...)
    exps = map(uniqueindices(inds)) do i
        vals = [:(xs[$j][$i]) for j in 1:N]
        :(f($(vals...)))
    end
    T = promote_type(map(eltype, xs)...)
    TT = tensortype(inds){T}
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

function contraction_exprs(::Type{S1}, ::Type{S2}, ::Val{N}) where {S1, S2, N}
    t = contraction(TensorIndices(S1), TensorIndices(S2), Val(N))
    s1 = map(i -> EinsumIndex(:(Tuple(x)), i), serialindices(S1))
    s2 = map(i -> EinsumIndex(:(Tuple(y)), i), serialindices(S2))
    J = prod(size(s2)[1:N])
    I = length(s1) ÷ J
    K = length(s2) ÷ J
    s1′ = reshape(s1, I, J)
    s2′ = reshape(s2, J, K)
    s′ = Array{EinsumIndexSum{2}}(undef, I, K)
    for k in 1:K, i in 1:I
        s′[i,k] = sum(s1′[i,j] * s2′[j,k] for j in 1:J)
    end
    s = reshape(s′, size(s1)[1:end-N]..., size(s2)[N+1:end]...)
    map(construct_expr, s[uniqueindices(t)])
end
contraction_exprs(::Type{<: Tensor{S1}}, ::Type{<: Tensor{S2}}, ::Val{N}) where {S1, S2, N} = contraction_exprs(S1, S2, Val(N))

@generated function contraction(x::Tensor, y::Tensor, ::Val{N}) where {N}
    t = contraction(TensorIndices(x), TensorIndices(y), Val(N))
    exps = contraction_exprs(x, y, Val(N))
    T = promote_type(eltype(x), eltype(y))
    if ndims(t) == 0
        TT = T
    else
        TT = tensortype(t){T}
    end
    quote
        @_inline_meta
        @inbounds $TT($(exps...))
    end
end

@inline otimes(x1::Tensor, x2::Tensor) = contraction(x1, x2, Val(0))
@inline dot(x1::Tensor, x2::Tensor) = contraction(x1, x2, Val(1))
@inline dcontraction(x1::Tensor, x2::Tensor) = contraction(x1, x2, Val(2))
@inline norm(x::Tensor) = sqrt(contraction(x, x, Val(ndims(x))))

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
        @inbounds typeof(x)($(exps[uniqueindices(x)]...))
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
        @inbounds typeof(x)($(exps[uniqueindices(x)]...))
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
    inds = TensorIndices(x)
    exps = contraction_exprs(x, y, Val(1))
    quote
        @_inline_meta
        @inbounds SymmetricSecondOrderTensor{dim}($(exps[uniqueindices(inds)]...))
    end
end
