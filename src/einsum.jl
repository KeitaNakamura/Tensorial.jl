struct EinsumIndex
    name::Union{Symbol, Expr}
    index::Int
end

Base.:*(x::EinsumIndex, y::EinsumIndex, z::EinsumIndex...) = EinsumIndexMul(1, (x, y, z...))

struct EinsumIndexMul{N}
    ndups::Int
    indices::NTuple{N, EinsumIndex}
    function EinsumIndexMul{N}(ndups::Int, indices::NTuple{N, EinsumIndex}) where {N}
        new{N}(ndups, tuple_sort(indices; by = hash))
    end
end
EinsumIndexMul(ndups::Int, indices::NTuple{N, EinsumIndex}) where {N} = EinsumIndexMul{N}(ndups, indices)

Base.:(==)(x::EinsumIndexMul, y::EinsumIndexMul) = x.indices == y.indices
Base.:*(x::EinsumIndex, y::EinsumIndexMul) = (@assert y.ndups == 1; EinsumIndexMul(1, (x, y.indices...)))
Base.:*(x::EinsumIndexMul, y::EinsumIndex) = y * x

add(x::EinsumIndexMul, y::EinsumIndexMul) = (@assert x == y; EinsumIndexMul(x.ndups + y.ndups, x.indices))

struct EinsumIndexSum{N} <: AbstractVector{EinsumIndexMul{N}}
    terms::Vector{EinsumIndexMul{N}}
end

Base.convert(::Type{EinsumIndexSum{N}}, x::EinsumIndexMul{N}) where {N} = EinsumIndexSum([x])

Base.size(x::EinsumIndexSum) = size(x.terms)
Base.getindex(x::EinsumIndexSum, i::Int) = (@_propagate_inbounds_meta; x.terms[i])

function Base.:+(x::EinsumIndexMul{N}, y::EinsumIndexMul{N}) where {N}
    x == y ? EinsumIndexSum([add(x, y)]) : EinsumIndexSum([x, y])
end
function Base.:+(x::EinsumIndexSum{N}, y::EinsumIndexMul{N}) where {N}
    terms = copy(x.terms)
    i = findfirst(==(y), terms)
    i === nothing && return EinsumIndexSum(push!(terms, y))
    terms[i] = add(terms[i], y)
    EinsumIndexSum(terms)
end
Base.:+(x::EinsumIndexMul, y::EinsumIndexSum) = y + x


construct_expr(x::EinsumIndex) = :($(x.name)[$(x.index)])

function construct_expr(x::EinsumIndexMul)
    if x.ndups == 1
        Expr(:call, :*, map(construct_expr, x.indices)...)
    else
        Expr(:call, :*, x.ndups, map(construct_expr, x.indices)...)
    end
end

function construct_expr(x::EinsumIndexSum)
    Expr(:call, :+, map(construct_expr, x)...)
end
