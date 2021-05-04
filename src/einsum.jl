# convert to `name[index]`
struct EinsumIndex
    name::Union{Symbol, Expr}
    index::Int
end

Base.:*(x::EinsumIndex, y::EinsumIndex, z::EinsumIndex...) = EinsumIndexMul(1, (x, y, z...))
Base.:(==)(x::EinsumIndex, y::EinsumIndex) = (x.name == y.name) && (x.index == y.index)

struct EinsumIndexMul{N}
    ndups::Int
    indices::NTuple{N, EinsumIndex}
    function EinsumIndexMul{N}(ndups::Int, indices::NTuple{N, EinsumIndex}) where {N}
        new{N}(ndups, tuple_sort(indices; by = hash))
    end
end
EinsumIndexMul(ndups::Int, indices::NTuple{N, EinsumIndex}) where {N} = EinsumIndexMul{N}(ndups, indices)

function Base.:(==)(x::EinsumIndexMul{N}, y::EinsumIndexMul{N}) where {N}
    # all(i -> x.indices[i] == y.indices[i], 1:N)
    false # considering number of duplications is slow
end
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
    quote
        v = tuple($(map(construct_expr, x)...))
        out = zero(eltype(v))
        @simd for i in eachindex(v)
            @inbounds out += v[i]
        end
        out
    end
end


# NOTE: only multiplication is supported
macro einsum_array(ex)
    freeinds, code = anonymous_args_body(ex)
    allinds = Dict{Symbol, Expr}()
    dummyinds = Set{Symbol}()
    _findindices!(allinds, dummyinds, ex)
    @assert Set(keys(allinds)) == union(Set(freeinds), (dummyinds))
    if !isempty(dummyinds)
        code = Expr(:call, :sum, Expr(:generator, code, [allinds[i] for i in dummyinds]...))
    end
    if !isempty(freeinds)
        code = Expr(:comprehension, Expr(:generator, code, [allinds[i] for i in freeinds]...))
    end
    esc(code)
end

function anonymous_args_body(func::Expr)
    @assert Meta.isexpr(func, :->)
    lhs = func.args[1]
    body = func.args[2]
    if Meta.isexpr(lhs, :tuple)
        args = lhs.args
    elseif fargs isa Symbol
        args = [lhs]
    else
        throw(ArgumentError("wrong arguments in anonymous function expression"))
    end
    args, body
end

_findindices!(allinds::Dict{Symbol, Expr}, dummyinds::Set{Symbol}, ::Any) = nothing
function _findindices!(allinds::Dict{Symbol, Expr}, dummyinds::Set{Symbol}, expr::Expr)
    if Meta.isexpr(expr, :ref)
        name = expr.args[1] # name of array `name[index]`
        for (j, index) in enumerate(expr.args[2:end])
            isa(index, Symbol) || throw(ArgumentError("@einsum: index must be symbol"))
            if haskey(allinds, index) # `index` is already in `allinds`
                push!(dummyinds, index)
            else
                allinds[index] = :($index = axes($name, $j))
            end
        end
    else
        for ex in expr.args
            _findindices!(allinds, dummyinds, ex)
        end
    end
    nothing
end
