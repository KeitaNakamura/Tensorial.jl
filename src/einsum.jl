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
    if Meta.isexpr(func, :->)
        lhs = func.args[1]
        body = func.args[2]
        if Meta.isexpr(lhs, :tuple)
            freeinds = lhs.args
        elseif lhs isa Symbol
            freeinds = [lhs]
        else
            throw(ArgumentError("wrong arguments in anonymous function expression"))
        end
        if Meta.isexpr(body, :block)
            if body.args[1] isa LineNumberNode
                @assert length(body.args) == 2
                body = body.args[2]
            else
                @assert length(body.args) == 1
                body = body.args[1]
            end
        end
        freeinds, body
    else
        nothing, func
    end
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


"""
    @einsum (i,j...) -> expr
    @einsum expr

Conducts tensor computation based on [Einstein summation convention](https://en.wikipedia.org/wiki/Einstein_notation).
The arguments of the anonymous function are regard as **free indices**.
If arguments are not given, they are guessed based on the order that indices appears from left to right.

# Examples
```jldoctest einsum
julia> A = rand(Mat{3,3})
3×3 Tensor{Tuple{3, 3}, Float64, 2, 9}:
 0.590845  0.460085  0.200586
 0.766797  0.794026  0.298614
 0.566237  0.854147  0.246837

julia> B = rand(Mat{3,3})
3×3 Tensor{Tuple{3, 3}, Float64, 2, 9}:
 0.579672   0.066423  0.112486
 0.648882   0.956753  0.276021
 0.0109059  0.646691  0.651664

julia> @einsum (i,j) -> A[i,k] * B[k,j]
3×3 Tensor{Tuple{3, 3}, Float64, 2, 9}:
 0.643225  0.609151  0.32417
 0.962977  1.00373   0.500018
 0.885164  1.01445   0.460311

julia> @einsum A[i,k] * B[k,j] # same as above
3×3 Tensor{Tuple{3, 3}, Float64, 2, 9}:
 0.643225  0.609151  0.32417
 0.962977  1.00373   0.500018
 0.885164  1.01445   0.460311

julia> @einsum A[i,j] * B[i,j]
2.454690093453888
```

!!! note

    `@einsum` is experimental and could change or disappear in future versions of Tensorial.
"""
macro einsum(expr)
    freeinds, body = anonymous_args_body(expr)
    einex = einsum_instantiate(body)
    freeinds === nothing && return einex.ex
    isempty(freeinds) && return einex.ex
    perm = find_perm(einex.freeinds => freeinds)
    :(permutedims($(einex.ex), $(ValTuple(perm...))))
end

ValTuple(x...) = Val(x)

function find_perm((inds, freeinds)::Pair)::Vector{Int}
    map(freeinds) do index
        I = findall(==(index), inds)
        @assert I !== nothing
        only(I)
    end
end

function find_freeindices(allinds::Vector)
    freeinds = []
    for index in unique(allinds)
        x = findall(==(index), allinds)
        length(x) > 2 && error("@einsum: index $index appears more than twice")
        length(x) == 1 && push!(freeinds, index)
    end
    freeinds
end

struct EinsumExpr
    ex::Any
    freeinds::Vector
    allinds::Vector
    function EinsumExpr(ex, freeinds, allinds)
        find_freeindices(allinds) # check dummy indices
        new(ex, freeinds, allinds)
    end
end

isscalarexpr(x::EinsumExpr) = isempty(x.freeinds)

function einsum_instantiate(expr)
    if Meta.isexpr(expr, :call)
        if expr.args[1] == :*
            einex = einsum_instantiate(expr.args[2])
            for ex in expr.args[3:end]
                einex = einsum_instantiate_contraction(einex, einsum_instantiate(ex))
            end
            return einex
        elseif expr.args[1] == :/
            lhs = einsum_instantiate(expr.args[2])
            rhs = einsum_instantiate(expr.args[3])
            return einsum_instantiate_division(lhs, rhs)
        elseif expr.args[1] == :+ || expr.args[1] == :-
            einex = einsum_instantiate(expr.args[2])
            for ex in expr.args[3:end]
                einex = einsum_instantiate_addition(expr.args[1], einex, einsum_instantiate(ex))
            end
            return einex
        end
    elseif Meta.isexpr(expr, :ref)
        return einsum_instantiate_tensor(esc(expr.args[1]), expr.args[2:end])
    end
    EinsumExpr(expr, [], [])
end

# ref case
function einsum_instantiate_tensor(tensor, inds)
    freeinds = find_freeindices(inds)
    if isempty(freeinds) # handle `A[i,i]`
        ex = :($einsum_contraction(Val(()), ($tensor,), ($(ValTuple(inds...)),)))
        return EinsumExpr(ex, freeinds, inds)
    else
        return EinsumExpr(tensor, inds, inds)
    end
end

# division
function einsum_instantiate_division(lhs::EinsumExpr, rhs::EinsumExpr)
    @assert isempty(rhs.freeinds)
    ex = Expr(:call, :/, lhs.ex, rhs.ex)
    EinsumExpr(ex, lhs.freeinds, lhs.allinds) # is it ok to ignore indices of `rhs`?
end

# summation
function einsum_instantiate_addition(op::Symbol, lhs::EinsumExpr, rhs::EinsumExpr)
    @assert Set(lhs.freeinds) == Set(rhs.freeinds)
    perm = find_perm(rhs.freeinds => lhs.freeinds)
    ex = Expr(:call, op, lhs.ex, :(permutedims($(rhs.ex), $(ValTuple(perm...)))))
    EinsumExpr(ex, lhs.freeinds, lhs.freeinds) # reset allinds
end

# contraction
function einsum_instantiate_contraction(lhs::EinsumExpr, rhs::EinsumExpr)
    if isempty(lhs.freeinds)
        ex = Expr(:call, :*, lhs.ex, rhs.ex)
        return EinsumExpr(ex, rhs.freeinds, vcat(lhs.allinds, rhs.allinds))
    elseif isempty(rhs.freeinds)
        ex = Expr(:call, :*, lhs.ex, rhs.ex)
        return EinsumExpr(ex, lhs.freeinds, vcat(lhs.allinds, rhs.allinds))
    else
        freeinds = find_freeindices(vcat(lhs.freeinds, rhs.freeinds))
        allinds = vcat(lhs.allinds, rhs.allinds)
        ex = :($einsum_contraction($(ValTuple(freeinds...)), ($(lhs.ex), $(rhs.ex)), ($(ValTuple(lhs.freeinds...)), $(ValTuple(rhs.freeinds...)))))
        return EinsumExpr(ex, freeinds, allinds)
    end
end

@generated function einsum_contraction(::Val{freeinds}, tensors::Tuple, _tensorinds::Tuple{Vararg{Val}}) where {freeinds}
    tensorinds = map(p -> p.parameters[1], _tensorinds.parameters)

    allinds = vcat([collect(x) for x in tensorinds]...)
    dummyinds = setdiff(allinds, freeinds)
    allinds = [freeinds..., dummyinds...]

    # check dimensions
    dummyaxes = Base.OneTo{Int}[]
    for symbol in dummyinds
        dim = 0
        count = 0
        for (i, inds) in enumerate(tensorinds)
            for I in findall(==(symbol), inds)
                if dim == 0
                    dim = size(tensors.parameters[i], I)
                    push!(dummyaxes, axes(tensors.parameters[i], I))
                else
                    size(tensors.parameters[i], I) == dim || error("@einsum: dimension mismatch")
                end
                count += 1
            end
        end
        count == 2 || error("@einsum: index $symbol appears more than twice")
    end

    # tensor -> global indices (connectivities)
    whichindices = Vector{Int}[]
    for (i, inds) in enumerate(tensorinds)
        length(inds) == ndims(tensors.parameters[i]) || error("@einsum: the number of indices does not match the number of dimensions")
        whichinds = map(inds) do index
            I = findall(==(index), allinds)
            @assert I !== nothing
            only(I)
        end
        push!(whichindices, collect(whichinds))
    end

    if freeinds == ()
        freeaxes = ()
    else
        perm = map(freeinds) do index
            only(findall(==(index), vcat(map(collect, tensorinds)...)))
        end
        TT = _permutedims(otimes(map(Space, tensors.parameters)...), Val(perm)) |> tensortype
        freeaxes = axes(TT)
    end

    sumexps = map(CartesianIndices(freeaxes)) do finds
        xs = map(CartesianIndices(Tuple(dummyaxes))) do dinds
            ainds = [Tuple(finds)..., Tuple(dinds)...]
            exps = map(enumerate(tensors.parameters)) do (i, t)
                inds = ainds[whichindices[i]]
                I = independent_indices(t)[inds...]
                :(Tuple(tensors[$i])[$I])
            end
            Expr(:tuple, exps...)
        end
        :($sumargs($(xs...)))
    end

    if freeinds == ()
        quote
            @_inline_meta
            $(only(sumexps))
        end
    else
        quote
            @_inline_meta
            $TT($(sumexps[indices(TT)]...))
        end
    end
end

function sumargs(x, ys...)
    ret = *(x...)
    @simd for i in eachindex(ys)
        @inbounds ret += *(ys[i]...)
    end
    ret
end
