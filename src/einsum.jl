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
 0.325977  0.894245  0.953125
 0.549051  0.353112  0.795547
 0.218587  0.394255  0.49425

julia> B = rand(Mat{3,3})
3×3 Tensor{Tuple{3, 3}, Float64, 2, 9}:
 0.748415  0.00744801  0.682533
 0.578232  0.199377    0.956741
 0.727935  0.439243    0.647855

julia> @einsum (i,j) -> A[i,k] * B[k,j]
3×3 Tensor{Tuple{3, 3}, Float64, 2, 9}:
 1.45486   0.599373  1.69554
 1.19421   0.42393   1.22798
 0.751346  0.297329  0.846595

julia> @einsum A[i,k] * B[k,j] # same as above
3×3 Tensor{Tuple{3, 3}, Float64, 2, 9}:
 1.45486   0.599373  1.69554
 1.19421   0.42393   1.22798
 0.751346  0.297329  0.846595

julia> @einsum A[i,j] * B[i,j]
2.7026716125808266
```
"""
macro einsum(expr)
    einsum_exprssion(:Any, expr)
end
macro einsum(TT, expr)
    einsum_exprssion(TT, expr)
end

function einsum_exprssion(TT, expr)
    freeinds, body = anonymous_args_body(expr)
    einex = einsum_instantiate(body, TT)
    freeinds === nothing && return einex.ex
    isempty(freeinds) && return einex.ex
    perm = find_perm(einex.freeinds => freeinds)
    :(convert($TT, permutedims($(einex.ex), $(ValTuple(perm...)))))
end

ValTuple(x...) = Val(x)

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
            body = only([x for x in body.args if !(x isa LineNumberNode)])
        end
        freeinds, body
    else
        nothing, func
    end
end

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

function einsum_instantiate(expr, TT) # TT is :Any if not given in @einsum or not top level
    if Meta.isexpr(expr, :call)
        if expr.args[1] == :*
            return einsum_instantiate_contraction(TT, map(x -> einsum_instantiate(x, :Any), expr.args[2:end]))
        elseif expr.args[1] == :/
            lhs = einsum_instantiate(expr.args[2], TT)
            rhs = einsum_instantiate(expr.args[3], :Any)
            return einsum_instantiate_division(lhs, rhs)
        elseif expr.args[1] == :+ || expr.args[1] == :-
            if length(expr.args) == 2 # handle unary operator `-a[i]` (#201)
                return einsum_instantiate(Expr(:call, :*, ifelse(expr.args[1]==:+, 1, -1), expr.args[2]), TT)
            else
                return mapreduce(x -> einsum_instantiate(x, TT),
                                 (x, y) -> einsum_instantiate_addition(expr.args[1], x, y),
                                 expr.args[2:end])
            end
        end
    elseif Meta.isexpr(expr, :ref)
        ex = esc(expr.args[1])
        allinds = expr.args[2:end]
        freeinds = find_freeindices(allinds)
        return einsum_instantiate_tensor(EinsumExpr(ex, freeinds, allinds))
    end
    EinsumExpr(expr, [], [])
end

# ref case
function einsum_instantiate_tensor(einex::EinsumExpr)
    if isscalarexpr(einex) # handle `A[i,i]`
        ex = :($einsum_contraction(Any, Val(()), ($(einex.ex),), ($(ValTuple(einex.allinds...)),)))
        return EinsumExpr(ex, einex.freeinds, einex.allinds)
    else
        return einex
    end
end

# division
function einsum_instantiate_division(lhs::EinsumExpr, rhs::EinsumExpr)
    @assert isscalarexpr(rhs)
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
    if isscalarexpr(lhs) || isscalarexpr(rhs)
        ex = Expr(:call, :*, lhs.ex, rhs.ex)
        return EinsumExpr(ex, [lhs.freeinds; rhs.freeinds], [lhs.allinds; rhs.allinds])
    else
        freeinds = find_freeindices([lhs.freeinds; rhs.freeinds])
        allinds = [lhs.allinds; rhs.allinds]
        ex = :($einsum_contraction(Any, $(ValTuple(freeinds...)), ($(lhs.ex), $(rhs.ex)), ($(ValTuple(lhs.freeinds...)), $(ValTuple(rhs.freeinds...)))))
        return EinsumExpr(ex, freeinds, allinds)
    end
end

function einsum_instantiate_contraction(TT, exprs::Vector{EinsumExpr})
    freeinds = find_freeindices(mapreduce(x->x.freeinds, vcat, exprs))

    list = findall(exprs) do einex # tensors having only dummy indices
        isscalarexpr(einex) || !any(in(freeinds), einex.freeinds)
    end

    if !isempty(list)
        dummy_tensors = exprs[list]
        deleteat!(exprs, list)
        if !isempty(exprs)
            dummy_tensors = [dummy_tensors; popfirst!(exprs)]
        end
        push!(exprs, reduce(einsum_instantiate_contraction, dummy_tensors))
    end

    length(exprs) == 1 && return only(exprs)

    exprs::Vector{EinsumExpr} = foldl(exprs) do x, y
        lhs::EinsumExpr = x isa Vector ? x[end] : x
        rhs::EinsumExpr = y
        if isscalarexpr(lhs) || isscalarexpr(rhs)
            ex = Expr(:call, :*, lhs.ex, rhs.ex)
            tails = [EinsumExpr(ex, [lhs.freeinds; rhs.freeinds], [lhs.allinds; rhs.allinds])]
        else
            tails = [lhs, rhs]
        end
        x isa Vector ? append!(x[1:end-1], tails) : tails
    end

    allinds = mapreduce(x->x.allinds, vcat, exprs)
    ex = :($einsum_contraction($TT, $(ValTuple(freeinds...)), ($([x.ex for x in exprs]...),), ($([ValTuple(x.freeinds...) for x in exprs]...),)))
    return EinsumExpr(ex, freeinds, allinds)
end

# for dummy indices
@inline function simdsum(x, ys...)
    @inbounds @simd for y in ys
        x += y
    end
    x
end

function einsum_contraction_expr(freeinds::Vector, tensors::Vector, tensorinds::Vector{<: AbstractVector})
    @assert length(tensors) == length(tensorinds)

    allinds = mapreduce(collect, vcat, tensorinds)
    dummyinds = setdiff(allinds, freeinds)
    allinds = [freeinds; dummyinds]

    # check dimensions
    dummyaxes = Base.OneTo{Int}[]
    for di in dummyinds
        dim = 0
        count = 0
        for (i, inds) in enumerate(tensorinds)
            for I in findall(==(di), inds)
                if dim == 0
                    dim = size(tensors[i], I)
                    push!(dummyaxes, axes(tensors[i], I))
                else
                    size(tensors[i], I) == dim || error("@einsum: dimension mismatch")
                end
                count += 1
            end
        end
        count == 2 || error("@einsum: index $symbol appears more than twice")
    end

    # tensor -> global indices
    whichindices = Vector{Int}[]
    for (i, inds) in enumerate(tensorinds)
        length(inds) == ndims(tensors[i]) || error("@einsum: the number of indices does not match the number of dimensions")
        whichinds = map(inds) do index
            I = findall(==(index), allinds)
            @assert I !== nothing
            only(I)
        end
        push!(whichindices, whichinds)
    end

    T = promote_type(map(eltype, tensors)...)
    if isempty(freeinds)
        TT = T
        freeaxes = ()
    else
        perm = map(freeinds) do index
            only(findall(==(index), reduce(vcat, tensorinds)))
        end
        TT = tensortype(_permutedims(otimes(map(Space, tensors)...), Val(tuple(perm...)))){T}
        freeaxes = axes(TT)
    end

    sumexps = map(CartesianIndices(freeaxes)) do finds
        xs = map(CartesianIndices(Tuple(dummyaxes))) do dinds
            ainds = [Tuple(finds)..., Tuple(dinds)...]
            exps = map(enumerate(tensors)) do (i, t)
                inds = ainds[whichindices[i]]
                getindex_expr(t, :(tensors[$i]), inds...)
            end
            Expr(:call, :*, exps...)
        end
        Expr(:call, simdsum, xs...)
    end

    TT, sumexps
end

@generated function einsum_contraction(::Type{TT1}, ::Val{freeinds}, tensors::Tuple{Vararg{AbstractTensor, N}}, tensorinds::Tuple{Vararg{Val, N}}) where {TT1, freeinds, N}
    TT2, exps = einsum_contraction_expr(collect(freeinds),
                                        collect(Type{<: AbstractTensor}, tensors.parameters),
                                        Vector{Symbol}[collect(p.parameters[1]) for p in tensorinds.parameters])
    TT = TT1 <: AbstractTensor ? TT1 : TT2
    tupleinds = TT <: Real ? Colon() : tensorindices_tuple(TT)
    quote
        @_inline_meta
        $TT($(exps[tupleinds]...))
    end
end
