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
    :(permutedims($(einex.ex), $(ValTuple(perm...))))
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
function einsum_instantiate_contraction(TT, exprs::Vector{EinsumExpr})
    freeinds = find_freeindices(mapreduce(x->x.freeinds, vcat, exprs))

    dummies_list = findall(exprs) do einex # tensors having only dummy indices
        isscalarexpr(einex) || !any(in(freeinds), einex.freeinds)
    end

    # compute dummy indices first
    if !isempty(dummies_list)
        dummies = exprs[dummies_list]
        deleteat!(exprs, dummies_list)
        exprs = [dummies; exprs]
    end

    # lastly apply `TT`
    ex = foldl(einsum_instantiate_contraction, exprs[1:end-1])
    einsum_instantiate_contraction(ex, exprs[end], TT)
end

function einsum_instantiate_contraction(lhs::EinsumExpr, rhs::EinsumExpr, TT = :Any)
    if isscalarexpr(lhs) || isscalarexpr(rhs)
        ex = Expr(:call, :*, lhs.ex, rhs.ex)
        return EinsumExpr(ex, [lhs.freeinds; rhs.freeinds], [lhs.allinds; rhs.allinds])
    else
        freeinds = find_freeindices([lhs.freeinds; rhs.freeinds])
        allinds = [lhs.allinds; rhs.allinds]
        ex = :($einsum_contraction($TT, $(ValTuple(freeinds...)), ($(lhs.ex), $(rhs.ex)), ($(ValTuple(lhs.freeinds...)), $(ValTuple(rhs.freeinds...)))))
        return EinsumExpr(ex, freeinds, allinds)
    end
end

# for dummy indices
@inline function simdsum(x, ys...)
    @inbounds @simd for y in ys
        x += y
    end
    x
end

function einsum_contraction_expr(free_indices::Vector, tensors::Vector, tensor_indices::Vector{<: AbstractVector})
    @assert length(tensors) == length(tensor_indices)

    all_indices = mapreduce(collect, vcat, tensor_indices)
    dummy_indices = setdiff(all_indices, free_indices)
    all_indices = [free_indices; dummy_indices]

    # check dimensions
    dummy_axes = Base.OneTo{Int}[]
    for dummy_index in dummy_indices
        axs = mapreduce(vcat, zip(tensors, tensor_indices)) do (tensor, inds) # (A, [:i,:j])
            map(i -> axes(tensor, i), findall(==(dummy_index), inds))
        end
        length(axs) < 2 && error("@einsum: wrong free indices given")
        length(axs) > 2 && error("@einsum: index $dummy_index appears more than twice")
        ax = unique(axs)
        length(ax) == 1 && push!(dummy_axes, only(ax))
        length(ax)  > 1 && error("@einsum: dimension mismatch at index $dummy_index")
    end

    # create indexmaps from each tensor to `all_indices`
    indexmaps = Vector{Int}[]
    for (tensor, inds) in zip(tensors, tensor_indices) # (A, [:i,:j])
        ndims(tensor) == length(inds) || error("@einsum: the number of indices does not match the order of tensor #$i")
        indices = map(index -> only(findall(==(index), all_indices)), inds)
        push!(indexmaps, indices)
    end

    T = promote_type(map(eltype, tensors)...)
    if isempty(free_indices)
        TT = T
        free_axes = ()
    else
        perm = map(index -> only(findall(==(index), reduce(vcat, tensor_indices))), free_indices)
        TT = tensortype(_permutedims(otimes(map(Space, tensors)...), Val(tuple(perm...)))){T}
        free_axes = axes(TT)
    end

    sumexps = map(CartesianIndices(free_axes)) do free_cartesian_index
        xs = map(CartesianIndices(Tuple(dummy_axes))) do dummy_cartesian_index
            cartesian_index = Tuple(CartesianIndex(free_cartesian_index, dummy_cartesian_index))
            exps = map(enumerate(tensors)) do (i, tensor)
                indices = cartesian_index[indexmaps[i]]
                getindex_expr(tensor, :(tensors[$i]), indices...)
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
