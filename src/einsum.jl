"""
    @einsum [TensorType] expr

Performs tensor computations using the [Einstein summation convention](https://en.wikipedia.org/wiki/Einstein_notation).
Since `@einsum` cannot fully infer tensor symmetries, it is possible to annotate
the returned tensor type (though this is not checked for correctness).
This can help eliminate the computation of the symmetric part, improving performance.

# Examples
```jldoctest einsum
julia> A = rand(Mat{3,3});

julia> B = rand(Mat{3,3});

julia> (@einsum C[i,j] := A[j,k] * B[k,i]) ≈ (A * B)'
true

julia> (@einsum c := A[i,j] * A[i,j]) ≈ A ⋅ A
true

julia> (@einsum SymmetricSecondOrderTensor{3} D[i,j] := A[k,i] * A[k,j]) ≈ A' * A
true
```
"""
macro einsum(expr)
    einsum_exprssion(:Any, expr)
end
macro einsum(TT, expr)
    einsum_exprssion(TT, expr)
end

function einsum_exprssion(TT, expr)
    varname, freeinds, body = split_defexpr(expr)
    einex = einsum_instantiate(body, TT)
    isnothing(freeinds) && return :($(esc(varname)) = $(einex.ex))
    isempty(freeinds) && return :($(esc(varname)) = Tensor{Tuple{}}($(einex.ex)))
    perm = find_perm(einex.freeinds => freeinds)
    :($(esc(varname)) = permutedims($(einex.ex), $(ValTuple(perm...))))
end

ValTuple(x...) = Val(x)

function split_defexpr(func::Expr)
    Meta.isexpr(func, :(:=)) || throw(ArgumentError("wrong @einsum expression"))
    lhs = func.args[1]
    body = func.args[2]
    if Meta.isexpr(lhs, :ref)
        varname = lhs.args[1]
        freeinds = lhs.args[2:end]
    elseif lhs isa Symbol
        varname = lhs
        freeinds = nothing
    else
        throw(ArgumentError("wrong @einsum expression"))
    end
    if Meta.isexpr(body, :block)
        body = only([x for x in body.args if !(x isa LineNumberNode)])
    end
    varname, freeinds, body
end

function find_perm((src, dest)::Pair{<: Vector, <: Vector})::Vector{Int}
    map(index -> only(findall(==(index), src)), dest)
end

function find_freeindices(indices::Vector)
    freeindices = eltype(indices)[]
    for index in unique(indices)
        x = findall(==(index), indices)
        length(x)  > 2 && error("@einsum: index $index appears more than twice")
        length(x) == 1 && push!(freeindices, index)
    end
    freeindices
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
        ex = :($contract_einsum(Any, ($(einex.ex),), ($(ValTuple(einex.allinds...)),)))
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

    if length(exprs) > 2
        dummies_list = findall(exprs) do einex # tensors having only dummy indices
            isscalarexpr(einex) || !any(in(freeinds), einex.freeinds)
        end
        # compute dummy indices first, improving performance (#191)
        if !isempty(dummies_list)
            dummies = exprs[dummies_list]
            deleteat!(exprs, dummies_list)
            exprs = [dummies; exprs]
        end
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
        all_indices = [lhs.freeinds; rhs.freeinds]
        free_indices = find_freeindices(all_indices)
        if TT == :Any && allunique(lhs.freeinds) && allunique(rhs.freeinds) # use faster computation if possible
            dummy_indices = setdiff(all_indices, free_indices)
            lhs_dims = map(dummy_index -> only(findall(==(dummy_index), lhs.freeinds)), dummy_indices)
            rhs_dims = map(dummy_index -> only(findall(==(dummy_index), rhs.freeinds)), dummy_indices)
            ex = :($contract($(lhs.ex), $(rhs.ex), $(ValTuple(lhs_dims...)), $(ValTuple(rhs_dims...))))
        else
            ex = :($contract_einsum($TT, ($(lhs.ex), $(rhs.ex)), ($(ValTuple(lhs.freeinds...)), $(ValTuple(rhs.freeinds...)))))
        end
        return EinsumExpr(ex, free_indices, [lhs.allinds; rhs.allinds])
    end
end

# this returns expressions for each index
function contract_einsum_expr(tensortypes::NTuple{N}, names::NTuple{N}, tensor_indices::NTuple{N, Vector}) where {N}
    all_indices = reduce(vcat, tensor_indices)
    free_indices = find_freeindices(all_indices)
    dummy_indices = setdiff(all_indices, free_indices)
    all_indices = [dummy_indices; free_indices]

    # check dimensions
    dummy_axes = Base.OneTo{Int}[]
    for dummy_index in dummy_indices
        axs = mapreduce(vcat, zip(tensortypes, tensor_indices)) do (tensor, inds) # (A, [:i,:j])
            map(i -> axes(tensor, i), findall(==(dummy_index), inds))
        end
        length(axs) < 2 && error("contract_einsum_expr: wrong free indices given")
        length(axs) > 2 && error("contract_einsum_expr: index $dummy_index appears more than twice")
        ax = unique(axs)
        length(ax) == 1 && push!(dummy_axes, only(ax))
        length(ax)  > 1 && error("contract_einsum_expr: dimension mismatch at index $dummy_index")
    end

    # create indexmaps from each tensor to `all_indices`
    indexmaps = Vector{Int}[]
    for (tensor, inds) in zip(tensortypes, tensor_indices) # (A, [:i,:j])
        ndims(tensor) == length(inds) || error("contract_einsum_expr: the number of indices does not match the order of tensor")
        indices = map(index -> only(findall(==(index), all_indices)), inds)
        push!(indexmaps, indices)
    end

    T = promote_type(map(eltype, tensortypes)...)
    if isempty(free_indices)
        TT = T
        free_axes = ()
    else
        perm = map(index -> only(findall(==(index), reduce(vcat, tensor_indices))), free_indices)
        TT = tensortype(_permutedims(⊗(map(Space, tensortypes)...), Val(tuple(perm...)))){T}
        free_axes = axes(TT)
    end

    sumexps = map(CartesianIndices(free_axes)) do free_cartesian_index
        xs = map(CartesianIndices(Tuple(dummy_axes))) do dummy_cartesian_index
            cartesian_index = Tuple(CartesianIndex(dummy_cartesian_index, free_cartesian_index))
            exps = map(enumerate(tensortypes)) do (i, tensor)
                indices = cartesian_index[indexmaps[i]]
                getindex_expr(tensor, names[i], indices...)
            end
            Expr(:call, :*, exps...)
        end
        Expr(:call, simdsum, xs...)
    end

    TT, sumexps
end

@inline function simdsum(x, ys...)
    @inbounds @simd for y in ys
        x += y
    end
    x
end

@generated function contract_einsum(::Type{TT1}, tensors::Tuple{Vararg{AbstractTensor, N}}, indices::Tuple{Vararg{Val, N}}) where {TT1, N}
    tensortypes = Tuple(tensors.parameters)
    names = ntuple(i -> :(tensors[$i]), Val(N))
    tensor_indices = Tuple(map(p -> collect(only(p.parameters)), indices.parameters))
    TT2, exps = contract_einsum_expr(tensortypes, names, tensor_indices)
    TT = TT1 <: AbstractTensor ? TT1 : TT2
    ex = TT <: Real ? only(exps) : Expr(:tuple, exps[tensorindices_tuple(TT)]...)
    quote
        @_inline_meta
        @inbounds $TT($ex)
    end
end
