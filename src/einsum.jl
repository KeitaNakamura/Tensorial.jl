"""
    @einsum [TensorType] expr

Performs tensor computations using the [Einstein summation convention](https://en.wikipedia.org/wiki/Einstein_notation).
Since `@einsum` cannot fully infer tensor symmetries, it is possible to annotate
the returned tensor type (though this is not checked for correctness).
This can help eliminate the computation of the symmetric part, improving performance.

The `expr` can be an anonymous function, in which case the arguments of the anonymous function are treated as free indices.
If no arguments are provided, the free indices are inferred based on the order in which they appear from left to right.

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

julia> (@einsum (i,j) -> A[j,k] * B[k,i]) ≈ (A * B)'
true

julia> (@einsum A[i,j] * A[i,j]) ≈ A ⋅ A
true

julia> (@einsum SymmetricSecondOrderTensor{3} A[k,i] * A[k,j]) ≈ A' * A
true
```
"""
macro einsum(expr)
    einsum_expression(:Any, expr)
end
macro einsum(TT, expr)
    einsum_expression(TT, expr)
end

function einsum_expression(TT, expr)
    varname, freeinds, body = split_defexpr(expr)

    probe = einsum_instantiate(body, :Any)
    body_TT = can_use_einsum_type_in_body(TT, probe, freeinds) ? TT : :Any
    einex = body_TT == :Any ? probe : einsum_instantiate(body, body_TT)

    if !isnothing(freeinds)
        if isempty(freeinds)
            isempty(einex.freeinds) || einsum_error("wrong free indices given")
        else
            einex = reorder_freeinds(einex, freeinds)
        end
    end

    einex = apply_einsum_type(TT, einex)

    if isnothing(varname) # anonymous function
        return esc(einex.ex)
    else
        if !isnothing(freeinds) && isempty(freeinds)
            return esc(:($varname = Tensor{Tuple{}}($(einex.ex))))
        else
            return esc(:($varname = $(einex.ex)))
        end
    end
end

ValTuple(x...) = Val(x)

einsum_error(msg) = throw(ArgumentError("@einsum: $msg"))

function split_defexpr(func)
    if Meta.isexpr(func, :(:=))
        lhs = func.args[1]
        body = func.args[2]
        if Meta.isexpr(lhs, :ref)
            varname = lhs.args[1]
            freeinds = check_freeinds(lhs.args[2:end])
        elseif lhs isa Symbol
            varname = lhs
            freeinds = nothing
        else
            einsum_error("wrong expression")
        end
    elseif Meta.isexpr(func, :->)
        varname = nothing
        lhs = func.args[1]
        body = func.args[2]
        if Meta.isexpr(lhs, :tuple)
            freeinds = check_freeinds(lhs.args)
        elseif lhs isa Symbol
            freeinds = [lhs]
        else
            einsum_error("wrong arguments in anonymous function expression")
        end
    else
        varname = nothing
        freeinds = nothing
        body = func
    end
    if Meta.isexpr(body, :block)
        body = only_body_expression(body)
    end
    varname, freeinds, body
end

function find_perm((src, dest)::Pair{<: Vector, <: Vector})::Vector{Int}
    map(index -> find_index_position(src, index), dest)
end

function find_index_position(indices::Vector, index)
    pos = findfirst(==(index), indices)
    isnothing(pos) && einsum_error("wrong free indices given")
    pos
end

function only_body_expression(body::Expr)
    args = filter(x -> !(x isa LineNumberNode), body.args)
    length(args) == 1 || einsum_error("expected a single expression")
    only(args)
end

function check_freeinds(freeinds::Vector)
    allunique(freeinds) || einsum_error("free indices must be unique")
    freeinds
end

function same_indices(lhs::Vector, rhs::Vector)
    length(lhs) == length(rhs) && Set(lhs) == Set(rhs)
end

function find_freeindices(indices::Vector)
    counts = Dict{Any, Int}()
    for index in indices
        counts[index] = get(counts, index, 0) + 1
        counts[index] > 2 && einsum_error("index $index appears more than twice")
    end
    [index for index in indices if counts[index] == 1]
end

struct EinsumExpr
    ex::Any
    freeinds::Vector
    allinds::Vector
    typed::Bool
    function EinsumExpr(ex, freeinds, allinds, typed::Bool=false)
        find_freeindices(allinds) # check dummy indices
        new(ex, freeinds, allinds, typed)
    end
end

isscalarexpr(x::EinsumExpr) = isempty(x.freeinds)

function can_use_einsum_type_in_body(TT, einex::EinsumExpr, freeinds)
    TT == :Any && return false
    isscalarexpr(einex) && return false
    !isnothing(freeinds) && !same_indices(einex.freeinds, freeinds) && return false
    true
end

function apply_einsum_type(TT, einex::EinsumExpr)
    if TT == :Any || einex.typed || isscalarexpr(einex)
        return einex
    else
        ex = :($contract_einsum($TT, ($(einex.ex),), ($(ValTuple(einex.freeinds...)),)))
        return EinsumExpr(ex, einex.freeinds, einex.freeinds, true)
    end
end

function reorder_freeinds(einex::EinsumExpr, freeinds::Vector)
    same_indices(einex.freeinds, freeinds) || einsum_error("wrong free indices given")
    einex.freeinds == freeinds && return einex
    perm = find_perm(einex.freeinds => freeinds)
    EinsumExpr(:(permutedims($(einex.ex), $(ValTuple(perm...)))), freeinds, freeinds, false)
end

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
                return mapreduce(x -> einsum_instantiate(x, :Any),
                                 (x, y) -> einsum_instantiate_addition(expr.args[1], x, y),
                                 expr.args[2:end])
            end
        end
    elseif Meta.isexpr(expr, :ref)
        ex = expr.args[1]
        allinds = expr.args[2:end]
        freeinds = find_freeindices(allinds)
        return einsum_instantiate_tensor(TT, EinsumExpr(ex, freeinds, allinds))
    end
    EinsumExpr(expr, [], [])
end

# ref case
function einsum_instantiate_tensor(TT, einex::EinsumExpr)
    if isempty(setdiff(einex.allinds, einex.freeinds)) # no dummy indices
        return apply_einsum_type(TT, einex)
    else
        contract_TT = (TT == :Any || isscalarexpr(einex)) ? :Any : TT
        ex = :($contract_einsum($contract_TT, ($(einex.ex),), ($(ValTuple(einex.allinds...)),)))
        return EinsumExpr(ex, einex.freeinds, einex.allinds, contract_TT != :Any)
    end
end

# division
function einsum_instantiate_division(lhs::EinsumExpr, rhs::EinsumExpr)
    isscalarexpr(rhs) || einsum_error("division is only supported by scalar expressions")
    ex = Expr(:call, :/, lhs.ex, rhs.ex)
    EinsumExpr(ex, lhs.freeinds, [lhs.allinds; rhs.allinds], lhs.typed)
end

# summation
function einsum_instantiate_addition(op::Symbol, lhs::EinsumExpr, rhs::EinsumExpr)
    same_indices(lhs.freeinds, rhs.freeinds) || einsum_error("summed terms must have the same free indices")
    if isscalarexpr(lhs)
        ex = Expr(:call, op, lhs.ex, rhs.ex)
    else
        perm = find_perm(rhs.freeinds => lhs.freeinds)
        ex = Expr(:call, op, lhs.ex, :(permutedims($(rhs.ex), $(ValTuple(perm...)))))
    end
    EinsumExpr(ex, lhs.freeinds, lhs.freeinds, false) # reset allinds
end

# contraction
function einsum_instantiate_contraction(TT, exprs::Vector{EinsumExpr})
    isempty(exprs) && einsum_error("empty contraction expression")

    if TT != :Any
        scalars = filter(isscalarexpr, exprs)
        tensors = filter(!isscalarexpr, exprs)

        if isempty(tensors)
            return instantiate_scalar_product(scalars)
        else
            tensor_TT = isempty(scalars) || length(tensors) >= 2 ? TT : :Any
            einex = instantiate_product_greedy(tensor_TT, tensors)
            if !isempty(scalars)
                scalar_ex = instantiate_scalar_product(scalars)
                einex = einsum_instantiate_contraction(scalar_ex, einex, :Any)
            end
            return apply_einsum_type(TT, einex)
        end
    else
        return instantiate_product_greedy(TT, exprs)
    end
end

function instantiate_scalar_product(exprs::Vector{EinsumExpr})
    isempty(exprs) && einsum_error("empty scalar product")
    length(exprs) == 1 && return only(exprs)

    foldl(exprs[2:end]; init=exprs[1]) do lhs, rhs
        einsum_instantiate_contraction(lhs, rhs, :Any)
    end
end

function instantiate_product_greedy(TT, exprs::Vector{EinsumExpr})
    isempty(exprs) && einsum_error("empty contraction expression")
    length(exprs) == 1 && return apply_einsum_type(TT, only(exprs))

    exprs = copy(exprs)
    while length(exprs) > 2
        i, j = best_contraction_pair(exprs)
        einex = einsum_instantiate_contraction(exprs[i], exprs[j], :Any)
        deleteat!(exprs, j)
        deleteat!(exprs, i)
        insert!(exprs, i, einex)
    end
    return einsum_instantiate_contraction(exprs[1], exprs[2], TT)
end

function einsum_instantiate_contraction(lhs::EinsumExpr, rhs::EinsumExpr, TT = :Any)
    if isscalarexpr(lhs) || isscalarexpr(rhs)
        ex = Expr(:call, :*, lhs.ex, rhs.ex)
        einex = EinsumExpr(ex, [lhs.freeinds; rhs.freeinds], [lhs.allinds; rhs.allinds], lhs.typed || rhs.typed)
        return apply_einsum_type(TT, einex)
    else
        all_indices = [lhs.freeinds; rhs.freeinds]
        free_indices = find_freeindices(all_indices)
        if TT == :Any && allunique(lhs.freeinds) && allunique(rhs.freeinds) # use faster computation if possible
            dummy_indices = setdiff(all_indices, free_indices)
            lhs_dims = find_perm(lhs.freeinds => dummy_indices)
            rhs_dims = find_perm(rhs.freeinds => dummy_indices)
            ex = :($contract($(lhs.ex), $(rhs.ex), $(ValTuple(lhs_dims...)), $(ValTuple(rhs_dims...))))
            return EinsumExpr(ex, free_indices, [lhs.allinds; rhs.allinds], false)
        else
            contract_TT = (TT == :Any || isempty(free_indices)) ? :Any : TT
            ex = :($contract_einsum($contract_TT, ($(lhs.ex), $(rhs.ex)), ($(ValTuple(lhs.freeinds...)), $(ValTuple(rhs.freeinds...)))))
            return EinsumExpr(ex, free_indices, [lhs.allinds; rhs.allinds], contract_TT != :Any)
        end
    end
end

function contraction_cost(lhs::EinsumExpr, rhs::EinsumExpr)
    inds = [lhs.freeinds; rhs.freeinds]
    freeinds = find_freeindices(inds)
    dummies = setdiff(inds, freeinds)

    N = length(dummies)

    is_fast_contract = false

    if N > 0
        lhs_dims = find_perm(lhs.freeinds => dummies)
        rhs_dims = find_perm(rhs.freeinds => dummies)

        lhs_tail = collect((length(lhs.freeinds)-N+1):length(lhs.freeinds))
        rhs_head = collect(1:N)

        is_fast_contract = lhs_dims == lhs_tail && rhs_dims == rhs_head
    end

    return (!is_fast_contract, -N, length(freeinds))
end

function best_contraction_pair(exprs::Vector{EinsumExpr})
    best = (1, 2)
    cost = contraction_cost(exprs[1], exprs[2])
    for i in 1:(length(exprs)-1), j in (i+1):length(exprs)
        c = contraction_cost(exprs[i], exprs[j])
        if c < cost
            best = (i, j)
            cost = c
        end
    end
    return best
end

# this returns expressions for each index
function contract_einsum_expr(tensortypes::NTuple{N}, names::NTuple{N}, tensor_indices::NTuple{N, Vector}) where {N}
    input_indices = reduce(vcat, tensor_indices)
    free_indices = find_freeindices(input_indices)
    dummy_indices = setdiff(input_indices, free_indices)
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
        indices = find_perm(all_indices => inds)
        push!(indexmaps, indices)
    end

    T = promote_type(map(eltype, tensortypes)...)
    if isempty(free_indices)
        TT = T
        free_axes = ()
    else
        perm = find_perm(input_indices => free_indices)
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
    ex = TT <: Real ? only(exps) : Expr(:tuple, exps[independent_to_component_map(TT)]...)
    quote
        @_inline_meta
        @inbounds $TT($ex)
    end
end
