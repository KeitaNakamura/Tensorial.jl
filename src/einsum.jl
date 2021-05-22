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
        ex = :($einsum_contraction(Val(()), ($(einex.ex),), ($(ValTuple(einex.allinds...)),)))
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
    if isscalarexpr(lhs)
        ex = Expr(:call, :*, lhs.ex, rhs.ex)
        return EinsumExpr(ex, rhs.freeinds, vcat(lhs.allinds, rhs.allinds))
    elseif isscalarexpr(rhs)
        ex = Expr(:call, :*, lhs.ex, rhs.ex)
        return EinsumExpr(ex, lhs.freeinds, vcat(lhs.allinds, rhs.allinds))
    else
        freeinds = find_freeindices(vcat(lhs.freeinds, rhs.freeinds))
        allinds = vcat(lhs.allinds, rhs.allinds)
        ex = :($einsum_contraction($(ValTuple(freeinds...)), ($(lhs.ex), $(rhs.ex)), ($(ValTuple(lhs.freeinds...)), $(ValTuple(rhs.freeinds...)))))
        return EinsumExpr(ex, freeinds, allinds)
    end
end

# for dummy indices
function sumargs(x, ys...)
    ret = *(x...)
    @simd for i in eachindex(ys)
        @inbounds ret += *(ys[i]...)
    end
    ret
end

function einsum_contraction_expr(freeinds::Tuple, tensors::Tuple{Vararg{Any, N}}, tensorinds::Tuple{Vararg{AbstractVector, N}}) where {N}
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

    # tensor -> global indices (connectivities)
    whichindices = Vector{Int}[]
    for (i, inds) in enumerate(tensorinds)
        length(inds) == ndims(tensors[i]) || error("@einsum: the number of indices does not match the number of dimensions")
        whichinds = map(inds) do index
            I = findall(==(index), allinds)
            @assert I !== nothing
            only(I)
        end
        push!(whichindices, collect(whichinds))
    end

    T = promote_type(map(eltype, tensors)...)
    if isempty(freeinds)
        TT = T
        freeaxes = ()
        tupleinds = Colon()
    else
        perm = map(freeinds) do index
            only(findall(==(index), vcat(tensorinds...)))
        end
        TT = tensortype(_permutedims(otimes(map(Space, tensors)...), Val(perm))){T}
        freeaxes = axes(TT)
        tupleinds = indices(TT)
    end

    sumexps = map(CartesianIndices(freeaxes)) do finds
        xs = map(CartesianIndices(Tuple(dummyaxes))) do dinds
            ainds = [Tuple(finds)..., Tuple(dinds)...]
            exps = map(enumerate(tensors)) do (i, t)
                inds = ainds[whichindices[i]]
                I = independent_indices(t)[inds...]
                :(Tuple(tensors[$i])[$I])
            end
            Expr(:tuple, exps...)
        end
        :($sumargs($(xs...)))
    end

    TT, sumexps[tupleinds]
end

@generated function _einsum_contraction(::Val{freeinds}, tensors::Tuple, tensorinds::Tuple{Vararg{Val}}) where {freeinds}
    TT, exps = einsum_contraction_expr(freeinds, Tuple(tensors.parameters), Tuple(map(p -> collect(p.parameters[1]), tensorinds.parameters)))
    quote
        @_inline_meta
        $TT($(exps...))
    end
end

# call contraction methods if possible
@generated function einsum_contraction(_freeinds::Val{freeinds}, tensors::Tuple, _tensorinds::Tuple{Vararg{Val}}) where {freeinds}
    default = quote
        @_inline_meta
        _einsum_contraction(_freeinds, tensors, _tensorinds)
    end

    length(tensors.parameters) != 2 && return default
    lhsall, rhsall = map(p -> collect(p.parameters[1]), _tensorinds.parameters)

    lhsdummy = Int[]
    rhsdummy = Int[]
    for i in eachindex(lhsall)
        index = lhsall[i]
        I = findall(==(index), rhsall)
        isempty(I) && continue
        push!(lhsdummy, i)
        push!(rhsdummy, only(I))
    end

    issequence(x) = isempty(x) ? false : x == first(x):first(x)+length(x)-1
    if issequence(lhsdummy) && issequence(rhsdummy) &&
       last(lhsdummy) == lastindex(lhsall) && first(rhsdummy) == firstindex(rhsall)
        lhsfree = deleteat!(lhsall, lhsdummy)
        rhsfree = deleteat!(rhsall, rhsdummy)
        if freeinds == tuple(lhsfree..., rhsfree...)
            return quote
                @_inline_meta
                contraction(tensors..., $(Val(length(lhsdummy))))
            end
        end
    end

    default
end
