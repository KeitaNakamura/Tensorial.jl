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
            args = lhs.args
        elseif lhs isa Symbol
            args = [lhs]
        else
            throw(ArgumentError("wrong arguments in anonymous function expression"))
        end
        args, body
    else
        nothing, Expr(:block, func)
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

Conducts tensor computation based on Einstein summation conversion.
The arguments of the anonymous function are regard as **free indices**.
If arguments are not given, they are guessed based on the order that indices appears from left to right.

# Examples
```jldoctest
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

Currently, this macro does not support summation of tensors.
So, you need to divide the equation into terms and then apply this macro to each term as follows:

```jldoctest
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

julia> @einsum (i,j) -> A[i,k]*B[k,j] + A[j,k]*B[k,i] # not supported
ERROR: LoadError: @einsum: unsupported computation
[...]

julia> (@einsum (i,j) -> A[i,k]*B[k,j]) + (@einsum (i,j) -> A[j,k]*B[k,i]) # this is ok
3×3 Tensor{Tuple{3, 3}, Float64, 2, 9}:
 1.28645  1.57213  1.20933
 1.57213  2.00746  1.51446
 1.20933  1.51446  0.920622
```

!!! note

    `@einsum` is experimental and could change or disappear in future versions of Tensorial.
"""
macro einsum(ex)
    freeinds, code = anonymous_args_body(ex)
    tensors = findtensors!(code)
    tensor_exprs = [ValTuple(t.args...) for t in tensors]

    allinds = vcat([collect(t.args[2:end]) for t in tensors]...)
    if freeinds === nothing
        freeinds = [index for index in unique(allinds) if length(findall(==(index), allinds)) == 1]
    end
    dummyinds = setdiff(allinds, freeinds)

    tensor_symbols = [t.args[1] for t in tensors]
    quote
        $einsum(tensors -> $code, tuple($(tensor_exprs...)), $(ValTuple(freeinds...)), $(ValTuple(dummyinds...)), tuple($(tensor_symbols...)))
    end |> esc
end

ValTuple(x...) = Val(x)

findtensors!(ex::Expr) = _findtensors!(Expr[], ex)
_findtensors!(tensors::Vector{Expr}, ::Any) = tensors
function _findtensors!(tensors::Vector{Expr}, expr::Expr)
    for i in eachindex(expr.args)
        ex = expr.args[i]

        # check for not `*` operator
        if Meta.isexpr(ex, :call)
            if ex.args[1] == :/ # devide by scalar is ok
                check_no_ref_exprs(ex.args[3])
            elseif ex.args[1] != :*
                foreach(check_no_ref_exprs, ex.args) # ok if arguments are not tensors
            end
        end

        if Meta.isexpr(ex, :ref)
            push!(tensors, ex)
            expr.args[i] = :(tensors[$(length(tensors))])
        else
            _findtensors!(tensors, ex)
        end
    end
    tensors
end

check_no_ref_exprs(ex) = nothing
function check_no_ref_exprs(ex::Expr)
    Meta.isexpr(ex, :ref) && error("@einsum: unsupported computation")
    foreach(check_no_ref_exprs, ex.args)
end

@generated function einsum(f, tensor_exprs::Tuple{Vararg{Val}}, ::Val{freeinds}, ::Val{dummyinds}, tensors::Tuple) where {freeinds, dummyinds}
    @assert unique([freeinds..., dummyinds]) == [freeinds..., dummyinds]

    texps = map(p -> p.parameters[1], tensor_exprs.parameters)
    allinds = [freeinds..., dummyinds...]

    # check dimensions
    dummyaxes = Base.OneTo{Int}[]
    for symbol in dummyinds
        dim = 0
        count = 0
        for (i, t) in enumerate(texps)
            indices = findall(==(symbol), t)
            for I in indices
                @assert I != 1 # 1 is for tensor name
                if dim == 0
                    dim = size(tensors.parameters[i], I-1)
                    push!(dummyaxes, axes(tensors.parameters[i], I-1))
                else
                    size(tensors.parameters[i], I-1) == dim || error("@einsum: dimension mismatch")
                end
                count += 1
            end
        end
        count == 2 || error("@einsum: index $symbol appears more than twice")
    end

    # tensor -> global indices (connectivities)
    whichindices = Vector{Int}[]
    for (i, t) in enumerate(texps)
        length(t[2:end]) == ndims(tensors.parameters[i]) || error("@einsum: the number of indices does not match the number of dimensions in expression $(t[1])[$(join(t[2:end], ","))]")
        inds = map(t[2:end]) do index
            I = findfirst(==(index), allinds)
            @assert I !== nothing
            I
        end
        push!(whichindices, collect(inds))
    end

    if freeinds == ()
        freeaxes = ()
    else
        perm = map(freeinds) do index
            only(findall(==(index), vcat([collect(t[2:end]) for t in texps]...)))
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
        :($sumargs(f, $(xs...)))
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

function sumargs(f, x, ys...)
    ret = f(x)
    @simd for i in eachindex(ys)
        @inbounds ret += f(ys[i])
    end
    ret
end
