const NumberOrTensor = Union{Number, Tensor}

####################
# dual generations #
####################

@inline _promote_partial(v, p) = p * one(v)

# generate duals from values and partials
@generated function generate_duals(::Tg, v::NTuple{N, Tv}, p::NTuple{N, Tp}) where {Tg, Tv, Tp, N}
    quote
        @_inline_meta
        @ntuple $N i -> begin
            pᵢ = _promote_partial(v[i], p[i])
            partials = @ntuple $N j -> j == i ? pᵢ : zero(pᵢ)
            Dual{Tg}(v[i], partials)
        end
    end
end

# generate values
@inline dual_values(x::Number) = (x,)
@inline dual_values(x::Tensor) = Tuple(x)
@inline dual_values(xs::Tuple{Vararg{NumberOrTensor}}) = _dual_values(promote_elements(xs...))
@inline _dual_values(xs::Tuple{Vararg{Union{T, Tensor{<: Any, T}}}}) where {T} = flatten_tuple(map(dual_values, xs))

# generate partials
@inline dual_partials(x::Number) = (one(x),)
@inline dual_partials(x::Tensor) = convert_ntuple(eltype(x), Tuple(inv.(independent_component_multiplicities(x))))
@inline dual_partials(xs::Tuple{Vararg{NumberOrTensor}}) = _dual_partials(promote_elements(xs...))
@inline _dual_partials(xs::Tuple{Vararg{Union{T, Tensor{<: Any, T}}}}) where {T} = flatten_tuple(map(dual_partials, xs))

@inline dualize(f, x) = _dualize(Tag(f, typeof(x)), x)
@inline dualize(f, x, ::Val{0}) = x
@inline dualize(f, x, ::Val{1}) = dualize(f, x)
@inline dualize(f, x, ::Val{N}) where {N} = dualize(f, dualize(f, x), Val(N-1))

# single argument
@inline function _dualize(::Tg, x::Number) where {Tg}
    Dual{Tg}(x, one(x))
end
@inline function _dualize(::Tg, x::Tensor{S, T}) where {Tg, S, T}
    Tensor{S}(generate_duals(Tg(), dual_values(x), dual_partials(x)))
end

# multiple arguments
@inline function _dualize(::Tg, xs::Tuple{Vararg{NumberOrTensor}}) where {Tg}
    v = dual_values(xs)
    p = dual_partials(xs)
    decompose_vec(Vec(generate_duals(Tg(), v, p)), xs)
end
# helpers for multiple arguments
@inline _reconstruct(v::Vec, x::Number) = only(Tuple(v))
@inline _reconstruct(v::Vec, x::Tensor) = tensortype(Space(x))(Tuple(v))

@inline function each_range(xs::Tuple{Vararg{NumberOrTensor}})
    lens = map(ncomponents, xs)
    stops = cumsum(lens)
    map((len, stop) -> StaticIndex((stop-len+1):stop), lens, stops)
end
# decompose `Vec` into multiple variables
@inline function decompose_vec(v::Vec, xs::Tuple{Vararg{NumberOrTensor}})
    rngs = each_range(xs)
    vs = map(rng -> getindex(v, rng), rngs)
    map(_reconstruct, vs, xs)
end

# for AD insertion
@inline function create_dual(::Tg, f::Number, dfdx::Number) where {Tg}
    Dual{Tg}(f, dfdx)
end
@inline function create_dual(::Tg, f::Number, dfdx::Tensor) where {Tg}
    Dual{Tg}(f, Tuple(dfdx))
end

#################
# extract value #
#################

@inline extract_value(v::NumberOrTensor) = v
@inline extract_value(v::Dual) = value(v)
@generated function extract_value(v::Tensor{S, <: Dual}) where {S <: Tuple}
    exps = [:(value(Tuple(v)[$i])) for i in 1:ncomponents(v)]
    quote
        @_inline_meta
        @inbounds Tensor{S}($(exps...))
    end
end
@inline extract_value(v::Tuple) = map(extract_value, v)

####################
# extract gradient #
####################

# Non-dual case
@inline extract_gradient(v::NumberOrTensor, ::Number) = zero(v)
@inline extract_gradient(v::Number, x::Tensor{S}) where {S} = zero(Tensor{S, typeof(v)})
@generated function extract_gradient(v::Tensor, x::Tensor)
    S = ⊗(Space(v), Space(x))
    TT = tensortype(S)
    quote
        @_inline_meta
        zero($TT{eltype(v)})
    end
end

# Dual case
@inline extract_gradient(v::Dual, ::Number, offset::Int=0) = partials(v, offset+1)
@generated function extract_gradient(v::Dual, x::Tensor{S}, offset::Int=0) where {S <: Tuple}
    exps = [:(partials(v, offset+$i)) for i in 1:ncomponents(x)]
    quote
        @_inline_meta
        @inbounds Tensor{S}(tuple($(exps...)))
    end
end
@generated function extract_gradient(v::Tensor{<: Tuple, <: Dual}, x::Tensor, offset::Int=0)
    S = ⊗(Space(v), Space(x))
    TT = tensortype(S)
    exps = [:(partials(Tuple(v)[$i], offset+$j)) for i in 1:ncomponents(v), j in 1:ncomponents(x)]
    return quote
        @_inline_meta
        @inbounds $TT($(exps...))
    end
end
@generated function extract_gradient(v::Tensor{S, <: Dual}, ::Number, offset::Int=0) where {S <: Tuple}
    exps = [:(partials(Tuple(v)[$i], offset+1)) for i in 1:ncomponents(v)]
    return quote
        @_inline_meta
        @inbounds Tensor{S}($(exps...))
    end
end
@generated extract_gradient(v::Tuple{Vararg{Any, N}}, x, args...) where {N} = :(@_inline_meta; @ntuple $N i -> extract_gradient(v[i], x, args...))

# extract_gradient for multiple arguments
ncomponents(::Number) = 1
@inline extract_gradient(v::NumberOrTensor, xs::Tuple{Vararg{NumberOrTensor}}) = map(x -> extract_gradient(v, x), xs)
@generated function extract_gradient(v::Union{Dual, Tensor{S, <: Dual}}, xs::Tuple{Vararg{NumberOrTensor, N}}) where {S <: Tuple, N}
    quote
        @_inline_meta
        offset = 0
        @nexprs $N i -> begin
            y_i = extract_gradient(v, xs[i], offset)
            offset += ncomponents(xs[i])
        end
        @ntuple $N i -> y_i
    end
end

"""
    ∂{N}

An operator representing the `N`th-order partial derivative.

`∂{N}` is callable. Applying it to a function `f` and its arguments computes
`N`th-order partial derivatives of `f` by automatic differentiation.
`∂(f, args...)` is equivalent to `∂{1}(f, args...)`.

If pseudo keyword `:all` is given as the last argument, derivatives of all orders up to `N`
are returned together with the function value.
For example, `∂{N}(f, x, :all)` returns `(∂{N}(f,x), ..., ∂{2}(f,x), ∂{1}(f,x), f(x))`.
"""
struct ∂{N} end

"""
    const gradient = ∂{1}
"""
const gradient = ∂{1}

"""
    const hessian = ∂{2}
"""
const hessian = ∂{2}

@inline _apply_dualized(f, x, ::Val{N}) where {N} = f(dualize(f, x, Val(N)))
@inline _apply_dualized(f, xs::Tuple, ::Val{N}) where {N} = f(dualize(f, xs, Val(N))...)

@generated function split_all_suffix(args::Tuple)
    args.parameters
    if !isempty(args.parameters) && args.parameters[end] <: Symbol
        expr = :((Base.front(args), :all))
    else
        expr = :((args,))
    end
    quote
        @_inline_meta
        return $expr
    end
end

@inline ∂(f, args...) = ∂{1}(f, args...)
@inline function ∂{N}(f, x) where {N}
    first(∂{N}(f, x, :all))
end
@inline function ∂{N}(f, x, ::Symbol) where {N}
    consider_symmetry(extract_all(_apply_dualized(f, x, Val(N)), x, Val(N)), x)
end
@inline function ∂{N}(f, x, y, z...) where {N}
    ∂{N}(f, split_all_suffix((x,y,z...))...)
end

@generated function extract_all(v, x, ::Val{N}) where {N}
    exps = Any[]
    for n in 0:N
        ex = :v
        for _ in 1:n
            ex = :(extract_value($ex))
        end
        for _ in 1:(N-n)
            ex = :(extract_gradient($ex, x))
        end
        push!(exps, ex)
    end
    quote
        @_inline_meta
        tuple($(exps...))
    end
end

# primitive: apply symmetry information to a single derivative block
@inline consider_symmetry(v::Number, ::Val{K}, ::Val{runs}) where {K, runs} = v
@inline consider_symmetry(v::Tuple, ::Val{K}, ::Val{runs}) where {K, runs} = map(x -> consider_symmetry(x, Val(K), Val(runs)), v)
@generated function consider_symmetry(v::TT, ::Val{K}, ::Val{runs}) where {TT <: Tensor, K, runs}
    K < 2 && return :v

    tup = Tuple(Space(TT))
    m = length(tup) - K

    pieces = Any[tup[1:m]...]
    pos = 1
    for run in runs
        s, len = run
        append!(pieces, tup[m + pos : m + s - 1])
        push!(pieces, Symmetry(tup[m + s : m + s + len - 1]))
        pos = s + len
    end
    append!(pieces, tup[m + pos : m + K])

    snew = Space(Tuple(pieces))
    exps = map(independent_to_component_map(snew)) do j
        getindex_expr(TT, :v, j)
    end
    TTnew = tensortype(snew)

    quote
        @_inline_meta
        @inbounds $TTnew(tuple($(exps...)))
    end
end

# number of tensor subspaces contributed by each input to a derivative block
nsubspaces(::Type{<:Number}) = 0
nsubspaces(::Type{TT}) where {TT <: Tensor} = length(Tuple(Space(TT)))

# single input
# symmetry reduction is only meaningful for repeated differentiation w.r.t. `Vec`.
@inline consider_symmetry(v::Tuple, x) = v
@generated function consider_symmetry(v::Tuple{Vararg{Any, M}}, ::Vec) where {M}
    exps = Expr[]
    for i in 1:M
        order = M - i
        push!(exps, :(consider_symmetry(v[$i], Val($order), Val(((1, $order),)))))
    end
    quote
        @_inline_meta
        @inbounds tuple($(exps...))
    end
end

# multiple inputs
@generated function consider_symmetry(v::Tuple{Vararg{Any, M}}, xs::Xs) where {M, Xs <: Tuple}
    xtypes = Xs.parameters
    nvars = length(xtypes)

    n = M - 1
    n < 2 && return :v

    subspaces = map(nsubspaces, xtypes)

    # Convert the differentiation history `path` into:
    #   1. the total number of derivative tensor subspaces
    #   2. repeated contiguous `Vec`-runs measured in subspace coordinates
    function subspace_runs(path)
        runs = Tuple{Int,Int}[]
        subspacepos = 1
        i = 1
        while i ≤ length(path)
            j = path[i]
            k = i + 1
            while k ≤ length(path) && path[k] == j
                k += 1
            end
            len = k - i
            s = subspaces[j]

            if len ≥ 2 && xtypes[j] <: Vec
                push!(runs, (subspacepos, len * s))
            end

            subspacepos += len * s
            i = k
        end
        return subspacepos - 1, Tuple(runs)
    end

    function build_tree(ex, depth, path=Int[])
        if depth == 0
            K, runs = subspace_runs(path)
            return :(consider_symmetry($ex, Val($K), Val($(runs))))
        else
            subs = [build_tree(:($ex[$i]), depth - 1, [path..., i]) for i in 1:nvars]
            return :(tuple($(subs...)))
        end
    end

    exps = Expr[]
    for i in 1:M
        order = M - i
        push!(exps, build_tree(:(v[$i]), order))
    end

    quote
        @_inline_meta
        @inbounds tuple($(exps...))
    end
end
