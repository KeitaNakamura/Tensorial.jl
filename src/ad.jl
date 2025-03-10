const NumberOrTensor = Union{Number, AbstractTensor}

####################
# dual generations #
####################

# generate duals from values and partials
@generated function generate_duals(::Tg, v::NTuple{N, T}, p::NTuple{N, T}) where {Tg, T, N}
    quote
        @_inline_meta
        @ntuple $N i -> begin
            partials = @ntuple $N j -> j==i ? p[i] : zero(T)
            Dual{Tg}(v[i], partials)
        end
    end
end

# generate values
dual_values(x::Number) = (x,)
dual_values(x::AbstractTensor) = Tuple(x)
dual_values(xs::Tuple{Vararg{NumberOrTensor}}) = _dual_values(promote_elements(xs...))
@generated _dual_values(xs::Tuple{Vararg{Union{T, Tensor{<: Any, T}}, N}}) where {T, N} = :(@_inline_meta; flatten_tuple(@ntuple $N i -> dual_values(xs[i])))

# generate partials
dual_partials(x::Number) = (one(x),)
dual_partials(x::AbstractTensor) = convert_ntuple(eltype(x), Tuple(inv.(nduplicates_tuple(x))))
dual_partials(xs::Tuple{Vararg{NumberOrTensor}}) = _dual_partials(promote_elements(xs...))
@generated _dual_partials(xs::Tuple{Vararg{Union{T, Tensor{<: Any, T}}, N}}) where {T, N} = :(@_inline_meta; flatten_tuple(@ntuple $N i -> dual_partials(xs[i])))

@inline function _dualize(::Tg, x::Number) where {Tg}
    Dual{Tg}(x, one(x))
end
@inline function _dualize(::Tg, x::AbstractTensor{S, T}) where {Tg, S, T}
    Tensor{S}(generate_duals(Tg(), dual_values(x), dual_partials(x)))
end
@inline dualize(f, x::NumberOrTensor) = _dualize(Tag(f, typeof(x)), x)
@inline dualize(f, x::NumberOrTensor, ::Val{0}) = x
@inline dualize(f, x::NumberOrTensor, ::Val{1}) = dualize(f, x)
@inline dualize(f, x::NumberOrTensor, ::Val{N}) where {N} = dualize(f, dualize(f, x), Val(N-1))

# for AD insertion
@inline function create_dual(::Tg, f::Number, dfdx::Number) where {Tg}
    Dual{Tg}(f, dfdx)
end
@inline function create_dual(::Tg, f::Number, dfdx::AbstractTensor) where {Tg}
    Dual{Tg}(f, Tuple(dfdx))
end

#################
# extract value #
#################

@inline extract_value(v::NumberOrTensor) = v
@inline extract_value(v::Dual) = value(v)
@generated function extract_value(v::AbstractTensor{S, <: Dual}) where {S <: Tuple}
    exps = [:(value(Tuple(v)[$i])) for i in 1:ncomponents(v)]
    quote
        @_inline_meta
        @inbounds Tensor{S}($(exps...))
    end
end

####################
# extract gradient #
####################

# Non-dual case
@inline extract_gradient(v::NumberOrTensor, ::Number) = zero(v)
@inline extract_gradient(v::Number, x::AbstractTensor{S}) where {S} = zero(Tensor{S, typeof(v)})
@generated function extract_gradient(v::AbstractTensor, x::AbstractTensor)
    S = ⊗(Space(v), Space(x))
    TT = tensortype(S)
    quote
        @_inline_meta
        zero($TT{eltype(v)})
    end
end

# Dual case
@inline extract_gradient(v::Dual, ::Number, offset::Int=0) = partials(v, offset+1)
@generated function extract_gradient(v::Dual, x::AbstractTensor{S}, offset::Int=0) where {S <: Tuple}
    exps = [:(partials(v, offset+$i)) for i in 1:ncomponents(x)]
    quote
        @_inline_meta
        @inbounds Tensor{S}(tuple($(exps...)))
    end
end
@generated function extract_gradient(v::AbstractTensor{<: Tuple, <: Dual}, x::AbstractTensor, offset::Int=0)
    S = ⊗(Space(v), Space(x))
    TT = tensortype(S)
    exps = [:(partials(Tuple(v)[$i], offset+$j)) for i in 1:ncomponents(v), j in 1:ncomponents(x)]
    return quote
        @_inline_meta
        @inbounds $TT($(exps...))
    end
end
@generated function extract_gradient(v::AbstractTensor{S, <: Dual}, ::Number, offset::Int=0) where {S <: Tuple}
    exps = [:(partials(Tuple(v)[$i], offset+1)) for i in 1:ncomponents(v)]
    return quote
        @_inline_meta
        @inbounds Tensor{S}($(exps...))
    end
end

"""
    ∂ⁿ{N}(f, x)
    ∂ⁿ{N}(f)

Differentiate the function `f` at `x` to the order `N`. `x` can be either a `Real` or a `Tensor`.
`∂` and `∂²` are aliases for `∂ⁿ{1}` and `∂ⁿ{2}`, respectively.
If only the function is provided, it creates a derivative function.
To obtain all derivatives, use `∂ⁿ{N, all}`.

```jldoctest
julia> ∂ⁿ{1}(x -> x^2 - 2x + 3, 4)
6

julia> f′ = ∂(x -> x^2 - 2x + 3)
∂{only, var"#3#4"}(var"#3#4"())

julia> f′(4)
6

julia> f, ∂f∂x = ∂{all}(x -> x^2 - 2x + 1, 4)
(9, 6)

julia> ∂²(norm, Vec(1,2,3)) ≈ ∂(x -> ∂(norm,x), Vec(1,2,3))
true
```
"""
struct ∂ⁿ{N, all, F}
    f::F
end

∂ⁿ{N}(f::F) where {N, F} = ∂ⁿ{N, only, F}(f)
∂ⁿ{N, all}(f::F) where {N, F} = ∂ⁿ{N, all, F}(f)

(F::∂ⁿ{N, only})(args...) where {N} = ∂ⁿ{N}(F.f, args...)
(F::∂ⁿ{N, all})(args...) where {N} = ∂ⁿ{N, all}(F.f, args...)

const ∂  = ∂ⁿ{1}
const ∂² = ∂ⁿ{2}

"""
    ∂(f, args...)

Differentiate the function `f` at the arguments `args`. `args` can be `Real`s and `Tensor`s.
See also [`∂ⁿ`](@ref).

```jldoctest
julia> x = rand(Mat{3,3})
3×3 Tensor{Tuple{3, 3}, Float64, 2, 9}:
 0.325977  0.894245  0.953125
 0.549051  0.353112  0.795547
 0.218587  0.394255  0.49425

julia> ∂(tr, x)
3×3 Tensor{Tuple{3, 3}, Float64, 2, 9}:
 1.0  0.0  0.0
 0.0  1.0  0.0
 0.0  0.0  1.0

julia> map(∂(*), (1,2,3), (4,5,6))
((4, 1), (5, 2), (6, 3))
```
"""
∂

@inline function ∂ⁿ{N}(f, x) where {N}
    last(∂ⁿ{N, all}(f, x))
end
@inline function ∂ⁿ{N, all}(f, x) where {N}
    consider_symmetry(extract_all(f(dualize(f, x, Val(N))), x, Val(N)), x)
end

@generated function extract_all(v, x::NumberOrTensor, ::Val{N}) where {N}
    expr = Expr(:tuple)
    for n in N:-1:0
        ex = :v
        for i in 1:n
            ex = :(extract_value($ex))
        end
        for i in 1:(N-n)
            ex = :(extract_gradient($ex, x))
        end
        push!(expr.args, ex)
    end
    quote
        @_inline_meta
        $expr
    end
end

@inline consider_symmetry(v::Tuple, x) = v
@generated function consider_symmetry(v::Tuple{Vararg{Any, M}}, x::Vec) where {M}
    N = M - 1
    N < 2 && return :v
    exps = map(2:N) do i
        TT = v.parameters[i+1]
        tup = Tuple(Space(TT))
        s = Space(tup[1:end-i]..., Symmetry(tup[end-i+1:end]))
        exps = map(tensorindices_tuple(s)) do j
            getindex_expr(TT, :(v[$(i+1)]), j)
        end
        TT_new = tensortype(s)
        :($TT_new(tuple($(exps...))))
    end
    quote
        @_inline_meta
        @inbounds (v[1], v[2], $(exps...))
    end
end

################################
# multiple-arguments interface #
################################

# when multiple arguments are given, those components are reduced to single `Vec`
# then additional decompose process is inserted before applying `f`
@inline function ∂{all}(f, x1::NumberOrTensor, x2::NumberOrTensor, x3::NumberOrTensor...)
    xs = (x1, x2, x3...)
    f★(v) = f(decompose_vec(v, xs)...)
    v, p = dual_values(xs), dual_partials(xs)
    ∇f★ = f★(Vec(generate_duals(Tag(f★, typeof(Vec(v))), v, p)))
    extract_value(∇f★), extract_gradient(∇f★, xs)
end
@inline ∂(f, x1::NumberOrTensor, x2::NumberOrTensor, x3::NumberOrTensor...) = last(∂{all}(f, x1, x2, x3...))

# decompose `Vec` into multiple variables
@generated function decompose_vec(v::Vec, xs::Tuple{Vararg{NumberOrTensor, N}}) where {N}
    @assert length(v) == sum(ncomponents, xs.parameters)
    offset = 0
    exps = []
    for i in 1:N
        TT = xs.parameters[i]
        if TT <: Number
            push!(exps, :(v[$(offset+1)]))
        else
            rng = offset .+ (1:ncomponents(TT))
            push!(exps, :($(tensortype(Space(TT)))(Tuple(@Tensor(v[$rng])))))
        end
        offset += ncomponents(TT)
    end
    quote
        @_inline_meta
        tuple($(exps...))
    end
end

# extract_gradient
@generated function extract_gradient(v::NumberOrTensor, xs::Tuple{Vararg{NumberOrTensor, N}}) where {N}
    quote
        @_inline_meta
        @ntuple $N i -> extract_gradient(v, xs[i])
    end
end
@generated function extract_gradient(v::Union{Dual, AbstractTensor{S, <: Dual}}, xs::Tuple{Vararg{NumberOrTensor, N}}) where {S <: Tuple, N}
    exps = []
    offset = 0
    for i in 1:N
        push!(exps, :(extract_gradient(v, xs[$i], $offset)))
        offset += ncomponents(xs.parameters[i])
    end
    quote
        @_inline_meta
        tuple($(exps...))
    end
end

"""
    gradient(f, x)
    gradient(f, x, :all)

Compute the gradient of `f` with respect to `x` by the automatic differentiation.
If pseudo keyword `:all` is given, the value of `f(x)` is also returned.

# Examples
```jldoctest
julia> x = rand(Mat{3,3})
3×3 Tensor{Tuple{3, 3}, Float64, 2, 9}:
 0.325977  0.894245  0.953125
 0.549051  0.353112  0.795547
 0.218587  0.394255  0.49425

julia> gradient(tr, x)
3×3 Tensor{Tuple{3, 3}, Float64, 2, 9}:
 1.0  0.0  0.0
 0.0  1.0  0.0
 0.0  0.0  1.0

julia> ∇f, f = gradient(tr, x, :all)
([1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0], 1.1733382401532275)
```
"""
@inline gradient(f, x::NumberOrTensor) = ∂(f, x)
@inline gradient(f, x::NumberOrTensor, ::Symbol) = reverse(∂{all}(f, x))

"""
    hessian(f, x)
    hessian(f, x, :all)

Compute the hessian of `f` with respect to `x` by the automatic differentiation.
If pseudo keyword `:all` is given, the value of `f(x)` is also returned.

# Examples
```jldoctest
julia> x = rand(Vec{3})
3-element Vec{3, Float64}:
 0.32597672886359486
 0.5490511363155669
 0.21858665481883066

julia> hessian(norm, x)
3×3 Tensor{Tuple{3, 3}, Float64, 2, 9}:
  1.13603   -0.582196  -0.231782
 -0.582196   0.501079  -0.390397
 -0.231782  -0.390397   1.32626

julia> ∇∇f, ∇f, f = hessian(norm, x, :all)
([1.1360324375454411 -0.5821964220304534 -0.23178236037013888; -0.5821964220304533 0.5010791569244991 -0.39039709608344814; -0.23178236037013886 -0.39039709608344814 1.3262640626479867], [0.4829957515506539, 0.8135223859352438, 0.3238771859304809], 0.6749059962060727)
```
"""
@inline hessian(f, x::NumberOrTensor) = last(extract_all(f(dualize(f, x, Val(2))), x, Val(2)))
@inline hessian(f, x::NumberOrTensor, ::Symbol) = reverse(extract_all(f(dualize(f, x, Val(2))), x, Val(2)))
