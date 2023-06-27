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

# generate partials
dual_partials(x::Number) = (one(x),)
dual_partials(x::AbstractTensor) = convert_ntuple(eltype(x), Tuple(inv.(indices_dup(x))))

@inline function dualize(::Tg, x::Number) where {Tg}
    Dual{Tg}(x, one(x))
end
@inline function dualize(::Tg, x::AbstractTensor{S, T}) where {Tg, S, T}
    Tensor{S}(generate_duals(Tg(), dual_values(x), dual_partials(x)))
end

# for AD insertion
@inline function dualize(::Tg, f::Number, dfdx::Number) where {Tg}
    Dual{Tg}(f, dfdx)
end
@inline function dualize(::Tg, f::Number, dfdx::AbstractTensor) where {Tg}
    Dual{Tg}(f, Tuple(dfdx))
end

const NumberOrTensor = Union{Number, AbstractTensor}

@inline extract_value(v::NumberOrTensor) = v
@inline extract_value(v::Dual) = value(v)
@generated function extract_value(v::AbstractTensor{S, <: Dual}) where {S <: Tuple}
    exps = [:(value(Tuple(v)[$i])) for i in 1:ncomponents(v)]
    quote
        @_inline_meta
        @inbounds Tensor{S}($(exps...))
    end
end

# Non-dual case
@inline extract_gradient(v::NumberOrTensor, ::Number) = zero(v)
@inline extract_gradient(v::Number, x::AbstractTensor{S}) where {S} = zero(Tensor{S, typeof(v)})
@generated function extract_gradient(v::AbstractTensor, x::AbstractTensor)
    S = otimes(Space(v), Space(x))
    TT = tensortype(S)
    quote
        @_inline_meta
        zero($TT{eltype(v)})
    end
end

# Dual case
@inline extract_gradient(v::Dual, ::Number) = partials(v, 1)
@inline extract_gradient(v::Dual, x::AbstractTensor{S}) where {S} = Tensor{S}(partials(v).values)
@generated function extract_gradient(v::AbstractTensor{<: Tuple, <: Dual}, x::AbstractTensor)
    S = otimes(Space(v), Space(x))
    TT = tensortype(S)
    exps = [:(partials(Tuple(v)[$i], $j)) for i in 1:ncomponents(v), j in 1:ncomponents(x)]
    return quote
        @_inline_meta
        @inbounds $TT($(exps...))
    end
end
@generated function extract_gradient(v::AbstractTensor{S, <: Dual}, ::Number) where {S <: Tuple}
    exps = [:(partials(Tuple(v)[$i], 1)) for i in 1:ncomponents(v)]
    return quote
        @_inline_meta
        @inbounds Tensor{S}($(exps...))
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
function gradient(f, x::V) where {V <: NumberOrTensor}
    dx = dualize(Tag(f, V), x)
    v = f(dx)
    extract_gradient(v, x)
end

function gradient(f, x::V, ::Symbol) where {V <: NumberOrTensor}
    dx = dualize(Tag(f, V), x)
    v = f(dx)
    extract_gradient(v, x), extract_value(v)
end

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
function hessian(f, x::NumberOrTensor)
    ∇f = v -> gradient(f, v)
    gradient(∇f, x)
end

function hessian(f, x::NumberOrTensor, ::Symbol)
    ∇f = v -> gradient(f, v)
    gradient(∇f, x), gradient(f, x, :all)...
end
