@inline function dualize(::Tg, x::Number) where {Tg}
    Dual{Tg}(x, one(x))
end
@generated function dualize(::Tg, x::AbstractTensor{S, T}) where {Tg, S, T}
    dups = indices_dup(x)
    ex = Expr(:block, [:($(Symbol(:v_, i)) = v_1 / $i) for i in unique(dups) if i != 1]...)
    n = ncomponents(x)
    exps = map(1:n) do i
        partials = [j == i ? Symbol(:v_, dups[i]) : :z for j in 1:n]
        :(Dual{Tg}(Tuple(x)[$i], tuple($(partials...))))
    end
    quote
        @_inline_meta
        z = zero(T)
        v_1 = one(T)
        $ex
        @inbounds Tensor{S}($(exps...))
    end
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

# Number case
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

function hessian(f, x::NumberOrTensor)
    ∇f = v -> gradient(f, v)
    gradient(∇f, x)
end

function hessian(f, x::NumberOrTensor, ::Symbol)
    ∇f = v -> gradient(f, v)
    gradient(∇f, x), gradient(f, x, :all)...
end
