@inline function dualize(x::Number)
    Dual(x, one(x))
end
@generated function dualize(x::Tensor{S, T}) where {S, T}
     dups = duplicates(x)
     ex = Expr(:block, [:($(Symbol(:v_, i)) = v_1 / $i) for i in unique(dups) if i != 1]...)
     n = ncomponents(x)
     exps = map(1:n) do i
         partials = [j == i ? Symbol(:v_, dups[i]) : :z for j in 1:n]
         :(Dual(Tuple(x)[$i], tuple($(partials...))))
     end
     quote
         @_inline_meta
         z = zero(T)
         v_1 = one(T)
         $ex
         @inbounds Tensor{S}($(exps...))
     end
 end

const NumberOrTensor = Union{Number, Tensor}

@inline extract_value(v::NumberOrTensor, ::NumberOrTensor) = v
@inline extract_value(v::Dual, ::NumberOrTensor) = value(v)
@generated function extract_value(v::Tensor{S, <: Dual}, ::NumberOrTensor) where {S <: Tuple}
    exps = [:(value(Tuple(v)[$i])) for i in 1:ncomponents(v)]
    quote
        @_inline_meta
        @inbounds Tensor{S}($(exps...))
    end
end

# Number case
@inline extract_gradient(v::NumberOrTensor, ::Number) = zero(v)
@inline extract_gradient(v::Number, x::Tensor{S}) where {S} = zero(Tensor{S, typeof(v)})
@generated function extract_gradient(v::Tensor, x::Tensor)
    S = otimes(Space(v), Space(x))
    TT = tensortype(S)
    quote
        @_inline_meta
        zero($TT{eltype(v)})
    end
end

# Dual case
@inline extract_gradient(v::Dual, ::Number) = partials(v, 1)
@inline extract_gradient(v::Dual, x::Tensor{S}) where {S} = Tensor{S}(partials(v).values)
@generated function extract_gradient(v::Tensor{<: Tuple, <: Dual}, x::Tensor)
    S = otimes(Space(v), Space(x))
    TT = tensortype(S)
    exps = [:(partials(Tuple(v)[$i], $j)) for i in 1:ncomponents(v), j in 1:ncomponents(x)]
    return quote
        @_inline_meta
        @inbounds $TT($(exps...))
    end
end
@generated function extract_gradient(v::Tensor{S, <: Dual}, ::Number) where {S <: Tuple}
    exps = [:(partials(Tuple(v)[$i], 1)) for i in 1:ncomponents(v)]
    return quote
        @_inline_meta
        @inbounds Tensor{S}($(exps...))
    end
end

function gradient(f::F, x::NumberOrTensor) where {F}
    dx = dualize(x)
    v = f(dx)
    extract_gradient(v, x)
end

function gradient(f::F, x::NumberOrTensor, ::Symbol) where {F}
    dx = dualize(x)
    v = f(dx)
    extract_gradient(v, x), extract_value(v, x)
end

function hessian(f::F, x::NumberOrTensor) where {F}
    ∇f = v -> gradient(f, v)
    gradient(∇f, x)
end

function hessian(f::F, x::NumberOrTensor, ::Symbol) where {F}
    ∇f = v -> gradient(f, v)
    gradient(∇f, x), gradient(f, x, :all)...
end
