@inline function dualize(x::Real)
    Dual(x, one(x))
end
@generated function dualize(x::Tensor{S, T}) where {S, T}
     dups = dupsindices(x)
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

const RealOrTensor{T <: Real} = Union{T, Tensor{S, T} where {S}}

@inline extract_value(v::RealOrTensor, ::RealOrTensor) = v
@inline extract_value(v::Dual, ::RealOrTensor) = value(v)
@generated function extract_value(v::Tensor{S, <: Dual}, ::RealOrTensor) where {S}
    exps = [:(value(Tuple(v)[$i])) for i in 1:ncomponents(v)]
    quote
        @_inline_meta
        @inbounds Tensor{S}($(exps...))
    end
end

# Real case
@inline extract_gradient(v::RealOrTensor, ::Real) = zero(v)
@inline extract_gradient(v::Real, x::Tensor{S}) where {S} = zero(Tensor{S, typeof(v)})
@generated function extract_gradient(v::Tensor{S}, x::Tensor) where {S}
    inds = otimes(TensorIndices(v), TensorIndices(x))
    TT = tensortype(inds)
    quote
        @_inline_meta
        zero($TT{eltype(v)})
    end
end

# Dual case
@inline extract_gradient(v::Dual, ::Real) = partials(v, 1)
@inline extract_gradient(v::Dual, x::Tensor{S}) where {S} = Tensor{S}(partials(v).values)
@generated function extract_gradient(v::Tensor{S, <: Dual}, x::Tensor) where {S}
    inds = otimes(TensorIndices(v), TensorIndices(x))
    TT = tensortype(inds)
    exps = [:(partials(Tuple(v)[$i], $j)) for i in 1:ncomponents(v), j in 1:ncomponents(x)]
    return quote
        @_inline_meta
        @inbounds $TT($(exps...))
    end
end
@generated function extract_gradient(v::Tensor{S, <: Dual}, ::Real) where {S}
    exps = [:(partials(Tuple(v)[$i], 1)) for i in 1:ncomponents(v)]
    return quote
        @_inline_meta
        @inbounds Tensor{S}($(exps...))
    end
end

function gradient(f::F, x::RealOrTensor) where {F}
    dx = dualize(x)
    v = f(dx)
    extract_gradient(v, x)
end

function gradient(f::F, x::RealOrTensor, ::Symbol) where {F}
    dx = dualize(x)
    v = f(dx)
    extract_gradient(v, x), extract_value(v, x)
end

function hessian(f::F, x::RealOrTensor) where {F}
    ∇f = v -> gradient(f, v)
    gradient(∇f, x)
end

function hessian(f::F, x::RealOrTensor, ::Symbol) where {F}
    ∇f = v -> gradient(f, v)
    gradient(∇f, x), gradient(f, x, :all)...
end
