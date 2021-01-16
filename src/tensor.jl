struct Tensor{S <: Tuple, T, N, L} <: AbstractArray{T, N}
    data::NTuple{L, T}
end

Base.IndexStyle(::Type{<: Tensor}) = IndexLinear()
Base.size(x::Tensor) = size(serialindices(x))

Base.Tuple(x::Tensor) = x.data

ncomponents(x::Tensor) = length(Tuple(x))

@pure function serialindices(::Type{S}) where {S}
    inds = serial(TensorIndices(S))
    dims = size(inds)
    SArray{Tuple{dims...}, Int}(inds)
end
serialindices(::Tensor{S}) where {S} = serialindices(S)

@pure function uniqueindices(::Type{S}) where {S}
    inds = unique(TensorIndices(S))
    dims = size(inds)
    SArray{Tuple{dims...}, Int}(inds)
end
uniqueindices(::Tensor{S}) where {S} = uniqueindices(S)

@inline function Base.getindex(x::Tensor, i::Int)
    @boundscheck checkbounds(x, i)
    @inbounds Tuple(x)[serialindices(x)[i]]
end

function _reduce(x::Vector{Expr})
    out = Expr[]
    ndups = ones(Int, length(x))
    for ex in x
        i = findfirst(isequal(ex), out)
        if i === nothing
            push!(out, ex)
        else
            ndups[i] += 1
        end
    end
    for i in 1:length(out)
        ndups[i] != 1 && (out[i] = :($(ndups[i]) * $(out[i])))
    end
    Expr(:call, :+, out...)
end

@generated function contract(x1::Tensor{S1}, x2::Tensor{S2}, ::Val{N}) where {S1, S2, N}
    t = contract(TensorIndices(S1), TensorIndices(S2), Val(N))
    s1 = serialindices(S1)
    s2 = serialindices(S2)
    J = prod(size(s2)[1:N])
    I = length(s1) ÷ J
    K = length(s2) ÷ J
    s1′ = reshape(s1, I, J)
    s2′ = reshape(s2, J, K)
    s′ = Array{Vector{Expr}}(undef, I, K)
    for k in 1:K, i in 1:I
        s′[i,k] = [:(Tuple(x1)[$(s1′[i,j])] * Tuple(x2)[$(s2′[j,k])]) for j in 1:J]
    end
    s = reshape(s′, size(s1)[1:end-N]..., size(s2)[N+1:end]...)
    exps = map(_reduce, s[unique(t)])
    TT = tensortype(t)
    quote
        @_inline_meta
        T = promote_type(eltype(x1), eltype(x2))
        @inbounds $TT{T}(($(exps...),))
    end
end

@inline otimes(x1::Tensor, x2::Tensor) = contract(x1, x2, Val(0))
