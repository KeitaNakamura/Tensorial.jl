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
    T = promote_type(eltype(x1), eltype(x2))
    if ndims(t) == 0
        TT = T
    else
        TT = tensortype(t){T}
    end
    quote
        @_inline_meta
        @inbounds $TT($(exps...))
    end
end

@inline otimes(x1::Tensor, x2::Tensor) = contract(x1, x2, Val(0))
@inline dot(x1::Tensor, x2::Tensor) = contract(x1, x2, Val(1))
@inline dcontract(x1::Tensor, x2::Tensor) = contract(x1, x2, Val(2))
