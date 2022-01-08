@generated function Base.vec(x::AbstractTensor)
    N = length(x)
    exps = [getindex_expr(x, :x, i) for i in 1:N]
    quote
        @_inline_meta
        @inbounds Vec($(exps...))
    end
end

@generated function resize(x::AbstractTensor{<: Any, <: Any, N}, inds::Vararg{Val, N}) where {N}
    exps = [Base.OneTo(only(I.parameters)) for I in inds]
    colons = [Colon() for _ in 1:ndims(x)]
    quote
        newspace = _getindex(Space(x), Val(tuple($(exps...))))
        _getindex(newspace, x, $(colons...))
    end
end
resizedim(x::Tensor, ::Val{dim}) where {dim} = resize(x, ntuple(i -> Val(dim), Val(ndims(x)))...)
