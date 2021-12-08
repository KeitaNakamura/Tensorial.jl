@generated function Base.vec(x::AbstractTensor)
    N = length(x)
    exps = [getindex_expr(x, :x, i) for i in 1:N]
    quote
        @_inline_meta
        @inbounds Vec($(exps...))
    end
end
