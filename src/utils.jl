@generated function flatten_tuple(x::Tuple{Vararg{Tuple, N}}) where {N}
    exps = [Expr(:..., :(x[$i])) for i in 1:N]
    :(tuple($(exps...)))
end
