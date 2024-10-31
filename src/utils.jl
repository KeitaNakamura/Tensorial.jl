@generated function flatten_tuple(x::Tuple{Vararg{Tuple, N}}) where {N}
    exps = [Expr(:..., :(x[$i])) for i in 1:N]
    :(tuple($(exps...)))
end

@inline fill_tuple(x, ::Val{N}) where {N} = ntuple(i -> x, Val(N))
@inline fill_tuple(f::Function, ::Val{N}) where {N} = ntuple(i -> f(), Val(N))

@inline convert_ntuple(::Type{T}, x::NTuple{N, T}) where {N, T} = x
@generated function convert_ntuple(::Type{T}, x::NTuple{N, Any}) where {N, T}
    exps = [:(convert(T, x[$i])) for i in 1:N]
    quote
        @_inline_meta
        tuple($(exps...))
    end
end

@inline promote_ntuple_eltype(x::NTuple{N, T}) where {N, T} = T
@generated function promote_ntuple_eltype(x::NTuple{N, Any}) where {N}
    T = promote_type(x.parameters...)
    quote
        @_inline_meta
        $T
    end
end
