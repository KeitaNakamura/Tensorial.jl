@generated function check_size_parameters(::Type{S}) where {S}
    if !all(x -> isa(x, Int) || x <: Symmetry, S.parameters)
        return :(throw(ArgumentError("Tensor's size parameter must be a tuple of `Int`s or `Symmetry` (e.g. `Tensor{Tuple{3,3}}` or `Tensor{Tuple{3, @Symmetry{3,3}}}`).")))
    end
end

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

tuple_sort(x::NTuple{N, Any}; kwargs...) where {N} = Tuple(sort(SVector(x); kwargs...))


if VERSION < v"1.4"
    Base.@propagate_inbounds function only(x)
        i = iterate(x)
        @boundscheck if i === nothing
            throw(ArgumentError("Collection is empty, must contain exactly 1 element"))
        end
        (ret, state) = i
        @boundscheck if iterate(x, state) !== nothing
            throw(ArgumentError("Collection has multiple elements, must contain exactly 1 element"))
        end
        return ret
    end

    # Collections of known size
    only(x::Ref) = x[]
    only(x::Number) = x
    only(x::Char) = x
    only(x::Tuple{Any}) = x[1]
    only(x::Tuple) = throw(
        ArgumentError("Tuple contains $(length(x)) elements, must contain exactly 1 element")
    )
    only(a::AbstractArray{<:Any, 0}) = @inbounds return a[]
    only(x::NamedTuple{<:Any, <:Tuple{Any}}) = first(x)
    only(x::NamedTuple) = throw(
        ArgumentError("NamedTuple contains $(length(x)) elements, must contain exactly 1 element")
    )
end
