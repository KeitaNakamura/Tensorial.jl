@generated function check_size_parameters(::Type{Size}) where {Size}
    if !all(x -> isa(x, Int) || x <: Symmetry, Size.parameters)
        return :(throw(ArgumentError("Tensor parameter size must be a tuple of `Int`s or `Symmetry` (e.g. `Tensor{Tuple{3,3}}` or `Tensor{Tuple{3, Symmetry{3,3}}}`).")))
    end
end

@generated function flatten_tuple(x::Tuple{Vararg{Tuple, N}}) where {N}
    exps = [Expr(:..., :(x[$i])) for i in 1:N]
    :(tuple($(exps...)))
end

@inline fill_tuple(x, ::Val{N}) where {N} = ntuple(i -> x, Val(N))
@inline fill_tuple(f::Function, ::Val{N}) where {N} = ntuple(i -> f(), Val(N))
