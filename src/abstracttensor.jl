abstract type AbstractTensor{S <: Tuple, T, N} <: AbstractArray{T, N} end

Base.size(::Type{TT}) where {S, TT <: AbstractTensor{S}} = Dims(Size(S))
Base.size(x::AbstractTensor) = size(typeof(x))

# indices
for func in (:independent_indices, :indices, :duplicates)
    @eval begin
        $func(::Type{TT}) where {S, TT <: AbstractTensor{S}} = $func(Size(S))
        $func(x::AbstractTensor) = $func(typeof(x))
    end
end

# getindex
@inline function Base.getindex(x::AbstractTensor, i::Int)
    @boundscheck checkbounds(x, i)
    @inbounds Tuple(x)[independent_indices(x)[i]]
end

Size(::Type{TT}) where {S, TT <: AbstractTensor{S}} = Size(S)
Size(::AbstractTensor{S}) where {S} = Size(S)

const AbstractVec{dim, T} = AbstractTensor{Tuple{dim}, T, 1}
const AbstractMat{m, n, T} = AbstractTensor{Tuple{m, n}, T, 2}