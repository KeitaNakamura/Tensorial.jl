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
