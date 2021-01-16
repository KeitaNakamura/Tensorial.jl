struct Tensor{S <: Tuple, T <: Real, N, L} <: AbstractArray{T, N}
    data::NTuple{L, T}
    function Tensor{S, T, N, L}(data::NTuple{L, T}) where {S, T, N, L}
        check_tensor_parameters(S, T, Val(N), Val(L))
        new{S, T, N, L}(data)
    end
    function Tensor{S, T, N, L}(data::NTuple{L, Any}) where {S, T, N, L}
        check_tensor_parameters(S, T, Val(N), Val(L))
        new{S, T, N, L}(map(T, data))
    end
end

@generated function check_tensor_parameters(::Type{Size}, ::Type{T}, ::Val{N}, ::Val{L}) where {Size, T, N, L}
    check_size_parameters(Size)
    if ndims(TensorIndices(Size)) != N
        return :(throw(ArgumentError("Number of dimensions must be $(ndims(TensorIndices(Size))) for $Size size, got $N.")))
    end
    if length(uniqueindices(Size)) != L
        return :(throw(ArgumentError("Length of tuple data must be $(length(uniqueindices(Size))) for $Size size, got $L.")))
    end
end

# constructors
@inline function Tensor{S, T}(data::Tuple) where {S, T}
    tensortype(TensorIndices(S)){T}(data)
end
@inline function Tensor{S}(data::Tuple) where {S}
    T = promote_type(map(eltype, data)...)
    tensortype(TensorIndices(S)){T}(data)
end
@inline function (::Type{TT})(data::Vararg{Real}) where {TT <: Tensor}
    TT(data)
end


# special constructors
for (op, el) in ((:zero, :(zero(T))), (:ones, :(one(T))), (:rand, :(()->rand(T))), (:randn,:(()->randn(T))))
    @eval begin
        @inline Base.$op(::Type{Tensor{S}}) where {S} = $op(Tensor{S, Float64})
        @inline Base.$op(::Type{Tensor{S, T}}) where {S, T} = Tensor{S, T}(fill_tuple($el, Val(length(uniqueindices(S)))))
    end
end

# for AbstractArray interface
Base.IndexStyle(::Type{<: Tensor}) = IndexLinear()
Base.size(x::Tensor) = size(serialindices(x))

# helpers
Base.Tuple(x::Tensor) = x.data
ncomponents(x::Tensor) = length(Tuple(x))

# indices
## serialindices
@pure function serialindices(::Type{S}) where {S}
    inds = serial(TensorIndices(S))
    dims = size(inds)
    SArray{Tuple{dims...}, Int}(inds)
end
serialindices(::Tensor{S}) where {S} = serialindices(S)
## uniqueindices
@pure function uniqueindices(::Type{S}) where {S}
    inds = unique(TensorIndices(S))
    dims = size(inds)
    SArray{Tuple{dims...}, Int}(inds)
end
uniqueindices(::Tensor{S}) where {S} = uniqueindices(S)

# getindex
@inline function Base.getindex(x::Tensor, i::Int)
    @boundscheck checkbounds(x, i)
    @inbounds Tuple(x)[serialindices(x)[i]]
end
