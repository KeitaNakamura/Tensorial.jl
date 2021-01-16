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

# aliases
const SecondOrderTensor{dim, T <: Real, L} = Tensor{NTuple{2, dim}, T, 2, L}
const ThirdOrderTensor{dim, T <: Real, L} = Tensor{NTuple{3, dim}, T, 3, L}
const FourthOrderTensor{dim, T <: Real, L} = Tensor{NTuple{4, dim}, T, 4, L}
const SymmetricSecondOrderTensor{dim, T <: Real, L} = Tensor{Tuple{@Symmetry{dim, dim}}, T, 2, L}
const SymmetricThirdOrderTensor{dim, T <: Real, L} = Tensor{Tuple{@Symmetry{dim, dim, dim}}, T, 3, L}
const SymmetricFourthOrderTensor{dim, T <: Real, L} = Tensor{NTuple{2, @Symmetry{dim, dim}}, T, 4, L}
const Vec{dim, T <: Real} = Tensor{Tuple{dim}, T, 1, dim}
const Mat{m, n, T <: Real, L} = Tensor{Tuple{m, n}, T, 2, L}

# constructors
@inline function Tensor{S, T}(data::Tuple{Vararg{Any, L}}) where {S, T, L}
    N = ndims(TensorIndices(S))
    Tensor{S, T, N, L}(data)
end
@inline function Tensor{S}(data::Tuple{Vararg{Any, L}}) where {S, L}
    N = ndims(TensorIndices(S))
    T = promote_type(map(eltype, data)...)
    Tensor{S, T, N, L}(data)
end
## for `tensortype` function
@inline function (::Type{Tensor{S, T, N, L} where {T <: Real}})(data::Tuple) where {S, N, L}
    T = promote_type(map(eltype, data)...)
    Tensor{S, T, N, L}(data)
end
## from Vararg
@inline function (::Type{TT})(data::Vararg{Real}) where {TT <: Tensor}
    TT(data)
end
## from Function
@generated function (::Type{TT})(f::Function) where {S, TT <: Tensor{S}}
    tocartesian = CartesianIndices(TensorIndices(S))
    exps = [:(f($(Tuple(tocartesian[i])...))) for i in uniqueindices(S)]
    quote
        @_inline_meta
        TT($(exps...))
    end
end
## from AbstractArray
@generated function (::Type{TT})(A::AbstractArray) where {S, TT <: Tensor{S}}
    inds = TensorIndices(S)
    if IndexStyle(A) isa IndexLinear
        exps = [:(A[$i]) for i in uniqueindices(S)]
    else
        tocartesian = CartesianIndices(inds)
        exps = [:(A[$(tocartesian[i])]) for i in uniqueindices(S)]
    end
    quote
        @_inline_meta
        promote_shape($inds, A)
        @inbounds TT($(exps...))
    end
end

## for aliases
@inline function Tensor{S, T, N}(data::Tuple{Vararg{Any, L}}) where {S, T, N, L}
    Tensor{S, T, N, L}(data)
end
@inline function (::Type{Tensor{S, T, N} where {T <: Real}})(data::Tuple) where {S, N}
    T = promote_type(map(eltype, data)...)
    Tensor{S, T, N}(data)
end
@inline Vec(data::Tuple{Vararg{Any, dim}}) where {dim} = Vec{dim}(data)
@inline Vec{dim}(data::Tuple) where {dim} = (T = promote_type(map(eltype, data)...); Vec{dim, T}(data))

# special constructors
for (op, el) in ((:zero, :(zero(T))), (:ones, :(one(T))), (:rand, :(()->rand(T))), (:randn,:(()->randn(T))))
    @eval begin
        @inline Base.$op(::Type{Tensor{S}}) where {S} = $op(Tensor{S, Float64})
        @inline Base.$op(::Type{Tensor{S, T}}) where {S, T} = Tensor{S, T}(fill_tuple($el, Val(length(uniqueindices(S)))))
        # for aliases
        @inline Base.$op(::Type{Tensor{S, T, N}}) where {S, T, N} = $op(Tensor{S, T})
        @inline Base.$op(::Type{Tensor{S, T, N, L}}) where {S, T, N, L} = $op(Tensor{S, T})
        @inline Base.$op(::Type{Tensor{S, T, N} where {T <: Real}}) where {S, N} = $op(Tensor{S, Float64})
        @inline Base.$op(::Type{Tensor{S, T, N, L} where {T <: Real}}) where {S, N, L} = $op(Tensor{S, Float64})
    end
end

# identity tensors
for TensorType in (SecondOrderTensor,
                   FourthOrderTensor,
                   SymmetricSecondOrderTensor,
                   SymmetricFourthOrderTensor)
    @eval Base.one(::Type{$TensorType{dim}}) where {dim} = one($TensorType{dim, Float64})
    @eval Base.one(x::$TensorType) = one(typeof(x))
end
@inline function Base.one(TT::Type{<: Union{SecondOrderTensor{dim, T}, SymmetricSecondOrderTensor{dim, T}}}) where {dim, T}
    o = one(T)
    z = zero(T)
    TT((i,j) -> i == j ? o : z)
end
@inline function Base.one(TT::Type{<: FourthOrderTensor{dim, T}}) where {dim, T}
    o = one(T)
    z = zero(T)
    TT((i,j,k,l) -> i == k && j == l ? o : z)
end
@inline function Base.one(TT::Type{<: SymmetricFourthOrderTensor{dim, T}}) where {dim, T}
    o = one(T)
    z = zero(T)
    δ(i,j) = i == j ? o : z
    TT((i,j,k,l) -> (δ(i,k)*δ(j,l) + δ(i,l)*δ(j,k))/2)
end

# for AbstractArray interface
Base.IndexStyle(::Type{<: Tensor}) = IndexLinear()
Base.size(x::Tensor) = size(serialindices(x))

# helpers
Base.Tuple(x::Tensor) = x.data
ncomponents(x::Tensor) = length(Tuple(x))
ncomponents(::Type{<: Tensor{<: Any, <: Any, <: Any, L}}) where {L} = L

# indices
## TensorIndices
TensorIndices(::Tensor{S}) where {S} = TensorIndices(S)
TensorIndices(::Type{<: Tensor{S}}) where {S} = TensorIndices(S)
## serialindices
@pure function serialindices(::Type{S}) where {S}
    inds = serial(TensorIndices(S))
    dims = size(inds)
    SArray{Tuple{dims...}, Int}(inds)
end
serialindices(::Tensor{S}) where {S} = serialindices(S)
@pure serialindices(::Type{<: Tensor{S}}) where {S} = serialindices(S)
@pure serialindices(inds::TensorIndices) = serialindices(tensortype(inds))
## uniqueindices
@pure function uniqueindices(::Type{S}) where {S}
    inds = unique(TensorIndices(S))
    dims = size(inds)
    SArray{Tuple{dims...}, Int}(inds)
end
uniqueindices(::Tensor{S}) where {S} = uniqueindices(S)
@pure uniqueindices(::Type{<: Tensor{S}}) where {S} = uniqueindices(S)
@pure uniqueindices(inds::TensorIndices) = uniqueindices(tensortype(inds))
## dupsindices
@pure function dupsindices(::Type{S}) where {S}
    inds = dups(TensorIndices(S))
    dims = size(inds)
    SArray{Tuple{dims...}, Int}(inds)
end
dupsindices(::Tensor{S}) where {S} = dupsindices(S)
@pure dupsindices(::Type{<: Tensor{S}}) where {S} = dupsindices(S)
@pure dupsindices(inds::TensorIndices) = dupsindices(tensortype(inds))

# getindex
@inline function Base.getindex(x::Tensor, i::Int)
    @boundscheck checkbounds(x, i)
    @inbounds Tuple(x)[serialindices(x)[i]]
end

# broadcast
Broadcast.broadcastable(x::Tensor) = Ref(x)

# getindex_expr
function getindex_expr(ex::Union{Symbol, Expr}, x::Type{<: Tensor}, i...)
    inds = serialindices(x)
    :(Tuple($ex)[$(inds[i...])])
end
