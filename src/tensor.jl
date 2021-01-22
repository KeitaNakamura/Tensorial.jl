abstract type AbstractTensor{S <: Tuple, T <: Real, N} <: AbstractArray{T, N} end

struct Tensor{S, T, N, L} <: AbstractTensor{S, T, N}
    data::NTuple{L, T}
    function Tensor{S, T, N, L}(data::NTuple{L, Any}) where {S, T, N, L}
        check_tensor_parameters(S, T, Val(N), Val(L))
        new{S, T, N, L}(convert_ntuple(T, data))
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
const SecondOrderTensor{dim, T, L} = Tensor{NTuple{2, dim}, T, 2, L}
const FourthOrderTensor{dim, T, L} = Tensor{NTuple{4, dim}, T, 4, L}
const SymmetricSecondOrderTensor{dim, T, L} = Tensor{Tuple{@Symmetry{dim, dim}}, T, 2, L}
const SymmetricFourthOrderTensor{dim, T, L} = Tensor{NTuple{2, @Symmetry{dim, dim}}, T, 4, L}
const Mat{m, n, T, L} = Tensor{Tuple{m, n}, T, 2, L}
const Vec{dim, T} = Tensor{Tuple{dim}, T, 1, dim}

# constructors
@inline function Tensor{S, T}(data::Tuple{Vararg{Any, L}}) where {S, T, L}
    N = ndims(TensorIndices(S))
    Tensor{S, T, N, L}(data)
end
@inline function Tensor{S}(data::Tuple{Vararg{Any, L}}) where {S, L}
    N = ndims(TensorIndices(S))
    T = promote_ntuple_eltype(data)
    Tensor{S, T, N, L}(data)
end
## for `tensortype` function
@inline function (::Type{Tensor{S, T, N, L} where {T}})(data::Tuple) where {S, N, L}
    T = promote_ntuple_eltype(data)
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
## from StaticArray
@inline function Tensor(A::StaticArray{S}) where {S}
    Tensor{S}(Tuple(A))
end

## for aliases
@inline function Tensor{S, T, N}(data::Tuple{Vararg{Any, L}}) where {S, T, N, L}
    Tensor{S, T, N, L}(data)
end
@inline function (::Type{Tensor{S, T, N} where {T}})(data::Tuple) where {S, N}
    T = promote_ntuple_eltype(data)
    Tensor{S, T, N}(data)
end
@inline Vec(data::Tuple{Vararg{Any, dim}}) where {dim} = Vec{dim}(data)
@inline Vec{dim}(data::Tuple) where {dim} = (T = promote_ntuple_eltype(data); Vec{dim, T}(data))

# macros
macro Vec(ex)
    esc(:(Tensor(Tensorial.@SVector $ex)))
end
macro Mat(ex)
    esc(:(Tensor(Tensorial.@SMatrix $ex)))
end
macro Tensor(ex)
    esc(:(Tensor(Tensorial.@SArray $ex)))
end

# special constructors
for (op, el) in ((:zero, :(zero(T))), (:ones, :(one(T))), (:rand, :(()->rand(T))), (:randn,:(()->randn(T))))
    @eval begin
        @inline Base.$op(::Type{Tensor{S}}) where {S} = $op(Tensor{S, Float64})
        @inline Base.$op(::Type{Tensor{S, T}}) where {S, T} = Tensor{S, T}(fill_tuple($el, Val(length(uniqueindices(S)))))
        # for aliases
        @inline Base.$op(::Type{Tensor{S, T, N}}) where {S, T, N} = $op(Tensor{S, T})
        @inline Base.$op(::Type{Tensor{S, T, N, L}}) where {S, T, N, L} = $op(Tensor{S, T})
        @inline Base.$op(::Type{Tensor{S, T, N} where {T}}) where {S, N} = $op(Tensor{S, Float64})
        @inline Base.$op(::Type{Tensor{S, T, N, L} where {T}}) where {S, N, L} = $op(Tensor{S, Float64})
    end
end
@inline Base.zero(x::Tensor) = zero(typeof(x))

# identity tensors
Base.one(::Type{Tensor{S}}) where {S} = _one(Tensor{S, Float64})
Base.one(::Type{Tensor{S, T}}) where {S, T} = _one(Tensor{S, T})
Base.one(::Type{TT}) where {TT} = one(basetype(TT))
Base.one(x::Tensor) = one(typeof(x))
@inline function _one(TT::Type{<: Union{Tensor{Tuple{dim,dim}, T}, Tensor{Tuple{@Symmetry{dim,dim}}, T}}}) where {dim, T}
    o = one(T)
    z = zero(T)
    TT((i,j) -> i == j ? o : z)
end
@inline function _one(TT::Type{Tensor{NTuple{4,dim}, T}}) where {dim, T}
    o = one(T)
    z = zero(T)
    TT((i,j,k,l) -> i == k && j == l ? o : z)
end
@inline function _one(TT::Type{Tensor{NTuple{2,@Symmetry{dim,dim}}, T}}) where {dim, T}
    o = one(T)
    z = zero(T)
    δ(i,j) = i == j ? o : z
    TT((i,j,k,l) -> (δ(i,k)*δ(j,l) + δ(i,l)*δ(j,k))/2)
end

# for AbstractArray interface
Base.IndexStyle(::Type{<: Tensor}) = IndexLinear()
@generated Base.size(::Type{TT}) where {TT <: Tensor}= :(size($(TensorIndices(TT))))
Base.size(x::Tensor) = size(typeof(x))

# helpers
Base.Tuple(x::Tensor) = x.data
ncomponents(x::Tensor) = length(Tuple(x))
ncomponents(::Type{<: Tensor{<: Any, <: Any, <: Any, L}}) where {L} = L
@pure basetype(::Type{<: Tensor{S}}) where {S} = Tensor{S}
@pure basetype(::Type{<: Tensor{S, T}}) where {S, T} = Tensor{S, T}

# indices
## TensorIndices
TensorIndices(::Tensor{S}) where {S} = TensorIndices(S)
TensorIndices(::Type{<: Tensor{S}}) where {S} = TensorIndices(S)
## serialindices
@pure serialindices(::Type{S}) where {S} = serialindices(TensorIndices(S))
@pure serialindices(::Type{<: Tensor{S}}) where {S} = serialindices(S)
serialindices(::Tensor{S}) where {S} = serialindices(S)
## uniqueindices
@pure uniqueindices(::Type{S}) where {S} = uniqueindices(TensorIndices(S))
@pure uniqueindices(::Type{<: Tensor{S}}) where {S} = uniqueindices(S)
uniqueindices(::Tensor{S}) where {S} = uniqueindices(S)
## dupsindices
@pure dupsindices(::Type{S}) where {S} = dupsindices(TensorIndices(S))
@pure dupsindices(::Type{<: Tensor{S}}) where {S} = dupsindices(S)
dupsindices(::Tensor{S}) where {S} = dupsindices(S)

# getindex
@generated function Base.getindex(x::Tensor, i::Int)
    quote
        @_inline_meta
        @boundscheck checkbounds(x, i)
        @inbounds Tuple(x)[$(serialindices(x))[i]]
    end
end

# broadcast
Broadcast.broadcastable(x::Tensor) = Ref(x)

# getindex_expr
function getindex_expr(ex::Union{Symbol, Expr}, x::Type{<: Tensor}, i...)
    inds = serialindices(x)
    :(Tuple($ex)[$(inds[i...])])
end
