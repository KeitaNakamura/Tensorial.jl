struct Tensor{S <: Tuple, T, N, L} <: AbstractTensor{S, T, N}
    data::NTuple{L, T}
    function Tensor{S, T, N, L}(data::NTuple{L, Real}) where {S, T, N, L}
        check_tensor_parameters(S, T, Val(N), Val(L))
        new{S, T, N, L}(check_data(S, T, data))
    end
end

@generated function check_data(::Type{S}, ::Type{T}, data::NTuple{N, Any}) where {S, T, N}
    ex = quote
        @_inline_meta
        data = convert_ntuple(T, data)
    end
    if any(s -> s isa Skew, Tuple(Space(S)))
        quote
            $ex
            (zero(T), $([:(data[$i]) for i in 2:N]...))
        end
    else
        ex
    end
end

@generated function check_tensor_parameters(::Type{S}, ::Type{T}, ::Val{N}, ::Val{L}) where {S, T, N, L}
    check_size_parameters(S)
    if ndims(Space(S)) != N
        return :(throw(ArgumentError("Number of dimensions must be $(ndims(Space(S))) for $S size, got $N.")))
    end
    if data_length(Space(S)) != L
        return :(throw(ArgumentError("Length of tuple data must be $(data_length(Space(S))) for $S size, got $L.")))
    end
    if L == 0
        return :(throw(ArgumentError("Tuple data is empty. Tensors having no independent components are not supported.")))
    end
end

# aliases
const SecondOrderTensor{dim, T, L} = Tensor{NTuple{2, dim}, T, 2, L}
const FourthOrderTensor{dim, T, L} = Tensor{NTuple{4, dim}, T, 4, L}
const SymmetricSecondOrderTensor{dim, T, L} = Tensor{Tuple{@Symmetry{dim, dim}}, T, 2, L}
const SkewSymmetricSecondOrderTensor{dim, T, L} = Tensor{Tuple{@Skew{dim, dim}}, T, 2, L}
const SymmetricFourthOrderTensor{dim, T, L} = Tensor{NTuple{2, @Symmetry{dim, dim}}, T, 4, L}
const Mat{m, n, T, L} = Tensor{Tuple{m, n}, T, 2, L}
const Vec{dim, T} = Tensor{Tuple{dim}, T, 1, dim}

# constructors
@inline function Tensor{S, T}(data::Tuple{Vararg{Any, L}}) where {S, T, L}
    N = ndims(Space(S))
    Tensor{S, T, N, L}(data)
end
@inline function Tensor{S}(data::Tuple{Vararg{Any, L}}) where {S, L}
    N = ndims(Space(S))
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
@generated function (::Type{TT})(f::Function) where {TT <: Tensor}
    S = Space(TT)
    tocartesian = CartesianIndices(S)
    exps = [:(f($(Tuple(tocartesian[i])...))) for i in indices(S)]
    quote
        @_inline_meta
        TT($(exps...))
    end
end
## from AbstractArray
@generated function (::Type{TT})(A::AbstractArray) where {TT <: Tensor}
    S = Space(TT)
    if IndexStyle(A) isa IndexLinear
        exps = [:(A[$i]) for i in indices(S)]
    else
        tocartesian = CartesianIndices(S)
        exps = [:(A[$(tocartesian[i])]) for i in indices(S)]
    end
    quote
        @_inline_meta
        promote_shape($(CartesianIndices(S)), A)
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
        @inline Base.$op(::Type{Tensor{S, T}}) where {S, T} = Tensor{S, T}(fill_tuple($el, Val(data_length(Space(S)))))
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
## second order tensors
@inline function _one(TT::Type{<: Union{Tensor{Tuple{dim,dim}, T}, Tensor{Tuple{@Symmetry{dim,dim}}, T}}}) where {dim, T}
    o = one(T)
    z = zero(T)
    TT((i,j) -> i == j ? o : z)
end
## fourth order tensors
@inline function _one(TT::Type{Tensor{NTuple{4,dim}, T}}) where {dim, T}
    o = one(T)
    z = zero(T)
    TT((i,j,k,l) -> i == k && j == l ? o : z)
end
## symmetric fourth order tensors
@inline function _one(TT::Type{Tensor{NTuple{2,@Symmetry{dim,dim}}, T}}) where {dim, T}
    o = one(T)
    z = zero(T)
    δ(i,j) = i == j ? o : z
    TT((i,j,k,l) -> (δ(i,k)*δ(j,l) + δ(i,l)*δ(j,k))/2)
end
## skew symmetric tensors
@inline function _one(TT::Type{Tensor{Tuple{Skew{NTuple{dim, dim}}}, T}}) where {dim, T}
    TT((ij...) -> -one(T))
end

# for AbstractArray interface
Base.IndexStyle(::Type{<: Tensor}) = IndexLinear()

# helpers
Base.Tuple(x::Tensor) = x.data
data_length(x::Tensor) = length(Tuple(x))
data_length(::Type{<: Tensor{<: Any, <: Any, <: Any, L}}) where {L} = L
ncomponents(::Type{TT}) where {TT} = ncomponents(Space(TT))
ncomponents(x::Tensor) = ncomponents(Space(x))
@pure basetype(::Type{<: Tensor{S}}) where {S} = Tensor{S}
@pure basetype(::Type{<: Tensor{S, T}}) where {S, T} = Tensor{S, T}

# broadcast
Broadcast.broadcastable(x::Tensor) = Ref(x)

# getindex
@generated function Base.getindex(x::Tensor, i::Int)
    S = Space(x)
    inds = independent_indices(S)
    if any(s -> s isa Skew, Tuple(S))
        return quote
            @_inline_meta
            @boundscheck checkbounds(x, i)
            @inbounds begin
                I = $inds[i]
                I > 0 ? Tuple(x)[I] : -Tuple(x)[-I]
            end
        end
    else
        return quote
            @_inline_meta
            @boundscheck checkbounds(x, i)
            @inbounds Tuple(x)[$inds[i]]
        end
    end
end

# convert
@inline Base.convert(::Type{TT}, x::TT) where {TT <: Tensor} = x
@inline Base.convert(::Type{TT}, x::AbstractArray) where {TT <: Tensor} = TT(x)
@generated function Base.convert(::Type{TT}, x::AbstractTensor) where {TT <: Tensor}
    S = promote_space(Space(TT), Space(x))
    S == Space(TT) ||
        return :(throw(ArgumentError("Cannot `convert` an object of type $(typeof(x)) to an object of type $TT")))
    exps = [getindex_expr(:x, x, i) for i in indices(S)]
    quote
        @_inline_meta
        @inbounds TT(tuple($(exps...)))
    end
end

# promotion
@inline convert_eltype(::Type{T}, x::Real) where {T <: Real} = T(x)
@generated function convert_eltype(::Type{T}, x::Tensor) where {T <: Real}
    S = Space(x)
    TT = tensortype(S)
    quote
        @_inline_meta
        $TT{T}(x)
    end
end
@generated function promote_elements(xs::Vararg{Union{Real, Tensor}, N}) where {N}
    T = promote_type(eltype.(xs)...)
    exps = [:(convert_eltype($T, xs[$i])) for i in 1:N]
    quote
        @_inline_meta
        tuple($(exps...))
    end
end
