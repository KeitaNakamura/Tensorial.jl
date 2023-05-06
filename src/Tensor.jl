struct Tensor{S <: Tuple, T, N, L} <: AbstractTensor{S, T, N}
    data::NTuple{L, T}
    function Tensor{S, T, N, L}(data::NTuple{L, Number}) where {S, T, N, L}
        check_tensor_parameters(S, T, Val(N), Val(L))
        new{S, T, N, L}(convert_ntuple(T, data))
    end
end

@generated function check_tensor_parameters(::Type{S}, ::Type{T}, ::Val{N}, ::Val{L}) where {S, T, N, L}
    check_size_parameters(S)
    if tensororder(Space(S)) != N
        return :(throw(ArgumentError("Number of dimensions must be $(tensororder(Space(S))) for $S size, got $N.")))
    end
    if ncomponents(Space(S)) != L
        return :(throw(ArgumentError("Length of tuple data must be $(ncomponents(Space(S))) for $S size, got $L.")))
    end
end

# aliases
const SecondOrderTensor{dim, T, L} = Tensor{NTuple{2, dim}, T, 2, L}
const FourthOrderTensor{dim, T, L} = Tensor{NTuple{4, dim}, T, 4, L}
const SymmetricSecondOrderTensor{dim, T, L} = Tensor{Tuple{@Symmetry({dim, dim})}, T, 2, L}
const SymmetricFourthOrderTensor{dim, T, L} = Tensor{NTuple{2, @Symmetry({dim, dim})}, T, 4, L}
const Mat{m, n, T, L} = Tensor{Tuple{m, n}, T, 2, L}
const Vec{dim, T} = Tensor{Tuple{dim}, T, 1, dim}

# constructors
@inline function Tensor{S, T}(data::Tuple{Vararg{Any, L}}) where {S, T, L}
    N = tensororder(Space(S))
    Tensor{S, T, N, L}(data)
end
@inline function Tensor{S}(data::Tuple{Vararg{Any, L}}) where {S, L}
    N = tensororder(Space(S))
    T = promote_ntuple_eltype(data)
    Tensor{S, T, N, L}(data)
end
## for `tensortype` function
@inline function (::Type{Tensor{S, T, N, L} where {T}})(data::Tuple) where {S, N, L}
    T = promote_ntuple_eltype(data)
    Tensor{S, T, N, L}(data)
end
## from Vararg
@inline function (::Type{TT})(data::Vararg{Number}) where {TT <: Tensor}
    TT(data)
end
## from Function
@generated function (::Type{TT})(f::Function) where {TT <: Tensor}
    S = Space(TT)
    tocartesian = CartesianIndices(tensorsize(S))
    exps = [:(f($(Tuple(tocartesian[i])...))) for i in indices_unique(S)]
    quote
        @_inline_meta
        TT($(exps...))
    end
end
## from AbstractArray
@generated function (::Type{TT})(A::AbstractArray) where {TT <: Tensor}
    S = Space(TT)
    if IndexStyle(A) isa IndexLinear
        exps = [:(A[$i]) for i in indices_unique(S)]
    else
        tocartesian = CartesianIndices(tensorsize(S))
        exps = [:(A[$(tocartesian[i])]) for i in indices_unique(S)]
    end
    quote
        @_inline_meta
        @boundscheck if length(TT) != length(A)
            throw(DimensionMismatch("expected input array of length $(length(A)), got length $(length(TT))"))
        end
        @inbounds TT($(exps...))
    end
end
## from StaticArray
@inline function Tensor(A::StaticArray{S, T}) where {S, T}
    Tensor{S, T}(Tuple(A))
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
@inline Vec(x::AbstractVec) = vec(x)

# macros
macro Vec(ex)
    esc(:(Tensor(Tensorial.@SVector $ex)))
end
macro Mat(ex)
    esc(:(Tensor(Tensorial.@SMatrix $ex)))
end
macro Tensor(expr)
    if Meta.isexpr(expr, :braces)
        esc(:(Tensor{Tuple{$(expr.args...)}}))
    elseif Meta.isexpr(expr, :ref)
        newargs = map(expr.args) do ex
            if Meta.isexpr(ex, :call) && ex.args[1] == :(:)
                :($StaticIndex($ex))
            elseif Meta.isexpr(ex, :vect)
                if all(x -> x isa Int, ex.args) # static indexing
                    :($StaticIndex($SVector($(ex.args...))))
                else
                    Expr(:call, :($SVector), ex.args...)
                end
            else
                ex
            end
        end
        esc(Expr(:block, :($check_Tensor_macro($(newargs[1]))), Expr(expr.head, newargs...)))
    else
        esc(:(Tensor(Tensorial.@SArray $expr)))
    end
end
check_Tensor_macro(x::AbstractTensor) = nothing
check_Tensor_macro(x) = throw(ArgumentError("$(typeof(x)) is not supported in @Tensor"))

# special constructors
for (op, el) in ((:zero, :(zero(T))), (:ones, :(one(T))), (:rand, :(()->rand(T))), (:randn,:(()->randn(T))))
    @eval begin
        @inline Base.$op(::Type{Tensor{S}}) where {S} = $op(Tensor{S, Float64})
        @inline Base.$op(::Type{Tensor{S, T}}) where {S, T} = Tensor{S, T}(fill_tuple($el, Val(ncomponents(Space(S)))))
        # for aliases
        @inline Base.$op(::Type{Tensor{S, T, N}}) where {S, T, N} = $op(Tensor{S, T})
        @inline Base.$op(::Type{Tensor{S, T, N, L}}) where {S, T, N, L} = $op(Tensor{S, T})
        @inline Base.$op(::Type{Tensor{S, T, N} where {T}}) where {S, N} = $op(Tensor{S, Float64})
        @inline Base.$op(::Type{Tensor{S, T, N, L} where {T}}) where {S, N, L} = $op(Tensor{S, Float64})
        @inline Base.$op(x::Tensor) = $op(typeof(x))
    end
end

# identity tensors
_one(::Type{Tensor{S}}) where {S} = __one(Tensor{S, Float64})
_one(::Type{Tensor{S, T}}) where {S, T} = __one(Tensor{S, T})
Base.one(::Type{TT}) where {TT <: Tensor} = _one(basetype(TT))
Base.one(x::Tensor) = one(typeof(x))
@inline function __one(TT::Type{<: Union{Tensor{Tuple{dim,dim}, T}, Tensor{Tuple{@Symmetry({dim,dim})}, T}}}) where {dim, T}
    o = one(T)
    z = zero(T)
    TT((i,j) -> i == j ? o : z)
end
@inline function __one(TT::Type{Tensor{NTuple{4,dim}, T}}) where {dim, T}
    o = one(T)
    z = zero(T)
    TT((i,j,k,l) -> i == k && j == l ? o : z)
end
@inline function __one(TT::Type{Tensor{NTuple{2,@Symmetry({dim,dim})}, T}}) where {dim, T}
    o = one(T)
    z = zero(T)
    δ(i,j) = i == j ? o : z
    TT((i,j,k,l) -> (δ(i,k)*δ(j,l) + δ(i,l)*δ(j,k))/2)
end

"""
    levicivita(::Val{N} = Val(3))

Return `N` dimensional Levi-Civita tensor.

# Examples
```jldoctest
julia> ϵ = levicivita()
3×3×3 Tensor{Tuple{3, 3, 3}, Int64, 3, 27}:
[:, :, 1] =
 0   0  0
 0   0  1
 0  -1  0

[:, :, 2] =
 0  0  -1
 0  0   0
 1  0   0

[:, :, 3] =
  0  1  0
 -1  0  0
  0  0  0
```
"""
@pure function levicivita(::Val{dim} = Val(3)) where {dim}
    Tensor{NTuple{dim, dim}, Int}(sgn)
end
@pure function sgn(x::Int...)
    N = length(x)
    even = true
    @inbounds for i in 1:N, j in i+1:N
        x[i] == x[j] && return 0
        even ⊻= x[i] > x[j]
    end
    even ? 1 : -1
end

# UniformScaling
@inline function (TT::Type{<: Tensor})(I::UniformScaling)
    TT((i,j) -> I[i,j])
end

# helpers
Base.Tuple(x::Tensor) = x.data
@pure basetype(::Type{<: Tensor{S}}) where {S} = Tensor{S}
@pure basetype(::Type{<: Tensor{S, T}}) where {S, T} = Tensor{S, T}

# getindex
@inline function Base.getindex(x::Tensor, i::Int)
    @boundscheck checkbounds(x, i)
    @inbounds Tuple(x)[indices_all(x)[i]]
end

# convert
@inline Base.convert(::Type{TT}, x::TT) where {TT <: Tensor} = x
@inline Base.convert(::Type{TT}, x::AbstractArray) where {TT <: Tensor} = TT(x)
@generated function Base.convert(::Type{TT}, x::AbstractTensor) where {TT <: Tensor}
    S = promote_space(Space(TT), Space(x))
    S == Space(TT) ||
        return :(throw(ArgumentError("Cannot `convert` an object of type $(typeof(x)) to an object of type $TT")))
    exps = [getindex_expr(x, :x, i) for i in indices_unique(S)]
    quote
        @_inline_meta
        @inbounds TT(tuple($(exps...)))
    end
end
@inline Base.convert(::Type{TT}, x::Tuple) where {TT <: Tensor} = TT(x)

# promotion
@inline convert_eltype(::Type{T}, x::Number) where {T <: Number} = convert(T, x)
@generated function convert_eltype(::Type{T}, x::Tensor) where {T <: Number}
    TT = tensortype(Space(x))
    quote
        @_inline_meta
        convert($TT{T}, x)
    end
end
@generated function promote_elements(xs::Vararg{Union{Number, Tensor}, N}) where {N}
    T = promote_type(eltype.(xs)...)
    exps = [:(convert_eltype($T, xs[$i])) for i in 1:N]
    quote
        @_inline_meta
        tuple($(exps...))
    end
end

function Base.promote_rule(::Type{Tensor{S, T, N, L}}, ::Type{Tensor{S, U, N, L}}) where {S, T, U, N, L}
    Tensor{S, promote_type(T, U), N, L}
end
