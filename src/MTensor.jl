mutable struct MTensor{S <: Tuple, T, N, L} <: AbstractTensor{S, T, N}
    data::NTuple{L, T}
    function MTensor{S, T, N, L}(data::NTuple{L, Real}) where {S, T, N, L}
        check_tensor_parameters(S, T, Val(N), Val(L))
        new{S, T, N, L}(convert_ntuple(T, data))
    end
end

@inline MTensor(x::Tensor{S, T, N, L}) where {S, T, N, L} = MTensor{S, T, N, L}(Tuple(x))

@generated function to_tensortype(::Type{TT}) where {TT <: MTensor}
    types = getvartypes((), TT)
    ex = :(Tensor{$(getparemters(TT)...)})
    for t in types
        ex = :(UnionAll($t, $ex))
    end
    quote
        @_inline_meta
        $ex
    end
end
getparemters(x::UnionAll) = getparemters(x.body)
getparemters(x::DataType) = x.parameters
getvartypes(types::Tuple, x::UnionAll) = getvartypes((x.var, types...), x.body)
getvartypes(types::Tuple, x::DataType) = types

# aliases
const MSecondOrderTensor{dim, T, L} = MTensor{NTuple{2, dim}, T, 2, L}
const MFourthOrderTensor{dim, T, L} = MTensor{NTuple{4, dim}, T, 4, L}
const MSymmetricSecondOrderTensor{dim, T, L} = MTensor{Tuple{@Symmetry{dim, dim}}, T, 2, L}
const MSymmetricFourthOrderTensor{dim, T, L} = MTensor{NTuple{2, @Symmetry{dim, dim}}, T, 4, L}
const MMat{m, n, T, L} = MTensor{Tuple{m, n}, T, 2, L}
const MVec{dim, T} = MTensor{Tuple{dim}, T, 1, dim}

# macros
macro MVec(ex)
    esc(:(MTensor(Tensor(Tensorial.@SVector $ex))))
end
macro MMat(ex)
    esc(:(MTensor(Tensor(Tensorial.@SMatrix $ex))))
end
macro MTensor(ex)
    esc(:(MTensor(Tensor(Tensorial.@SArray $ex))))
end

# special constructors
for op in (:zero, :ones, :rand, :randn, :one)
    @eval @inline Base.$op(::Type{TT}) where {TT <: MTensor} = MTensor($op(to_tensortype(TT)))
end
@inline Base.zero(x::MTensor) = zero(typeof(x))
@inline Base.one(x::MTensor) = one(typeof(x))

# for AbstractArray interface
Base.IndexStyle(::Type{<: MTensor}) = IndexLinear()

# helpers
Base.Tuple(x::MTensor) = x.data
ncomponents(x::MTensor) = length(Tuple(x))
ncomponents(::Type{<: MTensor{<: Any, <: Any, <: Any, L}}) where {L} = L

# broadcast
Broadcast.broadcastable(x::MTensor) = Ref(x)

# getindex/setindex!
@inline function Base.getindex(x::MTensor, i::Int)
    @boundscheck checkbounds(x, i)
    @inbounds I = independent_indices(x)[i]
    T = eltype(x)
    if isbitstype(T)
        return GC.@preserve x unsafe_load(Base.unsafe_convert(Ptr{T}, pointer_from_objref(x)), I)
    end
    @inbounds Tuple(x)[I]
end
@inline function Base.setindex!(x::MTensor, val, i::Int)
    @boundscheck checkbounds(x, i)
    @inbounds I = independent_indices(x)[i]
    T = eltype(x)
    if isbitstype(T)
        GC.@preserve x unsafe_store!(Base.unsafe_convert(Ptr{T}, pointer_from_objref(x)), convert(T, val), I)
    else
        # This one is unsafe (#27)
        # unsafe_store!(Base.unsafe_convert(Ptr{Ptr{Nothing}}, pointer_from_objref(x.data)), pointer_from_objref(val), I)
        error("setindex!() with non-isbitstype eltype is not supported by Tensorial.jl")
    end
    x
end

# convert
@inline Base.convert(::Type{TT}, x::AbstractArray) where {TT <: MTensor} = MTensor(to_tensortype(TT)(x))
