import Base.Broadcast: BroadcastStyle, DefaultArrayStyle, AbstractArrayStyle, Broadcasted, broadcastable

struct TensorStyle{N} <: AbstractArrayStyle{N} end

# Tensorial's `TensorStyle` broadcast is intentionally limited to componentwise
# operations that produce another tensor: tensor/tensor, tensor/scalar, and
# tensor/tuple. Ordinary arrays are delegated to `Base` via `DefaultArrayStyle`.
BroadcastStyle(::Type{<: AbstractTensor{<: Tuple, <: Any, N}}) where {N} = TensorStyle{N}()
# BroadcastStyle(::Type{<: AbstractMatLike}) = TensorStyle{2}() # shoud support this?

BroadcastStyle(::TensorStyle{M}, b::DefaultArrayStyle{N}) where {M, N} = DefaultArrayStyle(Val(max(M, N)))
BroadcastStyle(::TensorStyle{M}, ::DefaultArrayStyle{0}) where {M} = TensorStyle{M}()

BroadcastStyle(a::TensorStyle, ::Broadcast.Style{Tuple}) = a

broadcastable(bc::Broadcasted{<: TensorStyle}) = copy(bc)

@generated function _promote_space_for_broadcast(x::Tuple)
    spaces = [Space(t) for t in x.parameters if t <: AbstractTensor]
    promote_space(spaces...)
end

@inline _tensor_broadcast_arg(::Type{TT}, x::Any) where {TT} = x
@inline _tensor_broadcast_arg(::Type{TT}, x::AbstractTensor) where {TT} = Tuple(convert(TT, x))

# Since array arguments are handled by `DefaultArrayStyle`, `Any` here means a
# scalar broadcast argument such as a number or bool.
@inline _tensor_broadcast_getindex(x::Any, ::Val{i}) where {i} = x
@inline _tensor_broadcast_getindex(x::Base.RefValue, ::Val{i}) where {i} = x[]
@inline _tensor_broadcast_getindex(x::Tuple, ::Val{i}) where {i} = length(x) == 1 ? x[1] : x[i]
# `@.` can turn scalar subexpressions like `2a` into 0-dimensional broadcasts.
@inline _tensor_broadcast_getindex(bc::Broadcasted{DefaultArrayStyle{0}}, ::Val{i}) where {i} =
    bc.f(map(arg -> _tensor_broadcast_getindex(arg, Val(i)), bc.args)...)

@inline function _check_tensor_broadcast_tuple_length(x::Tuple, ::Val{L}) where {L}
    if length(x) != 1 && length(x) != L
        throw(DimensionMismatch("arrays could not be broadcast to a common size"))
    end
    nothing
end

@generated function _copy_tensor_broadcasted(::Type{TT}, f, args::Args) where {TT <: AbstractTensor, Args <: Tuple}
    L = ncomponents(TT)
    N = length(Args.parameters)
    xs = [gensym(:x) for _ in 1:N]
    setup = [:( $(xs[i]) = _tensor_broadcast_arg(TT, args[$i]) ) for i in 1:N]
    checks = Expr[]
    for i in 1:N
        T = Args.parameters[i]
        if T <: Tuple
            push!(checks, :(_check_tensor_broadcast_tuple_length($(xs[i]), Val($L))))
        end
    end
    values = map(1:L) do i
        callargs = [:( _tensor_broadcast_getindex($(xs[j]), Val($i)) ) for j in 1:N]
        :(f($(callargs...)))
    end
    tuple = Expr(:tuple, values...)
    quote
        @_inline_meta
        $(setup...)
        $(checks...)
        TT($tuple)
    end
end

@inline function Base.copy(bc::Broadcasted{<: TensorStyle})
    S = _promote_space_for_broadcast(bc.args)
    TT = tensortype(S)
    _copy_tensor_broadcasted(TT, bc.f, bc.args)
end
