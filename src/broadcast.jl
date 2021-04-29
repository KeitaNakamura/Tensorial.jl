import Base.Broadcast: BroadcastStyle, Broadcasted, broadcasted, broadcastable, materialize!

struct TensorStyle <: BroadcastStyle end
struct TensorAsScalarStyle <: BroadcastStyle end

BroadcastStyle(::Type{<: AbstractTensor}) = TensorStyle()
BroadcastStyle(x::TensorStyle, ::Broadcast.DefaultArrayStyle{0}) = x
BroadcastStyle(x::TensorStyle, ::Broadcast.Style{Tuple}) = x
BroadcastStyle(x::TensorStyle, ::BroadcastStyle) = TensorAsScalarStyle()
BroadcastStyle(x::TensorAsScalarStyle, ::BroadcastStyle) = TensorAsScalarStyle()

broadcastable(bc::Broadcasted{<: TensorStyle}) = copy(bc)

@generated function _promote_space_for_broadcast(x::Tuple)
    spaces = [Space(t) for t in x.parameters if t <: AbstractTensor]
    quote
        @_inline_meta
        promote_space($(spaces...))
    end
end

_broadcastable_for_tensorstyle(x::Any, ::Type{TT}) where {TT} = x
_broadcastable_for_tensorstyle(x::AbstractTensor, ::Type{TT}) where {TT} = Tuple(convert(TT, x))
broadcastable_for_tensorstyle(x::Tuple{Any}, ::Type{TT}) where {TT} = (_broadcastable_for_tensorstyle(x[1], TT),)
broadcastable_for_tensorstyle(x::Tuple{Any, Any}, ::Type{TT}) where {TT} = (_broadcastable_for_tensorstyle(x[1], TT), _broadcastable_for_tensorstyle(x[2], TT))
broadcastable_for_tensorstyle(x::Tuple, ::Type{TT}) where {TT} = (_broadcastable_for_tensorstyle(x[1], TT), broadcastable_for_tensorstyle(Base.tail(x), TT)...)
@inline function Base.copy(bc::Broadcasted{TensorStyle})
    S = _promote_space_for_broadcast(bc.args)
    TT = tensortype(S)
    TT(broadcast(bc.f, broadcastable_for_tensorstyle(bc.args, TT)...))
end

@inline _ref(x::AbstractTensor) = Ref(x)
@inline _ref(x::Any) = x
@inline function broadcasted(::TensorAsScalarStyle, f, args...)
    broadcasted(f, map(_ref, args)...)
end

# for broadcast!(op, ::Array, ::AbstractTensor)
function materialize!(dest, bc::Broadcasted{TensorStyle})
    materialize!(dest, (copy(bc),))
end
