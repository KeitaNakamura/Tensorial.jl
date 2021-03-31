import Base.Broadcast: BroadcastStyle, Broadcasted, instantiate, broadcasted, broadcastable

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

_to_tuple(x::Real, ::Type{TT}) where {TT} = (x,)
_to_tuple(x::AbstractTensor, ::Type{TT}) where {TT} = Tuple(convert(TT, x))
_to_tuple(x::Tuple, ::Type{TT}) where {TT} = x
to_tuple(x::Tuple{Any}, ::Type{TT}) where {TT} = (_to_tuple(x[1], TT),)
to_tuple(x::Tuple{Any, Any}, ::Type{TT}) where {TT} = (_to_tuple(x[1], TT), _to_tuple(x[2], TT))
to_tuple(x::Tuple, ::Type{TT}) where {TT} = (_to_tuple(x[1], TT), to_tuple(Base.tail(x), TT)...)
@inline function Base.copy(bc::Broadcasted{TensorStyle})
    S = _promote_space_for_broadcast(bc.args)
    TT = tensortype(S)
    TT(broadcast(bc.f, to_tuple(bc.args, TT)...))
end

@inline _ref(x::AbstractTensor) = Ref(x)
@inline _ref(x::Any) = x
@inline function instantiate(bc::Broadcasted{TensorAsScalarStyle})
    instantiate(broadcasted(bc.f, map(_ref, bc.args)...))
end
