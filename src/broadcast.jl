import Base.Broadcast: BroadcastStyle, DefaultArrayStyle, AbstractArrayStyle, Broadcasted, broadcastable

struct TensorStyle{N} <: AbstractArrayStyle{N} end

BroadcastStyle(::Type{<: AbstractTensor{<: Any, <: Any, N}}) where {N} = TensorStyle{N}()
BroadcastStyle(::TensorStyle{N}, ::DefaultArrayStyle{0}) where {N} = TensorStyle{N}()
BroadcastStyle(::TensorStyle, b::DefaultArrayStyle) = b
BroadcastStyle(a::TensorStyle, ::Broadcast.Style{Tuple}) = a

broadcastable(bc::Broadcasted{<: TensorStyle}) = copy(bc)

@generated function _promote_space_for_broadcast(x::Tuple)
    spaces = [Space(t) for t in x.parameters if t <: AbstractTensor]
    promote_space(spaces...)
end

_broadcastable(::Type{TT}, x::Any) where {TT} = x
_broadcastable(::Type{TT}, x::AbstractTensor) where {TT} = SVector(Tuple(convert(TT, x)))
@inline function Base.copy(bc::Broadcasted{<: TensorStyle})
    S = _promote_space_for_broadcast(bc.args)
    TT = tensortype(S)
    res = bc.f.(_broadcastable.(TT, bc.args)...)
    res isa SVector || error("unreachable")
    TT(Tuple(res))
end
