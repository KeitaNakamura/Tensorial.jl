import Base.Broadcast: BroadcastStyle, Style, Broadcasted, broadcastable, broadcasted

struct TensorStyle{N} <: BroadcastStyle end

BroadcastStyle(::Type{<: AbstractTensor{<: Any, <: Any, N}}) where {N} = TensorStyle{N}()
BroadcastStyle(x::TensorStyle, ::BroadcastStyle) = x

# basically AbstractTensor behaves like scalar in broadcast
@inline _ref(x::AbstractTensor) = Ref(x)
@inline _ref(x::Any) = x
@inline broadcasted(x::TensorStyle, f, args...) = broadcasted(f, map(_ref, args)...)

# special version between AbstractVec and Tuple
# only this case returns Broadcasted{TensorStyle{1}}
@inline broadcasted(x::TensorStyle{1}, f, args::Vararg{Union{AbstractVec, Tuple}}) = Broadcasted{TensorStyle{1}}(f, args)
_vec(x::Tuple) = Vec(x)
_vec(x::AbstractVec) = x
@inline Base.copy(bc::Broadcasted{TensorStyle{1}}) = bc.f(map(_vec, bc.args)...)
