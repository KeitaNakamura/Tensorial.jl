# experimental
struct Quaternion{T} <: AbstractVector{T}
    data::NTuple{4, T}
    function Quaternion{T}(data::NTuple{4, Any}) where {T}
        new{T}(convert_ntuple(T, data))
    end
end

@inline Quaternion(data::NTuple{4, Any}) = Quaternion{promote_ntuple_eltype(data)}(data)
@inline (::Type{T})(data::Vararg{Any}) where {T <: Quaternion} = T(data)
@inline Quaternion(v::Vec{4}) = Quaternion(Tuple(v))
@inline Vec(q::Quaternion) = Vec(Tuple(q))

Base.Tuple(q::Quaternion) = getfield(q, :data)

@inline function Base.getproperty(q::Quaternion, name::Symbol)
    name == :scalar && return @inbounds q[1]
    name == :vector && return @inbounds Vec(q[2], q[3], q[4])
    getfield(q, name)
end

Base.propertynames(q::Quaternion) = (:scalar, :vector, :data)

"""
    quaternion(x, θ; [degree = false])

Construct `Quaternion` from direction `x` and angle `θ`.
The constructed quaternion is normalized such as `norm(q) ≈ 1`.
"""
function quaternion(x::Vec{3}, θ::Real; degree::Bool = false)
    if degree
        θ = deg2rad(θ)
    end
    ϕ = θ / 2
    sinϕ = sin(ϕ)
    n = normalize(x) * sinϕ
    @inbounds Quaternion(cos(ϕ), n[1], n[2], n[3])
end

Base.size(::Quaternion) = (4,)

@inline function Base.getindex(q::Quaternion, i::Int)
    @boundscheck checkbounds(q, i)
    @inbounds Tuple(q)[i]
end

@inline Base.:+(q::Quaternion, p::Quaternion) = Quaternion(Vec(q) + Vec(p))
@inline Base.:-(q::Quaternion, p::Quaternion) = Quaternion(Vec(q) - Vec(p))

@inline Base.:*(a::Number, q::Quaternion) = Quaternion(a * Vec(q))
@inline Base.:*(q::Quaternion, a::Number) = Quaternion(Vec(q) * a)
@inline Base.:/(q::Quaternion, a::Number) = Quaternion(Vec(q) / a)

@inline function Base.:*(q::Quaternion, p::Quaternion)
    q1, q2, q3, q4 = Tuple(q)
    A = @Mat [ q1 -q2 -q3 -q4
               q2  q1 -q4  q3
               q3  q4  q1 -q2
               q4 -q3  q2  q1 ]
    Quaternion(A ⋅ Vec(p))
end

@inline function Base.:*(q::Quaternion, v::Vec{3, T}) where {T}
    @inbounds q * Quaternion(zero(T), v[1], v[2], v[3])
end
@inline function Base.:*(v::Vec{3, T}, q::Quaternion) where {T}
    @inbounds Quaternion(zero(T), v[1], v[2], v[3]) * q
end

@inline norm(q::Quaternion) = norm(Vec(q))

@inline function inv(q::Quaternion)
    qvec = Vec(q)
    @inbounds Quaternion(q[1], -q[2], -q[3], -q[4]) / dot(qvec, qvec)
end

function rotmat(q::Quaternion)
    s = 1 / norm(q)
    q1, q2, q3, q4 = Tuple(q)
    q1² = q1 * q1
    q2² = q2 * q2
    q3² = q3 * q3
    q4² = q4 * q4
    q1q2 = q1 * q2
    q2q3 = q2 * q3
    q3q4 = q3 * q4
    q1q3 = q1 * q3
    q1q4 = q1 * q4
    q2q4 = q2 * q4
    @Mat [1-2s*(q3²+q4²) 2(q2q3-q1q4)   2(q2q4+q1q3)
          2(q2q3+q1q4)   1-2s*(q2²+q4²) 2(q3q4-q1q2)
          2(q2q4-q1q3)   2(q3q4+q1q2)   1-2s*(q2²+q3²)]
end
