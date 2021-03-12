# experimental
struct Quaternion{T} <: AbstractVector{T}
    data::NTuple{4, T}
    function Quaternion{T}(data::NTuple{4, Any}) where {T}
        new{T}(convert_ntuple(T, data))
    end
end

@inline Quaternion(data::NTuple{4, Any}) = Quaternion{promote_ntuple_eltype(data)}(data)
@inline (::Type{T})(data::Vararg{Any}) where {T <: Quaternion} = T(data)

# Quaternion <-> Vec
@inline Quaternion(v::Vec{4}) = Quaternion(Tuple(v))
@inline Vec(q::Quaternion) = Vec(Tuple(q))

Base.Tuple(q::Quaternion) = getfield(q, :data)

@inline function Base.getproperty(q::Quaternion, name::Symbol)
    name == :vector && return @inbounds Vec(q[1], q[2], q[3])
    name == :scalar && return @inbounds q[4]
    getfield(q, name)
end

Base.propertynames(q::Quaternion) = (:scalar, :vector, :data)

"""
    quaternion(v::Vec, θ; [degree = false])

Construct `Quaternion` from direction `v` and angle `θ`.
The constructed quaternion is normalized such as `norm(q) ≈ 1`.

```jldoctest
julia> quaternion(Vec(1.0,2.0,3.0), π/2)
```
"""
function quaternion(v::Vec{3}, θ::Real; degree::Bool = false)
    if degree
        θ = deg2rad(θ)
    end
    ϕ = θ / 2
    sinϕ = sin(ϕ)
    n = normalize(v) * sinϕ
    T = promote_type(typeof(ϕ), eltype(n))
    @inbounds Quaternion{T}(n[1], n[2], n[3], cos(ϕ))
end

function Base.summary(io::IO, q::Quaternion)
    print(io, "Quaternion{", eltype(q), "}<q₁i+q₂j+q₃k+q₄>")
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
    A = @Mat [ q4 -q3  q2  q1
               q3  q4 -q1  q2
              -q2  q1  q4  q3
              -q1 -q2 -q3  q4 ]
    Quaternion(A ⋅ Vec(p))
end

@inline function Base.:*(q::Quaternion, v::Vec{3, T}) where {T}
    @inbounds q * Quaternion(v[1], v[2], v[3], zero(T))
end
@inline function Base.:*(v::Vec{3, T}, q::Quaternion) where {T}
    @inbounds Quaternion(v[1], v[2], v[3], zero(T)) * q
end

@inline norm(q::Quaternion) = norm(Vec(q))

@inline function inv(q::Quaternion)
    qvec = Vec(q)
    @inbounds Quaternion(-q[1], -q[2], -q[3], q[4]) / dot(qvec, qvec)
end

function rotmat(q::Quaternion)
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
    @Mat [q1²-q2²-q3²+q4²  2(q1q2-q3q4)     2(q1q3+q2q4)
          2(q1q2+q3q4)     -q1²+q2²-q3²+q4² 2(q2q3-q1q4)
          2(q1q3-q2q4)     2(q2q3+q1q4)     -q1²-q2²+q3²+q4²]
end
