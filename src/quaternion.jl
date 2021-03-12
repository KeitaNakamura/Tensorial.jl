"""
`Quaternion` represents ``q_1 + q_2 \\bm{i} + q_3 \\bm{j} + q_4 \\bm{k}``.
The salar part and vector part can be accessed by `q.scalar` and `q.vector`, respectively.

!!! note

    `Quaternion` is experimental and could change or disappear in future versions of Tensorial.
"""
struct Quaternion{T} <: Number
    data::NTuple{4, T}
    function Quaternion{T}(data::NTuple{4, Real}) where {T}
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
    name == :scalar && return @inbounds q[1]
    name == :vector && return @inbounds Vec(q[2], q[3], q[4])
    getfield(q, name)
end

Base.propertynames(q::Quaternion) = (:scalar, :vector, :data)

# conversion
Base.convert(::Type{Quaternion{T}}, x::Quaternion{T}) where {T} = x
function Base.convert(::Type{Quaternion{T}}, x::Quaternion{U}) where {T, U}
    @inbounds Quaternion(convert(T, x[1]), convert(T, x[2]), convert(T, x[3]), convert(T, x[4]))
end
function Base.convert(::Type{Quaternion{T}}, x::Real) where {T}
    Quaternion(convert(T, x), convert(T, 0), convert(T, 0), convert(T, 0))
end

# promotion
Base.promote_rule(::Type{Quaternion{T}}, ::Type{T}) where {T <: Real} = Quaternion{T}
Base.promote_rule(::Type{Quaternion{T}}, ::Type{U}) where {T <: Real, U <: Real} = Quaternion{promote_type(T, U)}
Base.promote_rule(::Type{Quaternion{T}}, ::Type{Quaternion{T}}) where {T <: Real} = Quaternion{T}
Base.promote_rule(::Type{Quaternion{T}}, ::Type{Quaternion{U}}) where {T <: Real, U <: Real} = Quaternion{promote_type(T, U)}

# used for `isapprox`
Base.real(q::Quaternion) = q.scalar
Base.isfinite(q::Quaternion) = prod(map(isfinite, Tuple(q)))

"""
    quaternion(θ, x::Vec; [normalize = true, degree = false])

Construct `Quaternion` from direction `x` and angle `θ`.
The constructed quaternion is normalized such as `norm(q) ≈ 1` by default.

```jldoctest
julia> q = quaternion(π/4, Vec(0,0,1))
0.9238795325112867 + 0.0𝙞 + 0.0𝙟 + 0.3826834323650898𝙠

julia> v = rand(Vec{3})
3-element Tensor{Tuple{3},Float64,1,3}:
 0.5908446386657102
 0.7667970365022592
 0.5662374165061859

julia> (q * v / q).vector ≈ rotmatz(π/4) ⋅ v
true
```
"""
function quaternion(θ::Real, x::Vec{3}; normalize::Bool = true, degree::Bool = false)
    if degree
        θ = deg2rad(θ)
    end
    ϕ = θ / 2
    sinϕ = sin(ϕ)
    if normalize
        n = LinearAlgebra.normalize(x) * sinϕ
    else
        n = x * sinϕ
    end
    @inbounds Quaternion(cos(ϕ), n[1], n[2], n[3])
end

Base.length(::Quaternion) = 4
Base.size(::Quaternion) = (4,)

@inline function Base.getindex(q::Quaternion, i::Int)
    @_propagate_inbounds_meta
    @inbounds Tuple(q)[i]
end

# quaternion vs quaternion
@inline Base.:-(q::Quaternion) = Quaternion(-Vec(q))
@inline Base.:+(q::Quaternion, p::Quaternion) = Quaternion(Vec(q) + Vec(p))
@inline Base.:-(q::Quaternion, p::Quaternion) = Quaternion(Vec(q) - Vec(p))
@inline Base.:/(q::Quaternion, p::Quaternion) = q * inv(p)
@inline function Base.:*(q::Quaternion, p::Quaternion)
    q1, q2, q3, q4 = Tuple(q)
    A = @Mat [ q1 -q2 -q3 -q4
               q2  q1 -q4  q3
               q3  q4  q1 -q2
               q4 -q3  q2  q1 ]
    Quaternion(A ⋅ Vec(p))
end

# quaternion vs number
@inline Base.:*(a::Number, q::Quaternion) = Quaternion(a * Vec(q))
@inline Base.:*(q::Quaternion, a::Number) = Quaternion(Vec(q) * a)
@inline Base.:/(q::Quaternion, a::Number) = Quaternion(Vec(q) / a)

# quaternion vs vector
@inline function Base.:*(q::Quaternion, v::Vec{3, T}) where {T}
    @inbounds q * Quaternion(zero(T), v[1], v[2], v[3])
end
@inline function Base.:*(v::Vec{3, T}, q::Quaternion) where {T}
    @inbounds Quaternion(zero(T), v[1], v[2], v[3]) * q
end

@inline Base.conj(q::Quaternion) = @inbounds Quaternion(q[1], -q[2], -q[3], -q[4])
@inline Base.abs2(q::Quaternion) = @inbounds (qvec = Vec(q); dot(qvec, qvec))
@inline Base.abs(q::Quaternion) = sqrt(abs2(q))
@inline norm(q::Quaternion) = abs(q)
@inline inv(q::Quaternion) = conj(q) / abs2(q)

function Base.exp(q::Quaternion)
    v = q.vector
    norm_v = norm(v)
    exp(q.scalar) * quaternion(2norm_v, v/norm_v; normalize = false)
end

function Base.log(q::Quaternion)
    norm_q = norm(q)
    v = q.vector
    norm_v = norm(v)
    Quaternion(log(norm_q), Tuple(v/norm_v * acos(q.scalar/norm_q))...)
end

normalize(q::Quaternion) = q / norm(q)

rotmat(q::Quaternion) = rotmat_normalized(normalize(q))

function rotmat_normalized(q::Quaternion)
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

function Base.show(io::IO, q::Quaternion)
    pm(x) = x < 0 ? " - $(-x)" : " + $x"
    print(io, q[1], pm(q[2]), "𝙞", pm(q[3]), "𝙟", pm(q[4]), "𝙠")
end
