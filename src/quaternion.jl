"""
`Quaternion` represents ``q_w + q_x \\bm{i} + q_y \\bm{j} + q_z \\bm{k}``.
The salar part and vector part can be accessed by `q.scalar` and `q.vector`, respectively.

# Examples
```jldoctest
julia> Quaternion(1,2,3,4)
1 + 2𝙞 + 3𝙟 + 4𝙠

julia> Quaternion(1)
1 + 0𝙞 + 0𝙟 + 0𝙠

julia> Quaternion(Vec(1,2,3))
0 + 1𝙞 + 2𝙟 + 3𝙠
```

See also [`quaternion`](@ref).

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
@inline Quaternion{T}(r::Real, v::Vec{3}) where {T} = @inbounds Quaternion{T}(r, v[1], v[2], v[3])
@inline Quaternion{T}(r::Real, v::Vec{2}) where {T} = @inbounds Quaternion{T}(r, v[1], v[2], zero(eltype(v)))
@inline Quaternion{T}(r::Real, v::Vec{1}) where {T} = @inbounds Quaternion{T}(r, v[1], zero(eltype(v)), zero(eltype(v)))
@inline Quaternion{T}(v::Vec{4}) where {T} = Quaternion{T}(Tuple(v))
for dim in 1:3
    @eval @inline Quaternion{T}(v::Vec{$dim}) where {T} = Quaternion{T}(zero(eltype(v)), v)
end
@inline Quaternion(v::Vec) = Quaternion{eltype(v)}(v)
@inline Quaternion(r::Real, v::Vec) = Quaternion{promote_type(typeof(r), eltype(v))}(r, v)
@inline Vec(q::Quaternion) = Vec(Tuple(q))

@inline Quaternion(x::Real) = (z = zero(x); Quaternion(x, z, z, z))

Base.Tuple(q::Quaternion) = getfield(q, :data)

@inline function Base.getproperty(q::Quaternion, name::Symbol)
    name == :scalar && return @inbounds q[1]
    name == :vector && return @inbounds Vec(q[2], q[3], q[4])
    name == :w && return @inbounds q[1]
    name == :x && return @inbounds q[2]
    name == :y && return @inbounds q[3]
    name == :z && return @inbounds q[4]
    getfield(q, name)
end

Base.propertynames(q::Quaternion) = (:scalar, :vector, :w, :x, :y, :z, :data)

# conversion
Base.convert(::Type{Quaternion{T}}, x::Quaternion{T}) where {T} = x
Base.convert(::Type{Quaternion{T}}, x::Quaternion{U}) where {T, U} = Quaternion(map(T, Tuple(x)))
Base.convert(::Type{Quaternion{T}}, x::Real) where {T} = convert(Quaternion{T}, Quaternion(x))

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

Construct `Quaternion` from angle `θ` and axis `x`.
The constructed quaternion is normalized such as `norm(q) ≈ 1` by default.

# Examples
```jldoctest
julia> q = quaternion(π/4, Vec(0,0,1))
0.9238795325112867 + 0.0𝙞 + 0.0𝙟 + 0.3826834323650898𝙠

julia> v = rand(Vec{3})
3-element Vec{3, Float64}:
 0.5908446386657102
 0.7667970365022592
 0.5662374165061859

julia> (q * v / q).vector ≈ rotmatz(π/4) ⋅ v
true
```
"""
function quaternion(::Type{T}, θ::Real, x::Vec{3}; normalize::Bool = true, degree::Bool = false) where {T}
    if degree
        θ = deg2rad(θ)
    end
    ϕ = θ / 2
    if normalize
        n = LinearAlgebra.normalize(x) * sin(ϕ)
    else
        n = x * sin(ϕ)
    end
    @inbounds Quaternion{T}(cos(ϕ), n)
end
quaternion(T::Type, θ::Real, x::Vec{2}; normalize::Bool = true, degree::Bool = false) =
    @inbounds quaternion(T, θ, Vec(x[1], x[2], 0); normalize, degree)
quaternion(θ::Real, x::Vec; normalize::Bool = true, degree::Bool = false) =
    quaternion(promote_type(typeof(θ), eltype(x)), θ, x; normalize, degree)

Base.length(::Quaternion) = 4
Base.size(::Quaternion) = (4,)

@inline function Base.getindex(q::Quaternion, i::Int)
    @boundscheck 1 ≤ i ≤ 4 || throw(BoundsError(q, i))
    @inbounds Tuple(q)[i]
end

# quaternion vs quaternion
@inline Base.:-(q::Quaternion) = Quaternion(-Vec(q))
@inline Base.:+(q::Quaternion, p::Quaternion) = Quaternion(Vec(q) + Vec(p))
@inline Base.:-(q::Quaternion, p::Quaternion) = Quaternion(Vec(q) - Vec(p))
@inline Base.:/(q::Quaternion, p::Quaternion) = q * inv(p)
@inline function Base.:*(q::Quaternion, p::Quaternion)
    q₁, q₂, q₃, q₄ = Tuple(q)
    A = @Mat [ q₁ -q₂ -q₃ -q₄
               q₂  q₁ -q₄  q₃
               q₃  q₄  q₁ -q₂
               q₄ -q₃  q₂  q₁ ]
    Quaternion(A ⋅ Vec(p))
end

# quaternion vs number
@inline Base.:*(a::Number, q::Quaternion) = Quaternion(a * Vec(q))
@inline Base.:*(q::Quaternion, a::Number) = Quaternion(Vec(q) * a)
@inline Base.:/(q::Quaternion, a::Number) = Quaternion(Vec(q) / a)

# quaternion vs vector
@inline Base.:*(q::Quaternion, v::Vec) = q * Quaternion(v)
@inline Base.:*(v::Vec, q::Quaternion) = Quaternion(v) * q
@inline Base.:/(v::Vec, q::Quaternion) = v * inv(q)

"""
    rotate(x::Vec, q::Quaternion)

Rotate `x` by quaternion `q`.

# Examples
```jldoctest
julia> v = Vec(1.0, 0.0, 0.0)
3-element Vec{3, Float64}:
 1.0
 0.0
 0.0

julia> rotate(v, quaternion(π/4, Vec(0,0,1)))
3-element Vec{3, Float64}:
 0.7071067811865475
 0.7071067811865476
 0.0
```
"""
@inline rotate(v::Vec{3}, q::Quaternion) = (q * v / q).vector
@inline rotate(v::Vec{2}, q::Quaternion) = (v = (q * v / q).vector; @inbounds Vec(v[1], v[2]))

@inline Base.conj(q::Quaternion) = @inbounds Quaternion(q[1], -q[2], -q[3], -q[4])
@inline Base.abs2(q::Quaternion) = (v = Vec(q); dot(v, v))
@inline Base.abs(q::Quaternion) = sqrt(abs2(q))
@inline norm(q::Quaternion) = abs(q)
@inline inv(q::Quaternion) = conj(q) / abs2(q)

function Base.exp(q::Quaternion)
    v = q.vector
    norm_v = norm(v)
    if norm_v > 0
        n = v / norm_v
    else
        n = zero(v)
    end
    exp(q.scalar) * quaternion(2norm_v, n; normalize = false)
end

function Base.log(q::Quaternion)
    norm_q = norm(q)
    v = q.vector
    norm_v = norm(v)
    Quaternion(log(norm_q), Tuple(v/norm_v * acos(q.scalar/norm_q))...)
end

@inline normalize(q::Quaternion) = q / norm(q)

function rotmat_normalized(q::Quaternion)
    q₁, q₂, q₃, q₄ = Tuple(q)
    q₁² = q₁ * q₁
    q₂² = q₂ * q₂
    q₃² = q₃ * q₃
    q₄² = q₄ * q₄
    q₁q₂ = q₁ * q₂
    q₂q₃ = q₂ * q₃
    q₃q₄ = q₃ * q₄
    q₁q₃ = q₁ * q₃
    q₁q₄ = q₁ * q₄
    q₂q₄ = q₂ * q₄
    @Mat [q₁²+q₂²-q₃²-q₄² 2(q₂q₃-q₁q₄)    2(q₂q₄+q₁q₃)
          2(q₂q₃+q₁q₄)    q₁²-q₂²+q₃²-q₄² 2(q₃q₄-q₁q₂)
          2(q₂q₄-q₁q₃)    2(q₃q₄+q₁q₂)    q₁²-q₂²-q₃²+q₄²]
end
@inline rotmat(q::Quaternion) = rotmat_normalized(normalize(q))

function Base.show(io::IO, q::Quaternion)
    pm(x) = x < 0 ? " - $(-x)" : " + $x"
    print(io, q[1], pm(q[2]), "𝙞", pm(q[3]), "𝙟", pm(q[4]), "𝙠")
end
