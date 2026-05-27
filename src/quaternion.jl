"""
`Quaternion` represents ``q_w + q_x \\bm{i} + q_y \\bm{j} + q_z \\bm{k}``.
The scalar part and vector part can be accessed by `q.scalar` and `q.vector`, respectively.

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

@inline _quaternion_vec3(v::Vec{2}) = @inbounds Vec(v[1], v[2], zero(eltype(v)))
@inline _quaternion_vec3(v::Vec{1}) = @inbounds Vec(v[1], zero(eltype(v)), zero(eltype(v)))

# from scalar and vector
@inline Quaternion{T}(r::Real, v::Vec{3}) where {T} = @inbounds Quaternion{T}(r, v[1], v[2], v[3])
@inline Quaternion{T}(r::Real, v::Vec{2}) where {T} = Quaternion{T}(r, _quaternion_vec3(v))
@inline Quaternion{T}(r::Real, v::Vec{1}) where {T} = Quaternion{T}(r, _quaternion_vec3(v))
@inline Quaternion(r::Real, v::Vec) = Quaternion{promote_type(typeof(r), eltype(v))}(r, v)

# from vector
@inline Quaternion{T}(v::Vec{4}) where {T} = Quaternion{T}(Tuple(v))
for dim in 1:3
    @eval @inline Quaternion{T}(v::Vec{$dim}) where {T} = Quaternion{T}(zero(eltype(v)), v)
end
@inline Quaternion(v::Vec) = Quaternion{eltype(v)}(v)

# from scalar
@inline Quaternion{T}(r::Real) where {T} = (z = zero(r); Quaternion{T}(r, z, z, z))
@inline Quaternion(r::Real) = Quaternion{typeof(r)}(r)

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
Base.convert(::Type{Quaternion{T}}, x::Quaternion{U}) where {T, U} = Quaternion(map(T, Tuple(x)))
Base.convert(::Type{Quaternion{T}}, x::Real) where {T} = convert(Quaternion{T}, Quaternion(x))

# promotion
Base.promote_rule(::Type{Quaternion{T}}, ::Type{T}) where {T <: Real} = Quaternion{T}
Base.promote_rule(::Type{Quaternion{T}}, ::Type{U}) where {T <: Real, U <: Real} = Quaternion{promote_type(T, U)}
Base.promote_rule(::Type{Quaternion{T}}, ::Type{Quaternion{T}}) where {T <: Real} = Quaternion{T}
Base.promote_rule(::Type{Quaternion{T}}, ::Type{Quaternion{U}}) where {T <: Real, U <: Real} = Quaternion{promote_type(T, U)}

# used for `isapprox`
Base.real(q::Quaternion) = q.scalar
Base.isfinite(q::Quaternion) = all(isfinite, Tuple(q))

"""
    quaternion(θ, n::Vec)

Construct `Quaternion` from angle `θ` and axis `n` as

```math
q = \\cos\\frac{\\theta}{2} + \\bm{n} \\sin\\frac{\\theta}{2}
```

`n` must be a unit vector.

# Examples
```jldoctest
julia> q = quaternion(π/4, Vec(0,0,1))
0.9238795325112867 + 0.0𝙞 + 0.0𝙟 + 0.3826834323650898𝙠

julia> x = rand(Vec{3})
3-element Vec{3, Float64}:
 0.32597672886359486
 0.5490511363155669
 0.21858665481883066

julia> (q * x / q).vector ≈ rotmatz(π/4) * x
true
```
"""
function quaternion(::Type{T}, θ::Real, n::Vec{3}) where {T}
    ϕ = θ / 2
    v = n * sin(ϕ)
    @inbounds Quaternion{T}(cos(ϕ), v)
end
quaternion(T::Type, θ::Real, n::Vec{2}) =
    @inbounds quaternion(T, θ, Vec(n[1], n[2], 0))
quaternion(θ::Real, n::Vec) =
    quaternion(promote_type(typeof(θ), eltype(n)), θ, n)

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
    Quaternion(A ⊡ Vec(p))
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
    angleaxis(::Quaternion)

Convert a quaternion to an angle-axis pair `(θ, n)`.
"""
function angleaxis(q::Quaternion)
    a = norm(q.vector)
    θ = 2atan(a, q.scalar)
    n = q.vector / a
    θ, n
end

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
@inline rotate(v::Vec, q::Quaternion) = (q * v / q).vector

@inline Base.conj(q::Quaternion) = Quaternion(q.scalar, -q.vector)
@inline Base.abs2(q::Quaternion) = (v = Vec(q); contract1(v, v))
@inline Base.abs(q::Quaternion) = sqrt(abs2(q))
@inline norm(q::Quaternion) = abs(q)
@inline inv(q::Quaternion) = conj(q) / abs2(q)

"""
    exp(::Quaternion)

Compute the exponential of quaternion as

```math
\\exp(q) = e^{q_w} \\left( \\cos\\| \\bm{v} \\| + \\frac{\\bm{v}}{\\| \\bm{v} \\|} \\sin\\| \\bm{v} \\| \\right)
```
"""
function Base.exp(q::Quaternion)
    v = q.vector
    v_norm = norm(v)
    if v_norm > 0
        n = v / v_norm
    else
        n = zero(v)
    end
    exp(q.scalar) * quaternion(2*v_norm, n)
end

"""
    log(::Quaternion)

Compute the logarithm of quaternion as

```math
\\ln(q) = \\ln\\| q \\| + \\frac{\\bm{v}}{\\| \\bm{v} \\|} \\arccos\\frac{q_w}{\\| q \\|}
```
"""
function Base.log(q::Quaternion)
    q_norm = norm(q)
    v = q.vector
    ϕ = acos(q.scalar/q_norm)
    Quaternion(log(q_norm), normalize(v) * ϕ)
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

"""
    rotmat(::Quaternion)

Construct rotation matrix from quaternion.

# Examples
```jldoctest
julia> q = quaternion(π/4, Vec(0,0,1))
0.9238795325112867 + 0.0𝙞 + 0.0𝙟 + 0.3826834323650898𝙠

julia> rotmat(q)
3×3 Tensor{Tuple{3, 3}, Float64, 2, 9}:
 0.707107  -0.707107  0.0
 0.707107   0.707107  0.0
 0.0        0.0       1.0
```
"""
@inline rotmat(q::Quaternion) = rotmat_normalized(normalize(q))

function Base.show(io::IO, q::Quaternion)
    pm(x) = x < 0 ? " - $(-x)" : " + $x"
    print(io, q[1], pm(q[2]), "𝙞", pm(q[3]), "𝙟", pm(q[4]), "𝙠")
end
