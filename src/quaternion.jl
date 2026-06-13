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
    function Quaternion{T}(data::NTuple{4, Number}) where {T}
        T <: Complex && throw(ArgumentError("complex-valued quaternions are not supported"))
        new{T}(convert_ntuple(T, data))
    end
end

@inline Quaternion(data::NTuple{4, Any}) = Quaternion{promote_ntuple_eltype(data)}(data)
@inline (::Type{T})(a, b, c, d) where {T <: Quaternion} = T((a, b, c, d))

# from scalar and vector
@inline Quaternion{T}(r::Number, v::Vec{3}) where {T} = @inbounds Quaternion{T}(r, v[1], v[2], v[3])
@inline Quaternion(r::Number, v::Vec{3}) = Quaternion{promote_type(typeof(r), eltype(v))}(r, v)

# from vector
@inline Quaternion{T}(v::Vec{4}) where {T} = Quaternion{T}(Tuple(v))
@inline Quaternion{T}(v::Vec{3}) where {T} = Quaternion{T}(zero(eltype(v)), v)
@inline Quaternion(v::Vec) = Quaternion{eltype(v)}(v)

# from scalar
@inline Quaternion{T}(r::Number) where {T} = (z = zero(r); Quaternion{T}(r, z, z, z))
@inline Quaternion(r::Number) = Quaternion{typeof(r)}(r)

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
Base.convert(::Type{Quaternion{T}}, x::Number) where {T} = convert(Quaternion{T}, Quaternion(x))

# promotion
Base.promote_rule(::Type{Quaternion{T}}, ::Type{T}) where {T <: Number} = Quaternion{T}
Base.promote_rule(::Type{Quaternion{T}}, ::Type{U}) where {T <: Number, U <: Number} = Quaternion{promote_type(T, U)}
Base.promote_rule(::Type{Quaternion{T}}, ::Type{Quaternion{T}}) where {T <: Number} = Quaternion{T}
Base.promote_rule(::Type{Quaternion{T}}, ::Type{Quaternion{U}}) where {T <: Number, U <: Number} = Quaternion{promote_type(T, U)}

# used for `isapprox`
Base.zero(::Type{Quaternion{T}}) where {T} = (z = zero(T); Quaternion{T}(z, z, z, z))
Base.zero(q::Quaternion) = zero(typeof(q))
Base.one(::Type{Quaternion{T}}) where {T} = (z = zero(T); Quaternion{T}(one(T), z, z, z))
Base.one(q::Quaternion) = one(typeof(q))
Base.real(::Type{Quaternion{T}}) where {T} = T
Base.real(q::Quaternion) = q.scalar
Base.isfinite(q::Quaternion) = all(isfinite, Tuple(q))
Base.isinf(q::Quaternion) = any(isinf, Tuple(q))
Base.isnan(q::Quaternion) = any(isnan, Tuple(q))
Base.isreal(q::Quaternion) = iszero(q[2]) & iszero(q[3]) & iszero(q[4])
Base.iszero(q::Quaternion) = iszero(q[1]) & iszero(q[2]) & iszero(q[3]) & iszero(q[4])

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
quaternion(θ::Real, n::Vec) = quaternion(promote_type(typeof(θ), eltype(n)), θ, n)

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
Base.:*(a::Complex, q::Quaternion) = throw(MethodError(*, (a, q)))
Base.:*(q::Quaternion, a::Complex) = throw(MethodError(*, (q, a)))
Base.:/(q::Quaternion, a::Complex) = throw(MethodError(/, (q, a)))

# quaternion vs vector
@inline Base.:*(q::Quaternion, v::Vec) = q * Quaternion(v)
@inline Base.:*(v::Vec, q::Quaternion) = Quaternion(v) * q
@inline Base.:/(v::Vec, q::Quaternion) = v * inv(q)

"""
    angleaxis(::Quaternion)

Convert a quaternion to an angle-axis pair `(θ, n)`.
"""
function angleaxis(q::Quaternion)
    iszero(q) && throw(DomainError(q, "zero quaternion does not define a rotation"))
    q = normalize(q)
    q.scalar < zero(q.scalar) && (q = -q)
    a = norm(q.vector)
    θ = 2atan(a, q.scalar)
    if iszero(a)
        return θ, Vec(one(θ), zero(θ), zero(θ))
    end
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
@inline rotate(v::Vec{3}, q::Quaternion) = (q * v / q).vector

@inline Base.conj(q::Quaternion) = Quaternion(q.scalar, -q.vector)
@inline Base.abs2(q::Quaternion) = (v = Vec(q); contract1(v, v))
@inline Base.abs(q::Quaternion) = hypot(Tuple(q)...)
@inline norm(q::Quaternion) = abs(q)
function inv(q::Quaternion)
    scale = maximum(abs, Tuple(q))
    if iszero(scale) || isinf(scale) || isnan(scale)
        return conj(q) / abs2(q)
    end
    p = q / scale
    conj(p) / (scale * abs2(p))
end

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
    sqrt(::Quaternion)

Return the principal square root of a quaternion.

On the negative real axis, the branch uses the first imaginary basis direction.
"""
function Base.sqrt(q::Quaternion)
    q_norm = norm(q)
    v = q.vector
    v_norm = norm(v)
    if iszero(v_norm)
        if q.scalar < zero(q.scalar)
            s = sqrt(-q.scalar)
            return Quaternion(zero(s), Vec(s, zero(s), zero(s)))
        end
        return Quaternion(sqrt(q.scalar), zero(v))
    end
    if q.scalar ≥ zero(q.scalar)
        s = sqrt((q_norm + q.scalar) / 2)
        return Quaternion(s, v / (2s))
    end
    s = v_norm / sqrt(2(q_norm - q.scalar))
    Quaternion(s, v * sqrt((q_norm - q.scalar) / (2v_norm * v_norm)))
end

"""
    log(::Quaternion)

Return the principal logarithm of a quaternion:

```math
\\ln(q) = \\ln\\| q \\| + \\frac{\\bm{v}}{\\| \\bm{v} \\|} \\arccos\\frac{q_w}{\\| q \\|}
```

On the negative real axis, the branch uses the first imaginary basis direction.
"""
function Base.log(q::Quaternion)
    q_norm = norm(q)
    v = q.vector
    v_norm = norm(v)
    iszero(q_norm) && return Quaternion(log(q_norm), zero(v))
    if iszero(v_norm)
        ϕ = q.scalar < zero(q.scalar) ? oftype(q_norm, π) : zero(q_norm)
        return Quaternion(log(q_norm), Vec(ϕ, zero(ϕ), zero(ϕ)))
    end
    ϕ = atan(v_norm, q.scalar)
    Quaternion(log(q_norm), v / v_norm * ϕ)
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

function _isnegative_for_show(x)
    try
        isnegative = x < zero(x)
        isnegative isa Bool && return isnegative
        return false
    catch
        return false
    end
end

function Base.show(io::IO, q::Quaternion)
    pm(x) = _isnegative_for_show(x) ? " - $(-x)" : " + $x"
    print(io, q[1], pm(q[2]), "𝙞", pm(q[3]), "𝙟", pm(q[4]), "𝙠")
end
