"""
    mean(::AbstractSecondOrderTensor{3})
    mean(::AbstractSymmetricSecondOrderTensor{3})

Compute the mean value of diagonal entries of a square tensor.

# Examples
```jldoctest
julia> x = rand(Mat{3,3})
3×3 Tensor{Tuple{3, 3}, Float64, 2, 9}:
 0.325977  0.894245  0.953125
 0.549051  0.353112  0.795547
 0.218587  0.394255  0.49425

julia> mean(x)
0.3911127467177425
```
"""
@inline mean(x::AbstractSquareTensor{3}) = tr(x) / 3

"""
    vol(::AbstractSecondOrderTensor{3})
    vol(::AbstractSymmetricSecondOrderTensor{3})

Compute the volumetric part of a square tensor.
This is only available in 3D.

# Examples
```jldoctest
julia> x = rand(Mat{3,3})
3×3 Tensor{Tuple{3, 3}, Float64, 2, 9}:
 0.325977  0.894245  0.953125
 0.549051  0.353112  0.795547
 0.218587  0.394255  0.49425

julia> vol(x)
3×3 Tensor{Tuple{3, 3}, Float64, 2, 9}:
 0.391113  0.0       0.0
 0.0       0.391113  0.0
 0.0       0.0       0.391113

julia> vol(x) + dev(x) ≈ x
true
```
"""
@inline function vol(x::AbstractSquareTensor{3})
    v = mean(x)
    z = zero(v)
    typeof(x)((i,j) -> ifelse(i == j, v, z))
end

"""
    vol(::AbstractVec{3})

Compute the volumetric part of a vector (assuming principal values of stresses and strains).
This is only available in 3D.

# Examples
```jldoctest
julia> x = rand(Vec{3})
3-element Vec{3, Float64}:
 0.32597672886359486
 0.5490511363155669
 0.21858665481883066

julia> vol(x)
3-element Vec{3, Float64}:
 0.3645381733326641
 0.3645381733326641
 0.3645381733326641

julia> vol(x) + dev(x) ≈ x
true
```
"""
@inline vol(x::AbstractVec{3}) = mean(x) * ones(x)

"""
    vol(::Type{FourthOrderTensor{3}})
    vol(::Type{SymmetricFourthOrderTensor{3}})

Construct volumetric fourth order identity tensor.
This is only available in 3D.

# Examples
```jldoctest
julia> x = rand(Mat{3,3});

julia> I_vol = vol(FourthOrderTensor{3});

julia> I_vol ⊡ x ≈ vol(x)
true

julia> vol(FourthOrderTensor{3}) + dev(FourthOrderTensor{3}) ≈ one(FourthOrderTensor{3})
true
```
"""
@inline vol(x::Type{TT}) where {TT <: Tensor} = _vol(basetype(TT))
@inline function _vol(x::Type{Tensor{NTuple{4, 3}, T}}) where {T}
    δ = one(SecondOrderTensor{3, T})
    δ ⊗ δ / T(3)
end
@inline function _vol(x::Type{Tensor{NTuple{2, @Symmetry({3,3})}, T}}) where {T}
    δ = one(SymmetricSecondOrderTensor{3, T})
    δ ⊗ δ / T(3)
end
@inline _vol(x::Type{Tensor{NTuple{4, 3}}}) = _vol(Tensor{NTuple{4, 3}, Float64})
@inline _vol(x::Type{Tensor{NTuple{2, @Symmetry({3,3})}}}) = _vol(Tensor{NTuple{2, @Symmetry({3,3})}, Float64})

"""
    dev(::AbstractSecondOrderTensor{3})
    dev(::AbstractSymmetricSecondOrderTensor{3})

Compute the deviatoric part of a square tensor.
This is only available in 3D.

# Examples
```jldoctest
julia> x = rand(Mat{3,3})
3×3 Tensor{Tuple{3, 3}, Float64, 2, 9}:
 0.325977  0.894245  0.953125
 0.549051  0.353112  0.795547
 0.218587  0.394255  0.49425

julia> dev(x)
3×3 Tensor{Tuple{3, 3}, Float64, 2, 9}:
 -0.065136   0.894245   0.953125
  0.549051  -0.0380011  0.795547
  0.218587   0.394255   0.103137

julia> tr(dev(x))
5.551115123125783e-17
```
"""
@inline dev(x::AbstractSquareTensor{3}) = x - vol(x)

"""
    dev(::AbstractVec{3})

Compute the deviatoric part of a vector (assuming principal values of stresses and strains).
This is only available in 3D.

# Examples
```jldoctest
julia> x = rand(Vec{3})
3-element Vec{3, Float64}:
 0.32597672886359486
 0.5490511363155669
 0.21858665481883066

julia> dev(x)
3-element Vec{3, Float64}:
 -0.03856144446906923
  0.18451296298290282
 -0.14595151851383342

julia> vol(x) + dev(x) ≈ x
true
```
"""
@inline dev(x::AbstractVec{3}) = x - vol(x)

"""
    dev(::Type{FourthOrderTensor{3}})
    dev(::Type{SymmetricFourthOrderTensor{3}})

Construct deviatoric fourth order identity tensor.
This is only available in 3D.

# Examples
```jldoctest
julia> x = rand(Mat{3,3});

julia> I_dev = dev(FourthOrderTensor{3});

julia> I_dev ⊡ x ≈ dev(x)
true

julia> vol(FourthOrderTensor{3}) + dev(FourthOrderTensor{3}) ≈ one(FourthOrderTensor{3})
true
```
"""
@inline dev(x::Type{TT}) where {TT <: Tensor} = one(x) - vol(x)

"""
    vonmises(::AbstractSymmetricSecondOrderTensor{3})

Compute the von Mises stress.

```math
q = \\sqrt{\\frac{3}{2} \\mathrm{dev}(\\bm{\\sigma}) : \\mathrm{dev}(\\bm{\\sigma})} = \\sqrt{3J_2}
```

# Examples
```jldoctest
julia> σ = rand(SymmetricSecondOrderTensor{3})
3×3 SymmetricSecondOrderTensor{3, Float64, 6}:
 0.325977  0.549051  0.218587
 0.549051  0.894245  0.353112
 0.218587  0.353112  0.394255

julia> vonmises(σ)
1.3078860814690232
```
"""
@inline function vonmises(σ::SymmetricSecondOrderTensor{3, T}) where {T}
    s = dev(σ)
    sqrt(T(3/2) * (s ⊡ s))
end

"""
    stress_invariants(::AbstractSecondOrderTensor{3})
    stress_invariants(::AbstractSymmetricSecondOrderTensor{3})
    stress_invariants(::Vec{3})

Return a tuple storing stress invariants.

```math
\\begin{aligned}
I_1(\\bm{\\sigma}) &= \\mathrm{tr}(\\bm{\\sigma}) \\\\
I_2(\\bm{\\sigma}) &= \\frac{1}{2} (\\mathrm{tr}(\\bm{\\sigma})^2 - \\mathrm{tr}(\\bm{\\sigma}^2)) \\\\
I_3(\\bm{\\sigma}) &= \\det(\\bm{\\sigma})
\\end{aligned}
```

# Examples
```jldoctest
julia> σ = rand(SymmetricSecondOrderTensor{3})
3×3 SymmetricSecondOrderTensor{3, Float64, 6}:
 0.325977  0.549051  0.218587
 0.549051  0.894245  0.353112
 0.218587  0.353112  0.394255

julia> I₁, I₂, I₃ = stress_invariants(σ)
(1.6144775244804341, 0.2986572249840249, -0.0025393241133506677)
```

"""
@inline function stress_invariants(σ::AbstractSquareTensor{3})
    trσ = tr(σ)
    I1 = trσ
    I2 = (trσ^2 - tr(σ^2)) / 2
    I3 = det(σ)
    I1, I2, I3
end
@inline function stress_invariants(σ::Vec{3})
    @inbounds σ1, σ2, σ3 = σ
    I1 = σ1 + σ2 + σ3
    I2 = σ1*σ2 + σ2*σ3 + σ1*σ3
    I3 = σ1 * σ2 * σ3
    I1, I2, I3
end

"""
    deviatoric_stress_invariants(::AbstractSecondOrderTensor{3})
    deviatoric_stress_invariants(::AbstractSymmetricSecondOrderTensor{3})
    deviatoric_stress_invariants(::Vec{3})

Return a tuple storing deviatoric stress invariants.

```math
\\begin{aligned}
J_1(\\bm{\\sigma}) &= \\mathrm{tr}(\\mathrm{dev}(\\bm{\\sigma})) = 0 \\\\
J_2(\\bm{\\sigma}) &= \\frac{1}{2} \\mathrm{tr}(\\mathrm{dev}(\\bm{\\sigma})^2) \\\\
J_3(\\bm{\\sigma}) &= \\frac{1}{3} \\mathrm{tr}(\\mathrm{dev}(\\bm{\\sigma})^3)
\\end{aligned}
```

# Examples
```jldoctest
julia> σ = rand(SymmetricSecondOrderTensor{3})
3×3 SymmetricSecondOrderTensor{3, Float64, 6}:
 0.325977  0.549051  0.218587
 0.549051  0.894245  0.353112
 0.218587  0.353112  0.394255

julia> J₁, J₂, J₃ = deviatoric_stress_invariants(σ)
(0.0, 0.5701886673667987, 0.14845380911930367)
```
"""
@inline function deviatoric_stress_invariants(σ::Union{AbstractSquareTensor{3}, Vec{3}})
    I1, I2, I3 = stress_invariants(σ)
    J1 = zero(eltype(σ))
    J2 = I1^2/3 - I2             # tr(s^2) / 2
    J3 = 2I1^3/27 - I1*I2/3 + I3 # tr(s^3) / 3
    J1, J2, J3
end
