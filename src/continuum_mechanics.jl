"""
    mean(::AbstractSecondOrderTensor{3})
    mean(::AbstractSymmetricSecondOrderTensor{3})

Compute the mean value of diagonal entries of a square tensor.

# Examples
```jldoctest
julia> x = rand(Mat{3,3})
3×3 Tensor{Tuple{3, 3}, Float64, 2, 9}:
 0.590845  0.460085  0.200586
 0.766797  0.794026  0.298614
 0.566237  0.854147  0.246837

julia> mean(x)
0.5439025118691712
```
"""
@inline mean(x::AbstractSquareTensor{3}) = tr(x) / 3

"""
    vol(::AbstractSecondOrderTensor{3})
    vol(::AbstractSymmetricSecondOrderTensor{3})

Compute the volumetric part of a square tensor.
Support only for tensors in 3D.

# Examples
```jldoctest
julia> x = rand(Mat{3,3})
3×3 Tensor{Tuple{3, 3}, Float64, 2, 9}:
 0.590845  0.460085  0.200586
 0.766797  0.794026  0.298614
 0.566237  0.854147  0.246837

julia> vol(x)
3×3 Tensor{Tuple{3, 3}, Float64, 2, 9}:
 0.543903  0.0       0.0
 0.0       0.543903  0.0
 0.0       0.0       0.543903

julia> vol(x) + dev(x) ≈ x
true
```
"""
@inline function vol(x::AbstractSquareTensor{3})
    v = mean(x)
    z = zero(v)
    typeof(x)((i,j) -> i == j ? v : z)
end

"""
    dev(::AbstractSecondOrderTensor{3})
    dev(::AbstractSymmetricSecondOrderTensor{3})

Compute the deviatoric part of a square tensor.
Support only for tensors in 3D.

# Examples
```jldoctest
julia> x = rand(Mat{3,3})
3×3 Tensor{Tuple{3, 3}, Float64, 2, 9}:
 0.590845  0.460085  0.200586
 0.766797  0.794026  0.298614
 0.566237  0.854147  0.246837

julia> dev(x)
3×3 Tensor{Tuple{3, 3}, Float64, 2, 9}:
 0.0469421  0.460085   0.200586
 0.766797   0.250123   0.298614
 0.566237   0.854147  -0.297065

julia> tr(dev(x))
0.0
```
"""
@inline dev(x::AbstractSquareTensor{3}) = x - vol(x)

"""
    stress_invariants(::AbstractSecondOrderTensor{3})
    stress_invariants(::AbstractSymmetricSecondOrderTensor{3})

Return `NamedTuple` storing stress invariants.

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
 0.590845  0.766797  0.566237
 0.766797  0.460085  0.794026
 0.566237  0.794026  0.854147

julia> I₁, I₂, I₃ = stress_invariants(σ)
(I1 = 1.9050765715072775, I2 = -0.3695921176777066, I3 = -0.10054272199258936)
```

"""
@inline function stress_invariants(σ::AbstractSquareTensor{3})
    trσ = tr(σ)
    I1 = trσ
    I2 = (trσ^2 - tr(σ^2)) / 2
    I3 = det(σ)
    (; I1, I2, I3)
end

"""
    deviatoric_stress_invariants(::AbstractSecondOrderTensor{3})
    deviatoric_stress_invariants(::AbstractSymmetricSecondOrderTensor{3})

Return `NamedTuple` storing deviatoric stress invariants.

```math
\\begin{aligned}
J_1(\\bm{\\sigma}) &= \\mathrm{tr}(\\mathrm{dev}(\\bm{\\sigma})) = 0 \\\\
J_2(\\bm{\\sigma}) &= -\\frac{1}{2} \\mathrm{tr}(\\mathrm{dev}(\\bm{\\sigma})^2) \\\\
J_3(\\bm{\\sigma}) &= \\frac{1}{3} \\mathrm{tr}(\\mathrm{dev}(\\bm{\\sigma})^3)
\\end{aligned}
```

# Examples
```jldoctest
julia> σ = rand(SymmetricSecondOrderTensor{3})
3×3 SymmetricSecondOrderTensor{3, Float64, 6}:
 0.590845  0.766797  0.566237
 0.766797  0.460085  0.794026
 0.566237  0.794026  0.854147

julia> J₁, J₂, J₃ = deviatoric_stress_invariants(σ)
(J1 = 0.0, J2 = 1.5793643654463476, J3 = 0.6463152097154271)
```
"""
@inline function deviatoric_stress_invariants(σ::AbstractSquareTensor{3})
    I1, I2, I3 = stress_invariants(σ)
    s = dev(σ)
    J1 = zero(eltype(σ))
    J2 = I1^2/3 - I2             # tr(s^2) / 2
    J3 = 2I1^3/27 - I1*I2/3 + I3 # tr(s^3) / 3
    (; J1, J2, J3)
end
