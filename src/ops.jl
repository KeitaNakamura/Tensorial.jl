# simd version is defined in simd.jl
@generated function _map(f, xs::Vararg{AbstractTensor, N}) where {N}
    S = promote_space(map(Space, xs)...)
    exps = map(indices_unique(S)) do i
        vals = [:(xs[$j][$i]) for j in 1:N]
        :(f($(vals...)))
    end
    TT = tensortype(S)
    return quote
        @_inline_meta
        @inbounds $TT($(exps...))
    end
end

@inline Base.:+(x::AbstractTensor, y::AbstractTensor) = _map(+, x, y)
@inline Base.:-(x::AbstractTensor, y::AbstractTensor) = _map(-, x, y)
@inline Base.:*(y::Number, x::AbstractTensor) = _map(x -> x*y, x)
@inline Base.:*(x::AbstractTensor, y::Number) = _map(x -> x*y, x)
@inline Base.:/(x::AbstractTensor, y::Number) = _map(x -> x/y, x)
@inline Base.:+(x::AbstractTensor) = x
@inline Base.:-(x::AbstractTensor) = _map(-, x)

# with AbstractArray
@generated Base.:+(x::AbstractTensor, y::AbstractArray) = :(@_inline_meta; x + convert($(tensortype(Space(size(x)))), y))
@generated Base.:+(x::AbstractArray, y::AbstractTensor) = :(@_inline_meta; convert($(tensortype(Space(size(y)))), x) + y)
@generated Base.:-(x::AbstractTensor, y::AbstractArray) = :(@_inline_meta; x - convert($(tensortype(Space(size(x)))), y))
@generated Base.:-(x::AbstractArray, y::AbstractTensor) = :(@_inline_meta; convert($(tensortype(Space(size(y)))), x) - y)

@inline _add_uniform(x::AbstractSquareTensor, λ::Number) = x + λ * one(x)
@inline Base.:+(x::AbstractTensor, y::UniformScaling) = _add_uniform( x,  y.λ)
@inline Base.:-(x::AbstractTensor, y::UniformScaling) = _add_uniform( x, -y.λ)
@inline Base.:+(x::UniformScaling, y::AbstractTensor) = _add_uniform( y,  x.λ)
@inline Base.:-(x::UniformScaling, y::AbstractTensor) = _add_uniform(-y,  x.λ)
@inline dot(x::Union{AbstractVec, AbstractMat, AbstractSquareTensor}, y::UniformScaling) = x * y.λ
@inline dot(x::UniformScaling, y::Union{AbstractVec, AbstractMat, AbstractSquareTensor}) = x.λ * y
@inline double_contraction(x::AbstractSquareTensor, y::UniformScaling) = tr(x) * y.λ
@inline double_contraction(x::UniformScaling, y::AbstractSquareTensor) = x.λ * tr(y)

# error for standard multiplications
error_multiply() = error("use `⋅` (`\\cdot`) for single contraction and `⊡` (`\\boxdot`) for double contraction instead of `*`")
Base.:*(::AbstractTensor, ::AbstractTensor) = error_multiply()
Base.:*(::AbstractTensor, ::UniformScaling) = error_multiply()
Base.:*(::UniformScaling, ::AbstractTensor) = error_multiply()

function contraction_exprs(x::Type{<: AbstractTensor}, y::Type{<: AbstractTensor}, N::Int)
    nx = ndims(x)
    ny = ndims(y)
    ij_x = UnitRange(1, nx)
    ij_y = UnitRange(nx + 1 - N, nx + ny - N)
    freeinds = find_freeindices([ij_x; ij_y])
    einsum_contraction_expr(freeinds, [x, y], [ij_x, ij_y])
end

"""
    contraction(::AbstractTensor, ::AbstractTensor, ::Val{N})

Conduct contraction of `N` inner indices.
For example, `N=2` contraction for third-order tensors ``A_{ij} = B_{ikl} C_{klj}``
can be computed in Tensorial.jl as

```jldoctest
julia> B = rand(Tensor{Tuple{3,3,3}});

julia> C = rand(Tensor{Tuple{3,3,3}});

julia> A = contraction(B, C, Val(2))
3×3 Tensor{Tuple{3, 3}, Float64, 2, 9}:
 3.70978  2.47156  3.91807
 2.90966  2.30881  3.25965
 1.78391  1.38714  2.2079
```

Following symbols are also available for specific contractions:

- `x ⊗ y` (where `⊗` can be typed by `\\otimes<tab>`): `contraction(x, y, Val(0))`
- `x ⋅ y` (where `⋅` can be typed by `\\cdot<tab>`): `contraction(x, y, Val(1))`
- `x ⊡ y` (where `⊡` can be typed by `\\boxdot<tab>`): `contraction(x, y, Val(2))`
"""
@generated function contraction(x::AbstractTensor, y::AbstractTensor, ::Val{N}) where {N}
    TT, exps = contraction_exprs(x, y, N)
    contraction(Space(x), Space(y), Val(N)) # check dimensions
    quote
        @_inline_meta
        tensors = (x, y)
        @inbounds $TT($(exps...))
    end
end

"""
    otimes(x::AbstractTensor, y::AbstractTensor)
    x ⊗ y

Compute tensor product such as ``A_{ij} = x_i y_j``.
`x ⊗ y` (where `⊗` can be typed by `\\otimes<tab>`) is a synonym for `otimes(x, y)`.

# Examples
```jldoctest
julia> x = rand(Vec{3})
3-element Vec{3, Float64}:
 0.32597672886359486
 0.5490511363155669
 0.21858665481883066

julia> y = rand(Vec{3})
3-element Vec{3, Float64}:
 0.8942454282009883
 0.35311164439921205
 0.39425536741585077

julia> A = x ⊗ y
3×3 Tensor{Tuple{3, 3}, Float64, 2, 9}:
 0.291503  0.115106   0.128518
 0.490986  0.193876   0.216466
 0.19547   0.0771855  0.086179
```
"""
@inline otimes(x1::AbstractTensor, x2::AbstractTensor) = contraction(x1, x2, Val(0))
@inline otimes(x1::AbstractTensor, x2::AbstractTensor, others...) = otimes(otimes(x1, x2), others...)
@inline otimes(x::AbstractTensor) = x

"""
    dot(x::AbstractTensor, y::AbstractTensor)
    x ⋅ y

Compute dot product such as ``a = x_i y_i``.
This is equivalent to [`contraction(::AbstractTensor, ::AbstractTensor, Val(1))`](@ref).
`x ⋅ y` (where `⋅` can be typed by `\\cdot<tab>`) is a synonym for `dot(x, y)`.

# Examples
```jldoctest
julia> x = rand(Vec{3})
3-element Vec{3, Float64}:
 0.32597672886359486
 0.5490511363155669
 0.21858665481883066

julia> y = rand(Vec{3})
3-element Vec{3, Float64}:
 0.8942454282009883
 0.35311164439921205
 0.39425536741585077

julia> a = x ⋅ y
0.5715585109976284
```
"""
@inline dot(x1::AbstractTensor, x2::AbstractTensor) = contraction(x1, x2, Val(1))
@inline double_contraction(x1::AbstractTensor, x2::AbstractTensor) = contraction(x1, x2, Val(2))

"""
    norm(::AbstractTensor)

Compute norm of a tensor.

# Examples
```jldoctest
julia> x = rand(Mat{3, 3})
3×3 Tensor{Tuple{3, 3}, Float64, 2, 9}:
 0.325977  0.894245  0.953125
 0.549051  0.353112  0.795547
 0.218587  0.394255  0.49425

julia> norm(x)
1.8223398556552728
```
"""
@inline norm(x::AbstractTensor) = sqrt(contraction(x, x, Val(ndims(x))))

@inline normalize(x::AbstractTensor) = x / norm(x)

# v_k * S_ikjl * u_l
@inline function dotdot(v1::Vec{dim}, S::SymmetricFourthOrderTensor{dim}, v2::Vec{dim}) where {dim}
    S′ = SymmetricFourthOrderTensor{dim}((i,j,k,l) -> @inbounds S[j,i,l,k])
    v1 ⋅ S′ ⋅ v2
end

"""
    tr(::AbstractSecondOrderTensor)
    tr(::AbstractSymmetricSecondOrderTensor)

Compute the trace of a square tensor.

# Examples
```jldoctest
julia> x = rand(Mat{3,3})
3×3 Tensor{Tuple{3, 3}, Float64, 2, 9}:
 0.325977  0.894245  0.953125
 0.549051  0.353112  0.795547
 0.218587  0.394255  0.49425

julia> tr(x)
1.1733382401532275
```
"""
@inline tr(x::AbstractSquareTensor) = @einsum x[i,i]

# symmetric
@inline symmetric(x::AbstractSymmetricSecondOrderTensor) = x
@inline function symmetric(x::AbstractSymmetricSecondOrderTensor, uplo::Symbol)
    if uplo == :U || uplo == :L
        return x
    end
    throw(ArgumentError("uplo argument must be either :U (upper) or :L (lower)"))
end
## :U or :L
for (uplo, filter) in ((:U, (inds,i,j) -> i ≤ j ? inds[i,j] : inds[j,i]),
                       (:L, (inds,i,j) -> i ≥ j ? inds[i,j] : inds[j,i]))
    @eval @generated function $(Symbol(:symmetric, uplo))(x::AbstractSecondOrderTensor{dim}) where {dim}
        exps = [:(Tuple(x)[$($filter(indices_all(x), Tuple(I)...))]) for I in CartesianIndices(x)]
        TT = SymmetricSecondOrderTensor{dim}
        quote
            @_inline_meta
            @inbounds $TT($(exps[indices_unique(TT)]...))
        end
    end
end
@inline function symmetric(x::AbstractSecondOrderTensor{dim}, uplo::Symbol) where {dim}
    uplo == :U && return symmetricU(x)
    uplo == :L && return symmetricL(x)
    throw(ArgumentError("uplo argument must be either :U (upper) or :L (lower)"))
end
## no uplo
@inline symmetric(x::AbstractSecondOrderTensor{dim}) where {dim} =
    SymmetricSecondOrderTensor{dim}((i,j) -> @inbounds i == j ? x[i,j] : (x[i,j] + x[j,i]) / 2)
@inline symmetric(x::AbstractSecondOrderTensor{1}) =
    @inbounds SymmetricSecondOrderTensor{1}(x[1])
@inline symmetric(x::AbstractSecondOrderTensor{2}) =
    @inbounds SymmetricSecondOrderTensor{2}(x[1], (x[2]+x[3])/2, x[4])
@inline symmetric(x::AbstractSecondOrderTensor{3}) =
    @inbounds SymmetricSecondOrderTensor{3}(x[1], (x[2]+x[4])/2, (x[3]+x[7])/2, x[5], (x[6]+x[8])/2, x[9])

"""
    skew(::AbstractSecondOrderTensor)
    skew(::AbstractSymmetricSecondOrderTensor)

Compute skew-symmetric (anti-symmetric) part of a second order tensor.
"""
@inline skew(x::AbstractSecondOrderTensor) = (x - x') / 2
@inline skew(x::AbstractSymmetricSecondOrderTensor{dim, T}) where {dim, T} = zero(SecondOrderTensor{dim, T})

"""
    skew(ω::Vec{3})

Construct a skew-symmetric (anti-symmetric) tensor `W` from a vector `ω` as

```math
\\bm{\\omega} = \\begin{Bmatrix}
    \\omega_1 \\\\
    \\omega_2 \\\\
    \\omega_3
\\end{Bmatrix}, \\quad
\\bm{W} = \\begin{bmatrix}
     0         & -\\omega_3 &  \\omega_2 \\\\
     \\omega_3 & 0          & -\\omega_1 \\\\
    -\\omega_2 &  \\omega_1 &  0
\\end{bmatrix}
```

# Examples
```jldoctest
julia> skew(Vec(1,2,3))
3×3 Tensor{Tuple{3, 3}, Int64, 2, 9}:
  0  -3   2
  3   0  -1
 -2   1   0
```
"""
@inline function skew(ω::Vec{3})
    z = zero(eltype(ω))
    @inbounds @Mat [ z    -ω[3]  ω[2]
                     ω[3]     z -ω[1]
                    -ω[2]  ω[1]     z]
end

# transpose/adjoint
@inline transpose(x::AbstractTensor{Tuple{@Symmetry({dim, dim})}}) where {dim} = x
@inline transpose(x::AbstractTensor{Tuple{m, n}}) where {m, n} = Tensor{Tuple{n, m}}((i,j) -> @inbounds x[j,i])
@inline adjoint(x::AbstractTensor) = transpose(x)

# det
@generated function extract_vecs(x::AbstractSquareTensor{dim}) where {dim}
    exps = map(1:dim) do j
        :(Vec($([getindex_expr(x, :x, i, j) for i in 1:dim]...)))
    end
    quote
        @_inline_meta
        @inbounds tuple($(exps...))
    end
end
@inline function det(x::AbstractSquareTensor{1})
    @inbounds x[1,1]
end
@inline function det(x::AbstractSquareTensor{2})
    @inbounds x[1,1] * x[2,2] - x[1,2] * x[2,1]
end
@inline function det(A::AbstractSquareTensor{3})
    a1, a2, a3 = extract_vecs(A)
    (a1 × a2) ⋅ a3
end
@inline function det(x::AbstractSquareTensor{dim}) where {dim}
    det(SMatrix{dim, dim}(x))
end

"""
    cross(x::Vec{3}, y::Vec{3}) -> Vec{3}
    cross(x::Vec{2}, y::Vec{2}) -> Vec{3}
    cross(x::Vec{1}, y::Vec{1}) -> Vec{3}
    x × y

Compute the cross product between two vectors.
The vectors are expanded to 3D frist for dimensions 1 and 2.
The infix operator `×` (written `\\times`) can also be used.
`x × y` (where `×` can be typed by `\\times<tab>`) is a synonym for `cross(x, y)`.

# Examples
```jldoctest
julia> x = rand(Vec{3})
3-element Vec{3, Float64}:
 0.32597672886359486
 0.5490511363155669
 0.21858665481883066

julia> y = rand(Vec{3})
3-element Vec{3, Float64}:
 0.8942454282009883
 0.35311164439921205
 0.39425536741585077

julia> x × y
3-element Vec{3, Float64}:
  0.13928086435138393
  0.0669520417303531
 -0.37588028973385323
```
"""
@inline cross(x::Vec{1, T1}, y::Vec{1, T2}) where {T1, T2} = zero(Vec{3, promote_type(T1, T2)})
@inline function cross(x::Vec{2, T1}, y::Vec{2, T2}) where {T1, T2}
    z = zero(promote_type(T1, T2))
    @inbounds Vec(z, z, x[1]*y[2] - x[2]*y[1])
end
@inline function cross(x::Vec{3}, y::Vec{3})
    @inbounds Vec(x[2]*y[3] - x[3]*y[2],
                  x[3]*y[1] - x[1]*y[3],
                  x[1]*y[2] - x[2]*y[1])
end

# power
@inline Base.literal_pow(::typeof(^), x::AbstractSquareTensor, ::Val{-1}) = inv(x)
@inline Base.literal_pow(::typeof(^), x::AbstractSquareTensor, ::Val{0})  = one(x)
@inline Base.literal_pow(::typeof(^), x::AbstractSquareTensor, ::Val{1})  = x
@inline function Base.literal_pow(::typeof(^), x::AbstractSquareTensor, ::Val{p}) where {p}
    p > 0 ? (y = x; q = p) : (y = inv(x); q = -p)
    z = y
    for i in 2:q
        y = _powdot(y, z)
    end
    y
end
## helper functions
@inline _powdot(x::AbstractSecondOrderTensor, y::AbstractSecondOrderTensor) = dot(x, y)
@generated function _powdot(x::AbstractSymmetricSecondOrderTensor{dim}, y::AbstractSymmetricSecondOrderTensor{dim}) where {dim}
    _, exps = contraction_exprs(x, y, 1)
    quote
        @_inline_meta
        tensors = (x, y)
        @inbounds SymmetricSecondOrderTensor{dim}($(exps[indices_unique(x)]...))
    end
end

# rotate
"""
    rotmat(θ::Number)

Construct 2D rotation matrix.

```math
\\bm{R} = \\begin{bmatrix}
\\cos{\\theta} & -\\sin{\\theta} \\\\
\\sin{\\theta} &  \\cos{\\theta}
\\end{bmatrix}
```

# Examples
```jldoctest
julia> rotmat(deg2rad(30))
2×2 Tensor{Tuple{2, 2}, Float64, 2, 4}:
 0.866025  -0.5
 0.5        0.866025
```
"""
@inline function rotmat(θ::Number)
    sinθ, cosθ = sincos(θ)
    @Mat [cosθ -sinθ
          sinθ  cosθ]
end

"""
    rotmat(θ::Vec{3}; sequence::Symbol)

Convert Euler angles to rotation matrix.
Use 3 characters belonging to the set (X, Y, Z) for intrinsic rotations,
or (x, y, z) for extrinsic rotations.

# Examples
```jldoctest
julia> α, β, γ = map(deg2rad, rand(3));

julia> rotmat(Vec(α,β,γ), sequence = :XYZ) ≈ rotmatx(α) ⋅ rotmaty(β) ⋅ rotmatz(γ)
true

julia> rotmat(Vec(α,β,γ), sequence = :xyz) ≈ rotmatz(γ) ⋅ rotmaty(β) ⋅ rotmatx(α)
true

julia> rotmat(Vec(α,β,γ), sequence = :XYZ) ≈ rotmat(Vec(γ,β,α), sequence = :zyx)
true
```
"""
function rotmat(θ::Vec{3}; sequence::Symbol)
    @inbounds α, β, γ = θ[1], θ[2], θ[3]
    # intrinsic
    sequence == :XZX && return rotmatx(α) ⋅ rotmaty(β) ⋅ rotmatz(γ)
    sequence == :XYX && return rotmatx(α) ⋅ rotmaty(β) ⋅ rotmatx(γ)
    sequence == :YXY && return rotmaty(α) ⋅ rotmatx(β) ⋅ rotmaty(γ)
    sequence == :YZY && return rotmaty(α) ⋅ rotmatz(β) ⋅ rotmaty(γ)
    sequence == :ZYZ && return rotmatz(α) ⋅ rotmaty(β) ⋅ rotmatz(γ)
    sequence == :ZXZ && return rotmatz(α) ⋅ rotmatx(β) ⋅ rotmatz(γ)
    sequence == :XZY && return rotmatx(α) ⋅ rotmatz(β) ⋅ rotmaty(γ)
    sequence == :XYZ && return rotmatx(α) ⋅ rotmaty(β) ⋅ rotmatz(γ)
    sequence == :YXZ && return rotmaty(α) ⋅ rotmatx(β) ⋅ rotmatz(γ)
    sequence == :YZX && return rotmaty(α) ⋅ rotmatz(β) ⋅ rotmatx(γ)
    sequence == :ZYX && return rotmatz(α) ⋅ rotmaty(β) ⋅ rotmatx(γ)
    sequence == :ZXY && return rotmatz(α) ⋅ rotmatx(β) ⋅ rotmaty(γ)
    # extrinsic
    sequence == :xzx && return rotmatx(γ) ⋅ rotmaty(β) ⋅ rotmatz(α)
    sequence == :xyx && return rotmatx(γ) ⋅ rotmaty(β) ⋅ rotmatx(α)
    sequence == :yxy && return rotmaty(γ) ⋅ rotmatx(β) ⋅ rotmaty(α)
    sequence == :yzy && return rotmaty(γ) ⋅ rotmatz(β) ⋅ rotmaty(α)
    sequence == :zyz && return rotmatz(γ) ⋅ rotmaty(β) ⋅ rotmatz(α)
    sequence == :zxz && return rotmatz(γ) ⋅ rotmatx(β) ⋅ rotmatz(α)
    sequence == :xzy && return rotmaty(γ) ⋅ rotmatz(β) ⋅ rotmatx(α)
    sequence == :xyz && return rotmatz(γ) ⋅ rotmaty(β) ⋅ rotmatx(α)
    sequence == :yxz && return rotmatz(γ) ⋅ rotmatx(β) ⋅ rotmaty(α)
    sequence == :yzx && return rotmatx(γ) ⋅ rotmatz(β) ⋅ rotmaty(α)
    sequence == :zyx && return rotmatx(γ) ⋅ rotmaty(β) ⋅ rotmatz(α)
    sequence == :zxy && return rotmaty(γ) ⋅ rotmatx(β) ⋅ rotmatz(α)
    throw(ArgumentError("sequence $sequence is not supported"))
end

"""
    rotmatx(θ::Number)

Construct rotation matrix around `x` axis.

```math
\\bm{R}_x = \\begin{bmatrix}
1 & 0 & 0 \\\\
0 & \\cos{\\theta} & -\\sin{\\theta} \\\\
0 & \\sin{\\theta} &  \\cos{\\theta}
\\end{bmatrix}
```
"""
@inline function rotmatx(θ::Number)
    o = one(θ)
    z = zero(θ)
    sinθ, cosθ = sincos(θ)
    @Mat [o z     z
          z cosθ -sinθ
          z sinθ  cosθ]
end

"""
    rotmaty(θ::Number)

Construct rotation matrix around `y` axis.

```math
\\bm{R}_y = \\begin{bmatrix}
\\cos{\\theta} & 0 & \\sin{\\theta} \\\\
0 & 1 & 0 \\\\
-\\sin{\\theta} & 0 & \\cos{\\theta}
\\end{bmatrix}
```
"""
@inline function rotmaty(θ::Number)
    o = one(θ)
    z = zero(θ)
    sinθ, cosθ = sincos(θ)
    @Mat [ cosθ z sinθ
           z    o z
          -sinθ z cosθ]
end

"""
    rotmatz(θ::Number)

Construct rotation matrix around `z` axis.

```math
\\bm{R}_z = \\begin{bmatrix}
\\cos{\\theta} & -\\sin{\\theta} & 0 \\\\
\\sin{\\theta} &  \\cos{\\theta} & 0 \\\\
0 & 0 & 1
\\end{bmatrix}
```
"""
@inline function rotmatz(θ::Number)
    o = one(θ)
    z = zero(θ)
    sinθ, cosθ = sincos(θ)
    @Mat [cosθ -sinθ z
          sinθ  cosθ z
          z     z    o]
end

"""
    rotmat(a => b)

Construct rotation matrix rotating vector `a` to `b`.
The norms of two vectors must be the same.

# Examples
```jldoctest
julia> a = normalize(rand(Vec{3}))
3-element Vec{3, Float64}:
 0.4829957515506539
 0.8135223859352438
 0.3238771859304809

julia> b = normalize(rand(Vec{3}))
3-element Vec{3, Float64}:
 0.8605677447967596
 0.3398133016944055
 0.3794075336718636

julia> R = rotmat(a => b)
3×3 Tensor{Tuple{3, 3}, Float64, 2, 9}:
 -0.00540771   0.853773   0.520617
  0.853773    -0.267108   0.446905
  0.520617     0.446905  -0.727485

julia> R ⋅ a ≈ b
true
```
"""
function rotmat(pair::Pair{Vec{dim, T}, Vec{dim, T}})::Mat{dim, dim, T} where {dim, T}
    # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d/2672702#2672702
    a = pair.first
    b = pair.second
    dot(a, a) ≈ dot(b, b) || throw(ArgumentError("the norms of two vectors must be the same"))
    a ==  b && return  one(Mat{dim, dim, T})
    a == -b && return -one(Mat{dim, dim, T})
    c = a + b
    2 * (c ⊗ c) / (c ⋅ c) - one(Mat{dim, dim, T})
end

"""
    rotmat(θ, n::Vec)

Construct rotation matrix from angle `θ` and axis `n`.

# Examples
```jldoctest
julia> x = Vec(1.0, 0.0, 0.0)
3-element Vec{3, Float64}:
 1.0
 0.0
 0.0

julia> n = Vec(0.0, 0.0, 1.0)
3-element Vec{3, Float64}:
 0.0
 0.0
 1.0

julia> rotmat(π/2, n) ⋅ x
3-element Vec{3, Float64}:
 6.123233995736766e-17
 1.0
 0.0
```
"""
function rotmat(θ::Number, n::Vec{3})
    n′ = normalize(n)
    sinθ, cosθ = sincos(θ)
    cosθ*I + sinθ*skew(n′) + (1-cosθ)*(n′⊗n′)
end

"""
    rotate(x::Vec, R::SecondOrderTensor)
    rotate(x::SecondOrderTensor, R::SecondOrderTensor)
    rotate(x::SymmetricSecondOrderTensor, R::SecondOrderTensor)

Rotate `x` by rotation matrix `R`.
This function can hold the symmetry of `SymmetricSecondOrderTensor`.

# Examples
```jldoctest
julia> A = rand(SymmetricSecondOrderTensor{3})
3×3 SymmetricSecondOrderTensor{3, Float64, 6}:
 0.325977  0.549051  0.218587
 0.549051  0.894245  0.353112
 0.218587  0.353112  0.394255

julia> R = rotmatz(π/4)
3×3 Tensor{Tuple{3, 3}, Float64, 2, 9}:
 0.707107  -0.707107  0.0
 0.707107   0.707107  0.0
 0.0        0.0       1.0

julia> rotate(A, R)
3×3 SymmetricSecondOrderTensor{3, Float64, 6}:
  0.0610599  -0.284134  -0.0951235
 -0.284134    1.15916    0.404252
 -0.0951235   0.404252   0.394255

julia> R ⋅ A ⋅ R'
3×3 Tensor{Tuple{3, 3}, Float64, 2, 9}:
  0.0610599  -0.284134  -0.0951235
 -0.284134    1.15916    0.404252
 -0.0951235   0.404252   0.394255
```
"""
@inline rotate(v::Vec, R::SecondOrderTensor) = R ⋅ v
@inline rotate(v::Vec{2}, R::SecondOrderTensor{3}) = rotate(vcat(v,0), R) # extend to 3d vector, then rotate it
@inline rotate(A::SecondOrderTensor, R::SecondOrderTensor) = @einsum R[i,j] * A[j,k] * R[l,k]
@generated function rotate(A::SymmetricSecondOrderTensor{dim}, R::SecondOrderTensor{dim}) where {dim}
    _, exps = contraction_exprs(SecondOrderTensor{dim}, SecondOrderTensor{dim}, 1)
    TT = SymmetricSecondOrderTensor{dim}
    quote
        @_inline_meta
        ARᵀ = @einsum A[i,j] * R[k,j]
        tensors = (R, ARᵀ)
        @inbounds $TT($(exps[indices_unique(TT)]...))
    end
end

function angleaxis(R::SecondOrderTensor{3})
    # https://math.stackexchange.com/questions/893984/conversion-of-rotation-matrix-to-quaternion
    θ = acos((tr(R)-1) / 2)
    n = Tensor(Real.(eigvecs(SArray(R))[:,3]))
    θ, n
end


# ----------------------------------------------#
# operations calling methods in StaticArrays.jl #
# ----------------------------------------------#

# eigvals/eigen
@inline function eigvals(x::AbstractSymmetricSecondOrderTensor; permute::Bool=true, scale::Bool=true)
    Tensor(eigvals(Symmetric(SArray(x)); permute=permute, scale = scale))
end
@inline function eigen(x::AbstractSymmetricSecondOrderTensor; permute::Bool=true, scale::Bool=true)
    _eig(x; permute=permute, scale=scale)
end
# special implementation for 3x3 case (https://hal.science/hal-01501221/document)
@inline function eigen(x::AbstractSymmetricSecondOrderTensor{3}; permute::Bool=true, scale::Bool=true)
    a, d, f, b, e, c = Tuple(x)
    iszero(f) && return _eig(x; permute=permute, scale=scale)

    λ₁, λ₂, λ₃ = eigvals(x; permute=permute, scale=scale)

    isapproxzero(x) = abs(x) < sqrt(eps(typeof(x)))
    if isapproxzero(λ₁) || isapproxzero(λ₂) || isapproxzero(λ₃) ||
       isapproxzero(λ₁-λ₂) || isapproxzero(λ₁-λ₃) || isapproxzero(λ₂-λ₃)
        return _eig(x; permute=permute, scale=scale)
    end

    v₁, v₂, v₃ = map((λ₁, λ₂, λ₃)) do λ
        m = (d*(c-λ) - e*f) / (f*(b-λ) - d*e)
        Vec((λ-c-e*m)/f, m, 1)
    end
    if !isfinite(v₁[2]) || !isfinite(v₂[2]) || !isfinite(v₃[2])
        return _eig(x; permute=permute, scale=scale)
    end

    values = Vec(λ₁, λ₂, λ₃)
    vectors = hcat(normalize(v₁), normalize(v₂), normalize(v₃))
    Eigen(values, vectors)
end
# fallback to StaticArrays.jl
@inline function _eig(x::AbstractSymmetricSecondOrderTensor; permute::Bool, scale::Bool)
    eig = eigen(Symmetric(SArray(x)); permute=permute, scale=scale)
    Eigen(Tensor(eig.values), Tensor(eig.vectors))
end

# exp
@inline Base.exp(x::AbstractSecondOrderTensor) = typeof(x)(exp(SArray(x)))
@inline Base.exp(x::AbstractSymmetricSecondOrderTensor) = typeof(x)(exp(Symmetric(SArray(x))))

# diag/diagm
@inline diag(x::Union{AbstractMat, AbstractSymmetricSecondOrderTensor}, ::Val{k} = Val(0)) where {k} = Tensor(diag(SArray(x), Val{k}))
@inline diagm(kvs::Pair{<: Val, <: AbstractVec}...) = Tensor(diagm(map(kv -> kv.first => SArray(kv.second), kvs)...))
@inline diagm(x::Vec) = diagm(Val(0) => x)

# qr
struct QR{Q, R, P}
    Q::Q
    R::R
    p::P
end

# iteration for destructuring into components
Base.iterate(S::QR) = (S.Q, Val(:R))
Base.iterate(S::QR, ::Val{:R}) = (S.R, Val(:p))
Base.iterate(S::QR, ::Val{:p}) = (S.p, Val(:done))
Base.iterate(S::QR, ::Val{:done}) = nothing

for pv in (:true, :false)
    @eval function qr(A::AbstractMat, pivot::Val{$pv})
        F = qr(SArray(A), pivot)
        QR(Tensor(F.Q), Tensor(F.R), Tensor(F.p))
    end
end
qr(A::AbstractMat) = qr(A, Val(false))

# lu
struct LU{L, U, p}
    L::L
    U::U
    p::p
end

# iteration for destructuring into components
Base.iterate(S::LU) = (S.L, Val(:U))
Base.iterate(S::LU, ::Val{:U}) = (S.U, Val(:p))
Base.iterate(S::LU, ::Val{:p}) = (S.p, Val(:done))
Base.iterate(S::LU, ::Val{:done}) = nothing

for pv in (:true, :false)
    @eval function lu(A::AbstractMat, pivot::Val{$pv}; check = true)
        F = lu(SArray(A), pivot; check = check)
        LU(LowerTriangular(Tensor(parent(F.L))), UpperTriangular(Tensor(parent(F.U))), Tensor(F.p))
    end
end
lu(A::AbstractMat; check = true) = lu(A, Val(true); check = check)

# svd
struct SVD{T, TU, TS, TVt} <: Factorization{T}
    U::TU
    S::TS
    Vt::TVt
end
SVD(U::AbstractArray{T}, S::AbstractVector, Vt::AbstractArray{T}) where {T} = SVD{T, typeof(U), typeof(S), typeof(Vt)}(U, S, Vt)

@inline function Base.getproperty(F::SVD, s::Symbol)
    if s === :V
        return getfield(F, :Vt)'
    else
        return getfield(F, s)
    end
end
Base.propertynames(::SVD) = (:U, :S, :V, :Vt)

# iteration for destructuring into components
Base.iterate(S::SVD) = (S.U, Val(:S))
Base.iterate(S::SVD, ::Val{:S}) = (S.S, Val(:V))
Base.iterate(S::SVD, ::Val{:V}) = (S.V, Val(:done))
Base.iterate(S::SVD, ::Val{:done}) = nothing

function svd(A::AbstractMat; full=Val(false))
    F = svd(SArray(A); full = full)
    SVD(Tensor(F.U), Tensor(F.S), Tensor(F.Vt))
end
