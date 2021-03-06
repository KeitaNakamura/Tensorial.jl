# simd version is defined in simd.jl
@generated function _map(f, xs::Vararg{AbstractTensor, N}) where {N}
    S = promote_space(map(Space, xs)...)
    exps = map(indices(S)) do i
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
@inline Base.:*(y::Real, x::AbstractTensor) = _map(x -> x*y, x)
@inline Base.:*(x::AbstractTensor, y::Real) = _map(x -> x*y, x)
@inline Base.:/(x::AbstractTensor, y::Real) = _map(x -> x/y, x)
@inline Base.:+(x::AbstractTensor) = x
@inline Base.:-(x::AbstractTensor) = _map(-, x)

# with AbstractArray
@generated Base.:+(x::AbstractTensor, y::AbstractArray) = :(@_inline_meta; x + convert($(tensortype(Space(size(x)))), y))
@generated Base.:+(x::AbstractArray, y::AbstractTensor) = :(@_inline_meta; convert($(tensortype(Space(size(y)))), x) + y)
@generated Base.:-(x::AbstractTensor, y::AbstractArray) = :(@_inline_meta; x - convert($(tensortype(Space(size(x)))), y))
@generated Base.:-(x::AbstractArray, y::AbstractTensor) = :(@_inline_meta; convert($(tensortype(Space(size(y)))), x) - y)

@inline _add_uniform(x::AbstractSquareTensor, λ::Real) = x + λ * one(x)
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
    einsum_contraction_expr(Tuple(freeinds), (x, y), (ij_x, ij_y))
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
 1.36912   1.86751  1.32531
 1.61744   2.34426  1.94101
 0.929252  1.89656  1.79015
```

Following symbols are also available for specific contractions:

- `x ⊗ y` (where `⊗` can be typed by `\\otimes<tab>`): `contraction(x, y, Val(0))`
- `x ⋅ y` (where `⋅` can be typed by `\\cdot<tab>`): `contraction(x, y, Val(1))`
- `x ⊡ y` (where `⊡` can be typed by `\\boxdot<tab>`): `contraction(x, y, Val(2))`
"""
@generated function contraction(x::AbstractTensor, y::AbstractTensor, ::Val{N}) where {N}
    TT, exps = contraction_exprs(x, y, N)
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
 0.5908446386657102
 0.7667970365022592
 0.5662374165061859

julia> y = rand(Vec{3})
3-element Vec{3, Float64}:
 0.4600853424625171
 0.7940257103317943
 0.8541465903790502

julia> A = x ⊗ y
3×3 Tensor{Tuple{3, 3}, Float64, 2, 9}:
 0.271839  0.469146  0.504668
 0.352792  0.608857  0.654957
 0.260518  0.449607  0.48365
```
"""
@inline otimes(x1::AbstractTensor, x2::AbstractTensor) = contraction(x1, x2, Val(0))

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
 0.5908446386657102
 0.7667970365022592
 0.5662374165061859

julia> y = rand(Vec{3})
3-element Vec{3, Float64}:
 0.4600853424625171
 0.7940257103317943
 0.8541465903790502

julia> a = x ⋅ y
1.3643452781654772
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
 0.590845  0.460085  0.200586
 0.766797  0.794026  0.298614
 0.566237  0.854147  0.246837

julia> norm(x)
1.7377443667834922
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
 0.590845  0.460085  0.200586
 0.766797  0.794026  0.298614
 0.566237  0.854147  0.246837

julia> tr(x)
1.6317075356075135
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
        exps = [:(Tuple(x)[$($filter(independent_indices(x), Tuple(I)...))]) for I in CartesianIndices(x)]
        TT = SymmetricSecondOrderTensor{dim}
        quote
            @_inline_meta
            @inbounds $TT($(exps[indices(TT)]...))
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
    @inbounds SymmetricSecondOrderTensor{1}(x[1,1])
@inline symmetric(x::AbstractSecondOrderTensor{2}) =
    @inbounds SymmetricSecondOrderTensor{2}(x[1,1], (x[2,1]+x[1,2])/2, x[2,2])
@inline symmetric(x::AbstractSecondOrderTensor{3}) =
    @inbounds SymmetricSecondOrderTensor{3}(x[1,1], (x[2,1]+x[1,2])/2, (x[3,1]+x[1,3])/2, x[2,2], (x[3,2]+x[2,3])/2, x[3,3])

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
@inline transpose(x::AbstractTensor{Tuple{@Symmetry{dim, dim}}}) where {dim} = x
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
 0.5908446386657102
 0.7667970365022592
 0.5662374165061859

julia> y = rand(Vec{3})
3-element Vec{3, Float64}:
 0.4600853424625171
 0.7940257103317943
 0.8541465903790502

julia> x × y
3-element Vec{3, Float64}:
  0.20535000738340053
 -0.24415039787171888
  0.11635375677388776
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
        @inbounds SymmetricSecondOrderTensor{dim}($(exps[indices(x)]...))
    end
end

# rotate
"""
    rotmat(θ::Real; degree::Bool = false)

Construct 2D rotation matrix.

# Examples
```jldoctest
julia> rotmat(30, degree = true)
2×2 Tensor{Tuple{2, 2}, Float64, 2, 4}:
 0.866025  -0.5
 0.5        0.866025
```
"""
@inline function rotmat(θ::Real; degree::Bool = false)
    if degree
        θ = deg2rad(θ)
    end
    sinθ = sin(θ)
    cosθ = cos(θ)
    @Mat [cosθ -sinθ
          sinθ  cosθ]
end

"""
    rotmat(θ::Vec{3}; sequence::Symbol, degree::Bool = false)
    rotmatx(θ::Real)
    rotmaty(θ::Real)
    rotmatz(θ::Real)

Convert Euler angles to rotation matrix.
Use 3 characters belonging to the set (X, Y, Z) for intrinsic rotations,
or (x, y, z) for extrinsic rotations.

# Examples
```jldoctest
julia> α, β, γ = rand(Vec{3});

julia> rotmat(Vec(α,β,γ), sequence = :XYZ) ≈ rotmatx(α) ⋅ rotmaty(β) ⋅ rotmatz(γ)
true

julia> rotmat(Vec(α,β,γ), sequence = :xyz) ≈ rotmatz(γ) ⋅ rotmaty(β) ⋅ rotmatx(α)
true

julia> rotmat(Vec(α,β,γ), sequence = :XYZ) ≈ rotmat(Vec(γ,β,α), sequence = :zyx)
true
```
"""
function rotmat(θ::Vec{3}; sequence::Symbol, degree::Bool = false)
    @inbounds α, β, γ = θ[1], θ[2], θ[3]
    if degree
        α, β, γ = map(deg2rad, (α, β, γ))
    end
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
@inline function rotmatx(θ::Real; degree::Bool = false)
    if degree
        θ = deg2rad(θ)
    end
    o = one(θ)
    z = zero(θ)
    sinθ = sin(θ)
    cosθ = cos(θ)
    @Mat [o z     z
          z cosθ -sinθ
          z sinθ  cosθ]
end
@inline function rotmaty(θ::Real; degree::Bool = false)
    if degree
        θ = deg2rad(θ)
    end
    o = one(θ)
    z = zero(θ)
    sinθ = sin(θ)
    cosθ = cos(θ)
    @Mat [ cosθ z sinθ
           z    o z
          -sinθ z cosθ]
end
@inline function rotmatz(θ::Real; degree::Bool = false)
    if degree
        θ = deg2rad(θ)
    end
    o = one(θ)
    z = zero(θ)
    sinθ = sin(θ)
    cosθ = cos(θ)
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
 0.526847334217759
 0.683741457787621
 0.5049054419691867

julia> b = normalize(rand(Vec{3}))
3-element Vec{3, Float64}:
 0.36698690362212083
 0.6333543148133657
 0.6813097125956302

julia> R = rotmat(a => b)
3×3 Tensor{Tuple{3, 3}, Float64, 2, 9}:
 -0.594528   0.597477   0.538106
  0.597477  -0.119597   0.792917
  0.538106   0.792917  -0.285875

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
    rotmat(θ, n::Vec; degree::Bool = false)

Construct rotation matrix from angle `θ` and direction `n`.

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
 1.1102230246251565e-16
 1.0
 0.0
```
"""
function rotmat(θ::Real, x::Vec{3}; degree::Bool = false)
    if degree
        θ = deg2rad(θ)
    end
    n = normalize(x)
    W = skew(n)
    I + W*sin(θ) + W^2*(1-cos(θ))
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
 0.590845  0.766797  0.566237
 0.766797  0.460085  0.794026
 0.566237  0.794026  0.854147

julia> R = rotmatz(π/4)
3×3 Tensor{Tuple{3, 3}, Float64, 2, 9}:
 0.707107  -0.707107  0.0
 0.707107   0.707107  0.0
 0.0        0.0       1.0

julia> rotate(A, R)
3×3 SymmetricSecondOrderTensor{3, Float64, 6}:
 -0.241332   0.0653796  -0.161071
  0.0653796  1.29226     0.961851
 -0.161071   0.961851    0.854147

julia> R ⋅ A ⋅ R'
3×3 Tensor{Tuple{3, 3}, Float64, 2, 9}:
 -0.241332   0.0653796  -0.161071
  0.0653796  1.29226     0.961851
 -0.161071   0.961851    0.854147
```
"""
@inline rotate(v::Vec, R::SecondOrderTensor) = R ⋅ v
@inline rotate(v::Vec{2}, R::SecondOrderTensor{3}) = (R2x2 = Mat{2, 2}((i,j) -> @inbounds R[i,j]); R2x2 ⋅ v)
@inline rotate(A::SecondOrderTensor, R::SecondOrderTensor) = @einsum R[i,j] * A[j,k] * R[l,k]
@generated function rotate(A::SymmetricSecondOrderTensor{dim}, R::SecondOrderTensor{dim}) where {dim}
    _, exps = contraction_exprs(SecondOrderTensor{dim}, SecondOrderTensor{dim}, 1)
    TT = SymmetricSecondOrderTensor{dim}
    quote
        @_inline_meta
        ARᵀ = @einsum A[i,j] * R[k,j]
        tensors = (R, ARᵀ)
        @inbounds $TT($(exps[indices(TT)]...))
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
@inline function eigvals(x::AbstractSymmetricSecondOrderTensor; permute::Bool = true, scale::Bool = true)
    Tensor(eigvals(Symmetric(SArray(x)); permute, scale))
end
@inline function eigen(x::AbstractSymmetricSecondOrderTensor)
    eig = eigen(Symmetric(SArray(x)))
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

function qr(A::AbstractMat, pivot::Union{Val{false}, Val{true}} = Val(false))
    F = qr(SArray(A), pivot)
    QR(Tensor(F.Q), Tensor(F.R), Tensor(F.p))
end

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

function lu(A::AbstractMat, pivot::Union{Val{false},Val{true}}=Val(true); check = true)
    F = lu(SArray(A), pivot; check)
    LU(LowerTriangular(Tensor(parent(F.L))), UpperTriangular(Tensor(parent(F.U))), Tensor(F.p))
end

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
    F = svd(SArray(A); full)
    SVD(Tensor(F.U), Tensor(F.S), Tensor(F.Vt))
end
