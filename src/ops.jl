@generated function _map(f, xs::Vararg{AbstractTensor, N}) where {N}
    S = promote_size(map(Size, xs)...)
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

@generated function _add_uniform(x::AbstractSquareTensor{dim}, λ::Real) where {dim}
    S = promote_size(Size(x), Size(Symmetry(dim, dim)))
    tocartesian = CartesianIndices(S)
    exps = map(indices(S)) do i
        i, j = Tuple(tocartesian[i])
        ex = getindex_expr(:x, x, i, j)
        return i == j ? :($ex + λ) : ex
    end
    TT = tensortype(S)
    return quote
        @_inline_meta
        @inbounds $TT($(exps...))
    end
end

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

function contraction_exprs(S1::Size, S2::Size, ::Val{N}) where {N}
    S = contraction(S1, S2, Val(N))
    s1 = map(i -> EinsumIndex(:(Tuple(x)), i), independent_indices(S1))
    s2 = map(i -> EinsumIndex(:(Tuple(y)), i), independent_indices(S2))
    J = prod(size(s2)[1:N])
    I = length(s1) ÷ J
    K = length(s2) ÷ J
    s1′ = reshape(s1, I, J)
    s2′ = reshape(s2, J, K)
    s′ = @einsum (i,k) -> s1′[i,j] * s2′[j,k]
    s = reshape(s′, size(s1)[1:end-N]..., size(s2)[N+1:end]...)
    map(construct_expr, s[indices(S)])
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
3×3 Tensor{Tuple{3,3},Float64,2,9}:
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
    S = contraction(Size(x), Size(y), Val(N))
    exps = contraction_exprs(Size(x), Size(y), Val(N))
    T = promote_type(eltype(x), eltype(y))
    if length(S) == 0
        TT = T
    else
        TT = tensortype(S){T}
    end
    quote
        @_inline_meta
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
julia> x = rand(Vec{3});

julia> y = rand(Vec{3});

julia> A = x ⊗ y
3×3 Tensor{Tuple{3,3},Float64,2,9}:
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
`x ⋅ y` (where `⋅` can be typed by `\\cdot<tab>`) is a synonym for `cdot(x, y)`.

# Examples
```jldoctest
julia> x = rand(Vec{3});

julia> y = rand(Vec{3});

julia> a = x ⋅ y
1.3643452781654775
```
"""
@inline dot(x1::AbstractTensor, x2::AbstractTensor) = contraction(x1, x2, Val(1))
@inline double_contraction(x1::AbstractTensor, x2::AbstractTensor) = contraction(x1, x2, Val(2))
@inline norm(x::AbstractTensor) = sqrt(contraction(x, x, Val(ndims(x))))

# v_k * S_ikjl * u_l
@generated function dotdot(v1::Vec{dim}, S::SymmetricFourthOrderTensor{dim}, v2::Vec{dim}) where {dim}
    v1inds = map(i -> EinsumIndex(:(Tuple(v1)), i), independent_indices(v1))
    v2inds = map(i -> EinsumIndex(:(Tuple(v2)), i), independent_indices(v2))
    Sinds = map(i -> EinsumIndex(:(Tuple(S)), i), independent_indices(S))
    exps = @einsum (i,j) -> v1inds[k] * Sinds[i,k,j,l] * v2inds[l]
    TT = Tensor{Tuple{dim, dim}}
    quote
        @_inline_meta
        @inbounds $TT($(map(construct_expr, exps)...))
    end
end

# tr/mean
@inline function tr(x::AbstractSquareTensor{dim}) where {dim}
    sum(i -> @inbounds(x[i,i]), 1:dim)
end
@inline tr(x::AbstractSquareTensor{1}) = @inbounds x[1,1]
@inline tr(x::AbstractSquareTensor{2}) = @inbounds x[1,1] + x[2,2]
@inline tr(x::AbstractSquareTensor{3}) = @inbounds x[1,1] + x[2,2] + x[3,3]
@inline mean(x::AbstractSquareTensor{dim}) where {dim} = tr(x) / dim

# vol/dev
@inline function vol(x::AbstractSquareTensor{3})
    v = mean(x)
    z = zero(v)
    typeof(x)((i,j) -> i == j ? v : z)
end
@inline dev(x::AbstractSquareTensor{3}) = x - vol(x)

"""
    skew(::AbstractSecondOrderTensor)
    skew(::AbstractSymmetricSecondOrderTensor)

Compute skew-symmetric (anti-symmetric) part of a second order tensor.
"""
@inline skew(x::AbstractSecondOrderTensor) = (x - x') / 2
@inline skew(x::AbstractSymmetricSecondOrderTensor{dim, T}) where {dim, T} = zero(SecondOrderTensor{dim, T})

# transpose/adjoint
@inline transpose(x::AbstractTensor{Tuple{@Symmetry{dim, dim}}}) where {dim} = x
@inline transpose(x::AbstractTensor{Tuple{m, n}}) where {m, n} = Tensor{Tuple{n, m}}((i,j) -> @inbounds x[j,i])
@inline adjoint(x::AbstractTensor) = transpose(x)

# symmetric
@inline symmetric(x::AbstractSymmetricSecondOrderTensor) = x
@inline symmetric(x::AbstractSecondOrderTensor{dim}) where {dim} =
    SymmetricSecondOrderTensor{dim}((i,j) -> @inbounds i == j ? x[i,j] : (x[i,j] + x[j,i]) / 2)
@inline symmetric(x::AbstractSecondOrderTensor{1}) =
    @inbounds SymmetricSecondOrderTensor{1}(x[1,1])
@inline symmetric(x::AbstractSecondOrderTensor{2}) =
    @inbounds SymmetricSecondOrderTensor{2}(x[1,1], (x[2,1]+x[1,2])/2, x[2,2])
@inline symmetric(x::AbstractSecondOrderTensor{3}) =
    @inbounds SymmetricSecondOrderTensor{3}(x[1,1], (x[2,1]+x[1,2])/2, (x[3,1]+x[1,3])/2, x[2,2], (x[3,2]+x[2,3])/2, x[3,3])

# det
@inline function det(x::AbstractSquareTensor{1})
    @inbounds x[1,1]
end
@inline function det(x::AbstractSquareTensor{2})
    @inbounds x[1,1] * x[2,2] - x[1,2] * x[2,1]
end
@inline function det(x::AbstractSquareTensor{3})
    @inbounds (x[1,1] * (x[2,2]*x[3,3] - x[2,3]*x[3,2]) -
               x[1,2] * (x[2,1]*x[3,3] - x[2,3]*x[3,1]) +
               x[1,3] * (x[2,1]*x[3,2] - x[2,2]*x[3,1]))
end
@inline function det(x::AbstractSquareTensor)
    det(Matrix(x))
end

# inv
@inline function inv(x::AbstractSquareTensor{1})
    typeof(x)(1/det(x))
end
@generated function inv(x::AbstractSquareTensor{2})
    x_11 = getindex_expr(:x, x, 1, 1)
    x_21 = getindex_expr(:x, x, 2, 1)
    x_12 = getindex_expr(:x, x, 1, 2)
    x_22 = getindex_expr(:x, x, 2, 2)
    exps = [:($x_22 * detinv), :(-$x_21 * detinv), :(-$x_12 * detinv), :($x_11 * detinv)]
    return quote
        @_inline_meta
        detinv = 1 / det(x)
        @inbounds typeof(x)($(exps[indices(x)]...))
    end
end
@generated function inv(x::AbstractSquareTensor{3})
    x_11 = getindex_expr(:x, x, 1, 1)
    x_21 = getindex_expr(:x, x, 2, 1)
    x_31 = getindex_expr(:x, x, 3, 1)
    x_12 = getindex_expr(:x, x, 1, 2)
    x_22 = getindex_expr(:x, x, 2, 2)
    x_32 = getindex_expr(:x, x, 3, 2)
    x_13 = getindex_expr(:x, x, 1, 3)
    x_23 = getindex_expr(:x, x, 2, 3)
    x_33 = getindex_expr(:x, x, 3, 3)
    exps = [:(($x_22*$x_33 - $x_23*$x_32) * detinv),
            :(($x_23*$x_31 - $x_21*$x_33) * detinv),
            :(($x_21*$x_32 - $x_22*$x_31) * detinv),
            :(($x_13*$x_32 - $x_12*$x_33) * detinv),
            :(($x_11*$x_33 - $x_13*$x_31) * detinv),
            :(($x_12*$x_31 - $x_11*$x_32) * detinv),
            :(($x_12*$x_23 - $x_13*$x_22) * detinv),
            :(($x_13*$x_21 - $x_11*$x_23) * detinv),
            :(($x_11*$x_22 - $x_12*$x_21) * detinv)]
    return quote
        @_inline_meta
        detinv = 1 / det(x)
        @inbounds typeof(x)($(exps[indices(x)]...))
    end
end
@inline function inv(x::AbstractSquareTensor)
    typeof(x)(inv(Matrix(x)))
end
@inline function inv(x::FourthOrderTensor{dim}) where {dim}
    @assert dim < 4
    fromvoigt(FourthOrderTensor{dim}, inv(tovoigt(x)))
end
@inline function inv(x::SymmetricFourthOrderTensor{dim}) where {dim}
    @assert dim < 4
    frommandel(SymmetricFourthOrderTensor{dim}, inv(tomandel(x)))
end

# cross
@inline cross(x::Vec{1}, y::Vec{1}) =
    zero(Vec{3, promote_type(eltype(x), eltype(y))})
@inline cross(x::Vec{2, T1}, y::Vec{2, T2}) where {T1, T2} =
    @inbounds Vec{3}((zero(T1)*zero(T2), zero(T1)*zero(T2), x[1]*y[2] - x[2]*y[1]))
@inline cross(x::Vec{3}, y::Vec{3}) =
    @inbounds Vec{3}((x[2]*y[3] - x[3]*y[2], x[3]*y[1] - x[1]*y[3], x[1]*y[2] - x[2]*y[1]))

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
    S = Size(x)
    exps = contraction_exprs(Size(x), Size(y), Val(1))
    quote
        @_inline_meta
        @inbounds SymmetricSecondOrderTensor{dim}($(exps[indices(S)]...))
    end
end

# rotate
"""
    rotmat(θ::Real)

Construct 2D rotation matrix.

# Examples
```jldoctest
julia> rotmat(deg2rad(30))
2×2 Tensor{Tuple{2,2},Float64,2,4}:
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

# Examples
```jldoctest
julia> a = rand(Vec{3}); a /= norm(a);

julia> b = rand(Vec{3}); b /= norm(b);

julia> R = rotmat(a => b);

julia> R ⋅ a ≈ b
true
```
"""
function rotmat(pair::Pair{Vec{dim, T}, Vec{dim, T}})::Mat{dim, dim, T} where {dim, T}
    # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d/2672702#2672702
    a = pair.first
    b = pair.second
    dot(a, a) ≈ dot(b, b) || throw(ArgumentError("the length of two vectors must be the same"))
    a ==  b && return  one(Mat{dim, dim, T})
    a == -b && return -one(Mat{dim, dim, T})
    c = a + b
    2 * (c ⊗ c) / (c ⋅ c) - one(Mat{dim, dim, T})
end

# eigvals/eigen (just call methos in StaticArrays.jl)
@inline function eigvals(x::AbstractSymmetricSecondOrderTensor; permute::Bool = true, scale::Bool = true)
    Tensor(eigvals(Symmetric(convert_to_SArray(x)); permute, scale))
end
@inline function eigen(x::AbstractSymmetricSecondOrderTensor)
    eig = eigen(Symmetric(convert_to_SArray(x)))
    Eigen(Tensor(eig.values), Tensor(eig.vectors))
end
