@inline function eigvals(x::AbstractSymmetricSecondOrderTensor; permute::Bool=true, scale::Bool=true)
    Tensor(eigvals(Symmetric(SArray(x)); permute=permute, scale = scale))
end

@inline function eigen(x::AbstractSymmetricSecondOrderTensor; permute::Bool=true, scale::Bool=true)
    _eig(x; permute=permute, scale=scale)
end

function eigen(x::AbstractSymmetricSecondOrderTensor{2, T}; permute::Bool=true, scale::Bool=true) where {T}
    _iszero(x) = abs(x) < eps(typeof(x))
    a, c, b = Tuple(x)
    if _iszero(c)
        # check diagonal case
        # `eigen(Symmetric(SMatrix{2,2}([1.0 1.9932760451045367e-17; 1.9932760451045367e-17 1.0])))` fails in StaticArrays.jl
        return Eigen(Vec(a,b), one(Mat{2,2,T}))
    else
        return _eig(x; permute=permute, scale=scale)
    end
end

function eigen(x::AbstractSymmetricSecondOrderTensor{3, T}; permute::Bool=true, scale::Bool=true) where {T}
    _isapproxzero(x) = abs(x) < cbrt(eps(typeof(x))) # sqrt fails, so use `cbrt`
    _iszero(x) = abs(x) < eps(typeof(x))

    a, d, f, b, e, c = Tuple(x)

    # block diagonal
    if !_isapproxzero(a) && _iszero(d) && _iszero(f)
        vals, vecs = eigen(@Tensor(x[[2,3], [2,3]]); permute=permute, scale=scale)
        λ₁ = a
        λ₂ = vals[1]
        λ₃ = vals[2]
        v₁ = Vec{3,T}(1,0,0)
        v₂ = vcat(0, vecs[:,1])
        v₃ = vcat(0, vecs[:,2])
        return _eig33_construct((λ₁, λ₂, λ₃), (v₁, v₂, v₃), true)
    elseif !_isapproxzero(b) && _iszero(d) && _iszero(e)
        vals, vecs = eigen(@Tensor(x[[1,3], [1,3]]); permute=permute, scale=scale)
        λ₁ = vals[1]
        λ₂ = b
        λ₃ = vals[2]
        v₁ = Vec(vecs[1,1], 0, vecs[2,1])
        v₂ = Vec{3,T}(0,1,0)
        v₃ = Vec(vecs[1,2], 0, vecs[2,2])
        return _eig33_construct((λ₁, λ₂, λ₃), (v₁, v₂, v₃), true)
    elseif !_isapproxzero(c) && _iszero(f) && _iszero(e)
        vals, vecs = eigen(@Tensor(x[[1,2], [1,2]]); permute=permute, scale=scale)
        λ₁ = vals[1]
        λ₂ = vals[2]
        λ₃ = c
        v₁ = vcat(vecs[:,1], 0)
        v₂ = vcat(vecs[:,2], 0)
        v₃ = Vec{3,T}(0,0,1)
        return _eig33_construct((λ₁, λ₂, λ₃), (v₁, v₂, v₃), true)
    else
        # special implementation for 3x3 case (https://hal.science/hal-01501221/document)
        _isapproxzero(f) && return _eig(x; permute=permute, scale=scale)
        λ₁, λ₂, λ₃ = eigvals(x; permute=permute, scale=scale)
        if _isapproxzero(λ₁) || _isapproxzero(λ₂) || _isapproxzero(λ₃) ||
           _isapproxzero(λ₁-λ₂) || _isapproxzero(λ₁-λ₃) || _isapproxzero(λ₂-λ₃)
            return _eig(x; permute=permute, scale=scale)
        end
        v₁, v₂, v₃ = map((λ₁, λ₂, λ₃)) do λ
            m = (d*(c-λ) - e*f) / (f*(b-λ) - d*e)
            Vec((λ-c-e*m)/f, m, 1)
        end
        if !isfinite(v₁[2]) || !isfinite(v₂[2]) || !isfinite(v₃[2])
            return _eig(x; permute=permute, scale=scale)
        end
        return _eig33_construct((λ₁, λ₂, λ₃), (v₁, v₂, v₃), false)
    end
end

@inline function _eig33_construct((λ₁,λ₂,λ₃), (v₁,v₂,v₃), permute)
    function _sortperm3(v)
        local perm = SVector(1,2,3)
        # unrolled bubble-sort
        (v[perm[1]] > v[perm[2]]) && (perm = SVector(perm[2], perm[1], perm[3]))
        (v[perm[2]] > v[perm[3]]) && (perm = SVector(perm[1], perm[3], perm[2]))
        (v[perm[1]] > v[perm[2]]) && (perm = SVector(perm[2], perm[1], perm[3]))
        perm
    end
    values = Vec(λ₁, λ₂, λ₃)
    vectors = hcat(normalize(v₁), normalize(v₂), normalize(v₃))
    if permute
        perm = _sortperm3(values)
        return Eigen(values[perm], vectors[:,perm])
    else
        return Eigen(values, vectors)
    end
end

# fallback to StaticArrays.jl
@inline function _eig(x::AbstractSymmetricSecondOrderTensor; permute::Bool, scale::Bool)
    eig = eigen(Symmetric(SArray(x)); permute=permute, scale=scale)
    Eigen(Tensor(eig.values), Tensor(eig.vectors))
end
