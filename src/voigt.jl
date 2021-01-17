const VOIGT_ORDER = ([1], [1 3; 4 2], [1 6 5; 9 2 4; 8 7 3])

@generated function tovoigt(x::SecondOrderTensor{dim, T}) where {dim, T}
    L = ncomponents(x)
    exps = Vector{Expr}(undef, L)
    for i in 1:dim, j in 1:dim
        exps[VOIGT_ORDER[dim][i,j]] = getindex_expr(:x, x, i, j)
    end
    return quote
        @_inline_meta
        @inbounds SVector{$L, T}(tuple($(exps...)))
    end
end
@generated function tovoigt(x::FourthOrderTensor{dim, T}) where {dim, T}
    L = ncomponents(x)
    L2 = Int(√L)
    exps = Matrix{Expr}(undef, L2, L2)
    for i in 1:dim, j in 1:dim, k in 1:dim, l in 1:dim
        exps[VOIGT_ORDER[dim][i,j], VOIGT_ORDER[dim][k,l]] = getindex_expr(:x, x, i, j, k, l)
    end
    return quote
        @_inline_meta
        @inbounds SMatrix{$L2, $L2, T}(tuple($(exps...)))
    end
end

@generated function tovoigt(x::SymmetricSecondOrderTensor{dim, T}; offdiagscale::T = one(T)) where {dim, T}
    L = ncomponents(x)
    exps = Vector{Expr}(undef, L)
    for i in 1:dim, j in i:dim
        ex = getindex_expr(:x, x, i, j)
        exps[VOIGT_ORDER[dim][i,j]] = i == j ? ex : :(offdiagscale * $ex)
    end
    return quote
        @_inline_meta
        @inbounds SVector{$L, T}(tuple($(exps...)))
    end
end
@generated function tovoigt(x::SymmetricFourthOrderTensor{dim, T}; offdiagscale::T = one(T)) where {dim, T}
    L = ncomponents(x)
    L2 = Int(√L)
    exps = Matrix{Expr}(undef, L2, L2)
    for i in 1:dim, j in i:dim, k in 1:dim, l in k:dim
        ex = getindex_expr(:x, x, i, j, k, l)
        exps[VOIGT_ORDER[dim][i,j], VOIGT_ORDER[dim][k,l]] =
            (i == j && k == l) ? ex :
            (i == j || k == l) ? :($ex * offdiagscale) :
                                 :($ex * (offdiagscale * offdiagscale))
    end
    return quote
        @_inline_meta
        @inbounds SMatrix{$L2, $L2, T}(tuple($(exps...)))
    end
end

@inline function tomandel(x::Union{SymmetricSecondOrderTensor, SymmetricFourthOrderTensor})
    tovoigt(x, offdiagscale = eltype(x)(√2))
end

@inline function fromvoigt(TT::Type{<: SecondOrderTensor{dim}}, v::AbstractVector) where {dim}
    @_propagate_inbounds_meta
    TT(function (i, j); v[VOIGT_ORDER[dim][i, j]]; end)
end
@inline function fromvoigt(TT::Type{<: FourthOrderTensor{dim}}, v::AbstractMatrix) where {dim}
    @_propagate_inbounds_meta
    TT(function (i, j, k, l); v[VOIGT_ORDER[dim][i, j], VOIGT_ORDER[dim][k, l]]; end)
end
@inline function fromvoigt(TT::Type{<: SymmetricSecondOrderTensor{dim}}, v::AbstractVector{T}; offdiagscale::T = T(1)) where {dim, T}
    @_propagate_inbounds_meta
    TT(function (i, j)
           i > j && ((i, j) = (j, i))
           i == j ? (return v[VOIGT_ORDER[dim][i, j]]) :
                    (return v[VOIGT_ORDER[dim][i, j]] / offdiagscale)
       end)
end
@inline function fromvoigt(TT::Type{<: SymmetricFourthOrderTensor{dim}}, v::AbstractMatrix{T}; offdiagscale::T = T(1)) where {dim, T}
    @_propagate_inbounds_meta
    TT(function (i, j, k, l)
           i > j && ((i, j) = (j, i))
           k > l && ((k, l) = (l, k))
           i == j && k == l ? (return v[VOIGT_ORDER[dim][i, j], VOIGT_ORDER[dim][k, l]]) :
           i == j || k == l ? (return v[VOIGT_ORDER[dim][i, j], VOIGT_ORDER[dim][k, l]] / offdiagscale) :
                              (return v[VOIGT_ORDER[dim][i, j], VOIGT_ORDER[dim][k, l]] / (offdiagscale * offdiagscale))
       end)
end

@inline function frommandel(TT::Type{<: Union{SymmetricSecondOrderTensor, SymmetricFourthOrderTensor}}, v::AbstractArray{T}) where T
    @_propagate_inbounds_meta
    fromvoigt(TT, v, offdiagscale = T(√2))
end
