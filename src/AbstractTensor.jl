abstract type AbstractTensor{S <: Tuple, T, N} <: AbstractArray{T, N} end

# aliases (too long name?)
const AbstractSecondOrderTensor{dim, T} = AbstractTensor{NTuple{2, dim}, T, 2}
const AbstractFourthOrderTensor{dim, T} = AbstractTensor{NTuple{4, dim}, T, 4}
const AbstractSymmetricSecondOrderTensor{dim, T} = AbstractTensor{Tuple{@Symmetry{dim, dim}}, T, 2}
const AbstractSymmetricFourthOrderTensor{dim, T} = AbstractTensor{NTuple{2, @Symmetry{dim, dim}}, T, 4}
const AbstractVec{dim, T} = AbstractTensor{Tuple{dim}, T, 1}
const AbstractMat{m, n, T} = AbstractTensor{Tuple{m, n}, T, 2}

const AbstractVecOrMat{T} = Union{AbstractVec{<: Any, T}, AbstractMat{<: Any, <: Any, T}}
const AbstractMatLike{T} = Union{
    AbstractMat{<: Any, <: Any, T},
    Transpose{T, <: AbstractVecOrMat{T}},
}
const AbstractVecOrMatLike{T} = Union{AbstractVec{<: Any, T}, AbstractMatLike{T}}

# special name (not exported)
const AbstractSquareTensor{dim, T} = Union{AbstractTensor{Tuple{dim, dim}, T, 2},
                                           AbstractTensor{Tuple{@Symmetry{dim, dim}}, T, 2}}



# for AbstractArray interface
Base.IndexStyle(::Type{<: AbstractTensor}) = IndexLinear()

@pure Base.size(::Type{TT}) where {TT <: AbstractTensor} = tensorsize(Space(TT))
@inline function Base.size(TT::Type{<: AbstractTensor}, d::Int)
    S = size(TT)
    d > length(S) ? 1 : S[d]
end
@inline Base.size(x::AbstractTensor) = tensorsize(Space(x))

@pure Base.ndims(::Type{TT}) where {TT <: AbstractTensor} = length(size(TT))

@pure Base.axes(::Type{TT}) where {TT <: AbstractTensor} = map(Base.OneTo, size(TT))
function Base.axes(TT::Type{<: AbstractTensor}, d::Int)
    A = axes(TT)
    d > length(A) ? Base.OneTo(1) : A[d]
end

# indices
for Indices in (:CartesianIndices, :LinearIndices)
    @eval Base.$Indices(::Type{TT}) where {TT <: AbstractTensor} = $Indices(size(TT))
end
for func in (:independent_indices, :indices, :duplicates)
    @eval begin
        $func(::Type{TT}) where {S, TT <: AbstractTensor{S}} = $func(Space(S))
        $func(x::AbstractTensor) = $func(typeof(x))
    end
end

Space(::Type{TT}) where {S, TT <: AbstractTensor{S}} = Space(S)
Space(::AbstractTensor{S}) where {S} = Space(S)

# getindex_expr
function getindex_expr(x::Type{<: AbstractTensor}, ex::Union{Symbol, Expr}, i...)
    inds = independent_indices(x)
    :(Tuple($ex)[$(inds[i...])])
end

# to SArray
@generated function StaticArrays.SArray(x::AbstractTensor)
    NewS = Space(size(x)) # remove Symmetry
    exps = [getindex_expr(x, :x, i) for i in indices(NewS)]
    quote
        @_inline_meta
        @inbounds SArray{Tuple{$(tensorsize(NewS)...)}}(tuple($(exps...)))
    end
end
function StaticArrays.SArray(x::Transpose{<: T, <: AbstractVec{dim, T}}) where {dim, T}
    SMatrix{1, dim}(Tuple(parent(x)))
end


# vcat
## vector or matrix
Base.vcat(a::AbstractMatLike) = a
Base.vcat(a::AbstractVecOrMatLike, b::AbstractVecOrMatLike) = Tensor(vcat(SArray(a), SArray(b)))
Base.vcat(a::AbstractVecOrMatLike, b::AbstractVecOrMatLike, c::AbstractVecOrMatLike...) = vcat(vcat(a, b), vcat(c...))
## vector or real
Base.vcat(a::AbstractVec) = a
Base.vcat(a::AbstractVec, b::AbstractVec) = Vec(Tuple(a)..., Tuple(b)...)
Base.vcat(a::AbstractVec, b::AbstractVec, c::AbstractVec...) = vcat(vcat(a, b), vcat(c...))
@generated function _vcat(xs::Union{AbstractVec, Real}...)
    exps = [xs[i] <: AbstractVec ? :(xs[$i]) : :(Vec(xs[$i])) for i in 1:length(xs)]
    quote
        vcat($(exps...))
    end
end
Base.vcat(a::AbstractVec, b::Union{AbstractVec, Real}...) = _vcat(a, b...)
Base.vcat(a::Real, b::AbstractVec, c::Union{AbstractVec, Real}...) = _vcat(a, b, c...)
Base.vcat(a::Real, b::Real, c::AbstractVec, d::Union{AbstractVec, Real}...) = _vcat(a, b, c, d...)
Base.vcat(a::Real, b::Real, c::Real, d::AbstractVec, e::Union{AbstractVec, Real}...) = _vcat(a, b, c, d, e...)
Base.vcat(a::Real, b::Real, c::Real, d::Real, e::AbstractVec, f::Union{AbstractVec, Real}...) = _vcat(a, b, c, d, e, f...)
Base.vcat(a::Real, b::Real, c::Real, d::Real, e::Real, f::AbstractVec, g::Union{AbstractVec, Real}...) = _vcat(a, b, c, d, e, f, g...)

# hcat
Base.hcat(a::AbstractVec) = Tensor(hcat(SArray(a)))
Base.hcat(a::AbstractMatLike) = a
Base.hcat(a::AbstractVecOrMatLike, b::AbstractVecOrMatLike) = Tensor(hcat(SArray(a), SArray(b)))
Base.hcat(a::AbstractVecOrMatLike, b::AbstractVecOrMatLike, c::AbstractVecOrMatLike...) = hcat(hcat(a, b), hcat(c...))
