abstract type AbstractTensor{S <: Tuple, T, N} <: AbstractArray{T, N} end

# aliases (too long name?)
const AbstractSecondOrderTensor{dim, T} = AbstractTensor{NTuple{2, dim}, T, 2}
const AbstractFourthOrderTensor{dim, T} = AbstractTensor{NTuple{4, dim}, T, 4}
const AbstractSymmetricSecondOrderTensor{dim, T} = AbstractTensor{Tuple{@Symmetry({dim, dim})}, T, 2}
const AbstractSymmetricFourthOrderTensor{dim, T} = AbstractTensor{NTuple{2, @Symmetry({dim, dim})}, T, 4}
const AbstractVec{dim, T} = AbstractTensor{Tuple{dim}, T, 1}
const AbstractMat{m, n, T} = AbstractTensor{Tuple{m, n}, T, 2}

const AbstractMatLike{T} = Union{
    AbstractMat{<: Any, <: Any, T},
    AbstractSymmetricSecondOrderTensor{<: Any, T},
}
const AbstractVecOrMatLike{T} = Union{AbstractVec{<: Any, T}, AbstractMatLike{T}}

# special name (not exported)
const AbstractSquareTensor{dim, T} = Union{AbstractTensor{Tuple{dim, dim}, T, 2},
                                           AbstractTensor{Tuple{@Symmetry({dim, dim})}, T, 2}}



# for AbstractArray interface
Base.IndexStyle(::Type{<: AbstractTensor}) = IndexLinear()

@pure Base.size(::Type{TT}) where {TT <: AbstractTensor} = tensorsize(Space(TT))
@inline function Base.size(TT::Type{<: AbstractTensor}, d::Int)
    S = size(TT)
    d > length(S) ? 1 : S[d]
end
@inline Base.size(x::AbstractTensor) = tensorsize(Space(x))
@pure Base.length(::Type{TT}) where {TT <: AbstractTensor} = prod(size(TT))

@pure Base.ndims(::Type{TT}) where {TT <: AbstractTensor} = length(size(TT))

@pure Base.axes(::Type{TT}) where {TT <: AbstractTensor} = map(Base.OneTo, size(TT))
function Base.axes(TT::Type{<: AbstractTensor}, d::Int)
    A = axes(TT)
    d > length(A) ? Base.OneTo(1) : A[d]
end

# indices
Base.LinearIndices(::Type{TT}) where {TT <: AbstractTensor} = LinearIndices(size(TT))
Base.CartesianIndices(::Type{TT}) where {TT <: AbstractTensor} = CartesianIndices(size(TT))
for func in (:indices_all, :indices_unique, :indices_dup)
    @eval begin
        $func(::Type{TT}) where {S, TT <: AbstractTensor{S}} = $func(Space(S))
        $func(x::AbstractTensor) = $func(typeof(x))
    end
end

Space(::Type{TT}) where {S, TT <: AbstractTensor{S}} = Space(S)
Space(::AbstractTensor{S}) where {S} = Space(S)

ncomponents(x::AbstractTensor) = ncomponents(typeof(x))
@pure ncomponents(::Type{TT}) where {TT <: AbstractTensor} = ncomponents(Space(TT))

@generated function getindex_expr(x::Type{<: AbstractTensor}, ex::Union{Symbol, Expr}, i...)
    if any(x -> x <: Union{Symbol, Expr}, i)
        quote
            :($ex[$(i...)])
        end
    else
        # static getindex
        quote
            inds = indices_all(x)
            :(Tuple($ex)[$(inds[i...])])
        end
    end
end

# to SArray
@generated function StaticArrays.SArray(x::AbstractTensor)
    exps = [getindex_expr(x, :x, i) for i in 1:length(x)]
    quote
        @_inline_meta
        @inbounds SArray{Tuple{$(size(x)...)}}($(exps...))
    end
end
