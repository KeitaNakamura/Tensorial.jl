abstract type AbstractTensor{S <: Tuple, T, N} <: AbstractArray{T, N} end

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
@generated function convert_to_SArray(x::AbstractTensor)
    S = Space(x)
    NewS = Space(tensorsize(S)) # remove Symmetry
    exps = [getindex_expr(x, :x, i) for i in indices(NewS)]
    quote
        @_inline_meta
        @inbounds SArray{Tuple{$(tensorsize(NewS)...)}}(tuple($(exps...)))
    end
end

# aliases (too long name?)
const AbstractSecondOrderTensor{dim, T} = AbstractTensor{NTuple{2, dim}, T, 2}
const AbstractFourthOrderTensor{dim, T} = AbstractTensor{NTuple{4, dim}, T, 4}
const AbstractSymmetricSecondOrderTensor{dim, T} = AbstractTensor{Tuple{@Symmetry{dim, dim}}, T, 2}
const AbstractSymmetricFourthOrderTensor{dim, T} = AbstractTensor{NTuple{2, @Symmetry{dim, dim}}, T, 4}
const AbstractVec{dim, T} = AbstractTensor{Tuple{dim}, T, 1}
const AbstractMat{m, n, T} = AbstractTensor{Tuple{m, n}, T, 2}

# special name (not exported)
const AbstractSquareTensor{dim, T} = Union{AbstractTensor{Tuple{dim, dim}, T, 2},
                                           AbstractTensor{Tuple{@Symmetry{dim, dim}}, T, 2}}
