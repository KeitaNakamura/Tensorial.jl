abstract type AbstractTensor{S <: Tuple, T, N} <: AbstractArray{T, N} end

Base.size(::Type{TT}) where {S, TT <: AbstractTensor{S}} = Dims(Size(S))
Base.size(x::AbstractTensor) = size(typeof(x))

# indices
for func in (:independent_indices, :indices, :duplicates)
    @eval begin
        $func(::Type{TT}) where {S, TT <: AbstractTensor{S}} = $func(Size(S))
        $func(x::AbstractTensor) = $func(typeof(x))
    end
end

Size(::Type{TT}) where {S, TT <: AbstractTensor{S}} = Size(S)
Size(::AbstractTensor{S}) where {S} = Size(S)

# getindex_expr
function getindex_expr(ex::Union{Symbol, Expr}, x::Type{<: AbstractTensor}, i...)
    inds = independent_indices(x)
    :(Tuple($ex)[$(inds[i...])])
end

# convert
Base.convert(::Type{TT}, x::TT) where {TT <: AbstractTensor} = x

# to SArray
@generated function convert_to_SArray(x::AbstractTensor)
    S = Size(x)
    NewS = Size(Dims(S)) # remove Symmetry
    exps = [getindex_expr(:x, x, i) for i in indices(NewS)]
    quote
        @_inline_meta
        @inbounds SArray{Tuple{$(Dims(NewS)...)}}(tuple($(exps...)))
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