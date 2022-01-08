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
    Transpose{T, <: AbstractVec{<: Any, T}},
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

@generated function getindex_expr(x::Type{<: AbstractTensor}, ex::Union{Symbol, Expr}, i...)
    if any(x -> x <: Union{Symbol, Expr}, i)
        quote
            :($ex[$(i...)])
        end
    else
        # static getindex
        quote
            inds = independent_indices(x)
            :(Tuple($ex)[$(inds[i...])])
        end
    end
end

@generated function Base.getindex(x::AbstractTensor, I::Int...)
    ex = :()
    stride = 1
    for i in 1:length(I)
        if i == 1
            ex = :(I[1])
        else
            ex = :($ex + $stride * (I[$i] - 1))
        end
        stride *= size(x, i)
    end
    quote
        @_inline_meta
        @boundscheck checkbounds(x, I...)
        @inbounds x[$ex] # call getindex(x::AbstractTensor, i::Int)
    end
end

function Base.getindex(x::AbstractTensor, inds::Union{Int, StaticArray{<: Any, Int}, Colon, Val}...)
    @_propagate_inbounds_meta
    _getindex(Space(x)[inds...], x, inds...)
end

_indexing(parent_size::Int, ::Type{Int}, ex) = [ex]
_indexing(parent_size::Int, ::Type{<: StaticVector{n, Int}}, ex) where {n} = [:($ex[$i]) for i in 1:n]
_indexing(parent_size::Int, ::Type{Colon}, ex) = [i for i in 1:parent_size]
_indexing(parent_size::Int, ::Type{Val{x}}, ex) where {x} = collect(x)
@generated function _getindex(::Space{S}, x::AbstractTensor, inds::Union{Int, StaticArray{<: Any, Int}, Colon, Val}...) where {S}
    newspace = Space(S)
    TT = tensortype(newspace)
    inds_dim = map(_indexing, tensorsize(Space(x)), inds, [:(inds[$i]) for i in 1:length(inds)]) # indices on each dimension
    inds_all = collect(Iterators.product(inds_dim...)) # product of indices to get all indices
    if prod(tensorsize(newspace)) == length(inds_all)
        exps = map(i -> getindex_expr(x, :x, inds_all[i]...), indices(newspace))
    else # this is for `resize` function
        exps = map(indices(newspace)) do i
            I = CartesianIndices(newspace)[i]
            if checkbounds(Bool, inds_all, I)
                getindex_expr(x, :x, inds_all[I]...)
            else
                zero(eltype(x))
            end
        end
    end
    quote
        @_inline_meta
        @inbounds $TT($(exps...))
    end
end

Base.getindex(x::AbstractTensor, ::Colon) = vec(x)

# to SArray
@generated function StaticArrays.SArray(x::AbstractTensor)
    exps = [getindex_expr(x, :x, i) for i in 1:length(x)]
    quote
        @_inline_meta
        @inbounds SArray{Tuple{$(size(x)...)}}($(exps...))
    end
end
function StaticArrays.SArray(x::Transpose{<: T, <: AbstractVec{dim, T}}) where {dim, T}
    SMatrix{1, dim}(Tuple(parent(x)))
end


# vcat
Base.vcat(a::AbstractVecOrMatLike) = a
Base.vcat(a::AbstractVecOrMatLike, b::AbstractVecOrMatLike) = Tensor(vcat(SArray(a), SArray(b)))
Base.vcat(a::AbstractVecOrMatLike, b::AbstractVecOrMatLike, c::AbstractVecOrMatLike...) = vcat(vcat(a, b), vcat(c...))
# hcat
Base.hcat(a::AbstractVec) = Tensor(hcat(SArray(a)))
Base.hcat(a::AbstractMatLike) = a
Base.hcat(a::AbstractVecOrMatLike, b::AbstractVecOrMatLike) = Tensor(hcat(SArray(a), SArray(b)))
Base.hcat(a::AbstractVecOrMatLike, b::AbstractVecOrMatLike, c::AbstractVecOrMatLike...) = hcat(hcat(a, b), hcat(c...))
for op in (:vcat, :hcat)
    _op = Symbol(:_, op)
    @eval @generated function $_op(xs::Union{AbstractVecOrMatLike, Number}...)
        exps = [xs[i] <: Number ? :(Vec(xs[$i])) : :(xs[$i]) for i in 1:length(xs)]
        quote
            $($op)($(exps...))
        end
    end
    for I in 0:10
        xs = map(i -> :($(Symbol(:x, i))::Number), 1:I)
        @eval Base.$op($(xs...), y::AbstractVecOrMatLike, zs::Union{AbstractVecOrMatLike, Number}...) = $_op($(xs...), y, zs...)
    end
end

if VERSION ≥ v"1.6"
    # reverse
    Base.reverse(x::AbstractTensor; dims = :) = Tensor(reverse(SArray(x); dims = dims))
else
    Base.reverse(x::AbstractTensor) = Tensor(reverse(SArray(x)))
end
