############
# getindex #
############

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

@inline function Base.getindex(x::AbstractTensor, indices::Union{Int, StaticArray{<: Any, Int}, StaticIndex, Colon}...)
    @_propagate_inbounds_meta
    _getindex(Space(x)[indices...], x, indices...)
end

_indexing(parent_size::Int, ::Type{Int}, ex) = [ex]
_indexing(parent_size::Int, ::Type{<: StaticVector{n, Int}}, ex) where {n} = [:($ex[$i]) for i in 1:n]
_indexing(parent_size::Int, ::Type{Colon}, ex) = [i for i in 1:parent_size]
_indexing(parent_size::Int, ::Type{StaticIndex{x}}, ex) where {x} = collect(x)
@generated function _getindex(::Space{S}, x::AbstractTensor, indices::Union{Int, StaticArray{<: Any, Int}, StaticIndex, Colon}...) where {S}
    newspace = Space(S)
    TT = tensortype(newspace){eltype(x)}
    inds_dim = map(_indexing, tensorsize(Space(x)), indices, [:(indices[$i]) for i in 1:length(indices)]) # indices on each dimension
    inds_all = collect(Iterators.product(inds_dim...)) # product of indices to get all indices
    if prod(tensorsize(newspace)) == length(inds_all)
        exps = map(i -> getindex_expr(x, :x, inds_all[i]...), tensorindices_tuple(newspace))
    else # this is for `resize` function
        exps = map(tensorindices_tuple(newspace)) do i
            I = CartesianIndices(tensorsize(newspace))[i]
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

#################
# Concatenation #
#################

# vcat/hcat
Base.vcat(x::AbstractVecOrMatLike) = x
Base.hcat(x::AbstractMatLike) = x
for op in (:vcat, :hcat)
    _op = Symbol(:_, op)
    @eval @generated function $_op(xs::Union{AbstractVecOrMatLike, Number}...)
        exps = [xs[i] <: Number ? :(SVector(xs[$i])) : :(SArray(xs[$i])) for i in 1:length(xs)]
        quote
            @_inline_meta
            Tensor($($op)($(exps...)))
        end
    end
    for I in 0:10
        xs = map(i -> :($(Symbol(:x, i))::Number), 1:I)
        @eval @inline Base.$op(
            $(xs...),
            y::AbstractVecOrMatLike,
            zs::Union{AbstractVecOrMatLike, Number}...,
        ) = $_op($(xs...), y, zs...)
    end
end

# hvcat
# TODO: keep symmetries
@generated function _hvcat(::Val{rows}, a...) where {rows}
    offset = 0
    exps = map(rows) do ncolumns
        args = map(1:ncolumns) do i
            :(a[$offset + $i])
        end
        offset += ncolumns
        :(hcat($(args...)))
    end
    quote
        @_inline_meta
        vcat($(exps...))
    end
end
for I in 0:10
    xs = map(i -> :($(Symbol(:x, i))::Number), 1:I)
    @eval @inline Base.hvcat( # this inline is necessary for constant propagation
        rows::Dims,
        $(xs...),
        y::AbstractVecOrMatLike,
        zs::Union{AbstractVecOrMatLike, Number}...,
    )::Tensor = _hvcat(Val(rows), $(xs...), y, zs...)
end

########
# Misc #
########

Base.reverse(x::AbstractTensor; dims = :) = Tensor(reverse(SArray(x); dims = dims))

@generated function Base.vec(x::AbstractTensor)
    N = length(x)
    exps = [getindex_expr(x, :x, i) for i in 1:N]
    quote
        @_inline_meta
        @inbounds Vec($(exps...))
    end
end

@generated function resize(x::AbstractTensor, I::Val{dims}) where {dims}
    @assert dims isa Tuple{Vararg{Int}}
    inds = map(n->StaticIndex(1:n), dims)
    newspace = _getindex(Space(x), inds...)
    colons = [Colon() for _ in 1:ndims(x)]
    quote
        _getindex($newspace, x, $(colons...))
    end
end
@inline resize(x::AbstractTensor, i::Int, j::Int...) = resize(x, Val((i,j...)))
@inline resize(x::AbstractTensor, I::Tuple{Vararg{Int}}) = resize(x, Val(I))
