struct SymmetricIndices{order} <: AbstractArray{Int, order}
    dims::NTuple{order, Int}
    function SymmetricIndices{order}(dims::NTuple{order, Int}) where {order}
        only(unique(dims))
        @assert order > 1
        new{order}(dims)
    end
end
SymmetricIndices(dims::Vararg{Int, order}) where {order} = SymmetricIndices{order}(dims)

Base.size(x::SymmetricIndices) = x.dims

function Base.getindex(s::SymmetricIndices{order}, I::Vararg{Int, order}) where {order}
    @boundscheck checkbounds(s, I...)
    sorted = sort!(collect(I), rev = true) # `reverse` is for column-major order
    @inbounds LinearIndices(size(s))[CartesianIndex{order}(sorted...)]
end

dropfirst(x::SymmetricIndices{2}) = (LinearIndices((x.dims[2],)),)
dropfirst(x::SymmetricIndices) = (SymmetricIndices(Base.tail(x.dims)...),)

@pure ncomponents(x::SymmetricIndices{order}) where {order} = (dim = size(x, 1); binomial(dim + order - 1, order))
@pure ncomponents(x::LinearIndices) = length(x)

size_to_indices(x::Int) = LinearIndices((x,))
@pure size_to_indices(::Type{Symmetry{S}}) where {S} = (check_symmetry_parameters(S); SymmetricIndices(S.parameters...))

struct TensorIndices{order, I <: Tuple{Vararg{Union{LinearIndices{1}, SymmetricIndices}}}} <: AbstractArray{Int, order}
    indices::I
    function TensorIndices{order, I}(indices::I) where {order, I}
        new{order::Int, I}(indices)
    end
end
@pure function TensorIndices(indices::Tuple{Vararg{Union{LinearIndices{1}, SymmetricIndices}}})
    order = length(flatten_tuple(map(size, indices)))
    TensorIndices{order, typeof(indices)}(indices)
end

@pure function TensorIndices(::Type{Size}) where {Size <: Tuple}
    check_size_parameters(Size)
    indices = map(size_to_indices, tuple(Size.parameters...))
    TensorIndices(indices)
end

Base.size(x::TensorIndices) = flatten_tuple(map(size, indices(x)))
indices(x::TensorIndices) = x.indices

ncomponents(x::TensorIndices) = prod(map(ncomponents, indices(x)))

function Base.getindex(x::TensorIndices{order}, I::Vararg{Int, order}) where {order}
    @boundscheck checkbounds(x, I...)
    st = 1
    inds = Vector{Int}(undef, length(indices(x)))
    @inbounds begin
        for (i,x) in enumerate(indices(x))
            n = ndims(x)
            inds[i] = x[I[st:(st+=n)-1]...]
        end
        LinearIndices(length.(indices(x)))[inds...]
    end
end

@pure function serialindices(inds::TensorIndices)
    dict = Dict{Int, Int}()
    arr = map(inds) do i
        get!(dict, i, length(dict) + 1)
    end
    SArray{Tuple{size(arr)...}, Int}(arr)
end

@pure function uniqueindices(inds::TensorIndices)
    arr = unique(inds)
    SArray{Tuple{size(arr)...}, Int}(arr)
end

@pure function dupsindices(inds::TensorIndices)
    dups = Dict{Int, Int}()
    for i in inds
        if !haskey(dups, i)
            dups[i] = 1
        else
            dups[i] += 1
        end
    end
    arr = map(x -> x.second, sort(collect(dups), by = x->x[1]))
    SArray{Tuple{size(arr)...}, Int}(arr)
end


# dropfirst/droplast
## helper functions
dropfirst() = error()
dropfirst(x::LinearIndices{1}, ys...) = ys
dropfirst(x::SymmetricIndices, ys...) = (dropfirst(x)..., ys...)
## dropfirst/droplast for TensorIndices
dropfirst(x::TensorIndices{0}) = error()
droplast(x::TensorIndices{0}) = error()
@pure dropfirst(x::TensorIndices) = TensorIndices(dropfirst(indices(x)...))
@pure droplast(x::TensorIndices) = TensorIndices(reverse(dropfirst(reverse(indices(x))...)))
for op in (:dropfirst, :droplast)
    @eval begin
        @pure $op(x::TensorIndices, ::Val{0}) = x
        @pure $op(x::TensorIndices, ::Val{N}) where {N} = $op($op(x), Val(N-1))
    end
end

# otimes/contract
@pure otimes(x::TensorIndices, y::TensorIndices) = TensorIndices((indices(x)..., indices(y)...))
@pure function contract(x::TensorIndices, y::TensorIndices, ::Val{N}) where {N}
    if !(0 ≤ N ≤ ndims(x) && 0 ≤ N ≤ ndims(y) && size(x)[end-N+1:end] === size(y)[1:N])
        throw(DimensionMismatch("dimensions must match"))
    end
    otimes(droplast(x, Val(N)), dropfirst(y, Val(N)))
end

# promote_indices
promote_indices(x::TensorIndices) = x
@pure function promote_indices(x::TensorIndices, y::TensorIndices)
    @assert size(x) == size(y)
    TensorIndices(_promote_indices(indices(x), indices(y), ()))
end
@pure promote_indices(x::TensorIndices, y::TensorIndices, z::TensorIndices...) = promote_indices(promote_indices(x, y), z...)
## helper functions
_promote_indices(x::Tuple{}, y::Tuple{}, promoted::Tuple) = promoted
@pure function _promote_indices(x::Tuple, y::Tuple, promoted::Tuple)
    if x[1] == y[1]
        _promote_indices(Base.tail(x), Base.tail(y), (promoted..., x[1]))
    else
        _promote_indices(dropfirst(x...), dropfirst(y...), (promoted..., LinearIndices((size(x[1], 1),))))
    end
end

@pure _size(x::LinearIndices{1}) = length(x)
@pure _size(x::SymmetricIndices) = Symmetry{Tuple{size(x)...}}
@pure function tensortype(x::TensorIndices)
    Tensor{Tuple{map(_size, indices(x))...}, T, ndims(x), length(unique(x))} where {T}
end
