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
    sorted = sort!(collect(I), rev = true) # `reverse` is for column-major order
    LinearIndices(size(s))[CartesianIndex{order}(sorted...)]
end

dropfirst(x::SymmetricIndices{2}) = (LinearIndices((x.dims[2],)),)
dropfirst(x::SymmetricIndices) = (SymmetricIndices(Base.tail(x.dims)...),)


struct TensorIndices{order, I <: Tuple{Vararg{Union{LinearIndices{1}, SymmetricIndices}}}} <: AbstractArray{Int, order}
    tup::I
end
TensorIndices{order}(tup) where {order} = TensorIndices{order, typeof(tup)}(tup)

_indices(x::Int) = LinearIndices((x,))
@pure _indices(::Type{Symmetry{S}}) where {S} = (check_symmetry_parameters(S); SymmetricIndices(S.parameters...))
@pure function TensorIndices(::Type{Size}) where {Size <: Tuple}
    check_size_parameters(Size)
    tup = map(_indices, tuple(Size.parameters...))
    order = length(flatten_tuple(map(size, tup)))
    TensorIndices{order}(tup)
end

Base.size(x::TensorIndices) = flatten_tuple(map(size, x.tup))

function Base.getindex(indices::TensorIndices{order}, I::Vararg{Int, order}) where {order}
    tup = indices.tup
    st = 1
    inds = Vector{Int}(undef, length(tup))
    for (i,x) in enumerate(tup)
        n = ndims(x)
        inds[i] = x[I[st:(st+=n)-1]...]
    end
    LinearIndices(length.(tup))[inds...]
end

function serial(inds::TensorIndices)
    dict = Dict{Int, Int}()
    map(inds) do i
        get!(dict, i, length(dict) + 1)
    end
end


# dropfirst/droplast
## helper functions
dropfirst() = error()
dropfirst(x::LinearIndices{1}, ys...) = ys
dropfirst(x::SymmetricIndices, ys...) = (dropfirst(x)..., ys...)
## dropfirst/droplast for TensorIndices
dropfirst(x::TensorIndices{0}) = error()
droplast(x::TensorIndices{0}) = error()
dropfirst(x::TensorIndices{order}) where {order} = TensorIndices{order-1}(dropfirst(x.tup...))
droplast(x::TensorIndices{order}) where {order} = TensorIndices{order-1}(reverse(dropfirst(reverse(x.tup)...)))
for op in (:dropfirst, :droplast)
    @eval begin
        $op(x::TensorIndices, ::Val{0}) = x
        $op(x::TensorIndices, ::Val{N}) where {N} = $op($op(x), Val(N-1))
    end
end

@pure otimes(x::TensorIndices{m}, y::TensorIndices{n}) where {m, n} = TensorIndices{m + n}((x.tup..., y.tup...))

@pure function contract(x::TensorIndices, y::TensorIndices, ::Val{N}) where {N}
    if !(0 ≤ N ≤ ndims(x) && 0 ≤ N ≤ ndims(y) && size(x)[end-N+1:end] === size(y)[1:N])
        throw(DimensionMismatch("dimensions must match"))
    end
    otimes(droplast(x, Val(N)), dropfirst(y, Val(N)))
end

@pure _size(x::LinearIndices{1}) = length(x)
@pure _size(x::SymmetricIndices) = Symmetry{Tuple{size(x)...}}
@pure function tensortype(x::TensorIndices)
    Tensor{Tuple{map(_size, x.tup)...}, T, ndims(x), length(unique(x))} where {T}
end
