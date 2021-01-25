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

ncomponents(x::SymmetricIndices{order}) where {order} = (dim = size(x, 1); binomial(dim + order - 1, order))
ncomponents(x::LinearIndices) = length(x)

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
    map(x -> x.second, sort(collect(dups), by = x->x[1]))
end

for IndicesType in (:independent_indices, :indices, :duplicates)
    @eval @generated function $IndicesType(::Size{S}) where {S}
        arr = $(Symbol(:_, IndicesType))(TensorIndices(S))
        quote
            SArray{Tuple{$(size(arr)...)}, Int}($arr)
        end
    end
end
