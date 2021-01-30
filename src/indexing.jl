#################
# LinearIndices #
#################
toindices(x::Int) = LinearIndices((x,))
ncomponents(x::LinearIndices) = length(x)

####################
# SymmetricIndices #
####################
struct SymmetricIndices{order, dim} <: AbstractArray{Int, order}
    sym::Symmetry{NTuple{order, dim}}
end
SymmetricIndices(dims::Int...) = SymmetricIndices(Symmetry(dims))
SymmetricIndices(dims::Tuple{Vararg{Int}}) = SymmetricIndices(Symmetry(dims))

toindices(sym::Symmetry) = SymmetricIndices(sym)
ncomponents(x::SymmetricIndices) = ncomponents(x.sym)

Base.size(x::SymmetricIndices) = Dims(x.sym)
function Base.getindex(s::SymmetricIndices{order}, I::Vararg{Int, order}) where {order}
    @boundscheck checkbounds(s, I...)
    sorted = sort!(collect(I), rev = true) # `reverse` is for column-major order
    @inbounds LinearIndices(size(s))[CartesianIndex{order}(sorted...)]
end

struct TensorIndices{order, S <: Tuple{Vararg{Union{Int, Symmetry}}}} <: AbstractArray{Int, order}
    dims::NTuple{order, Int}
    size::S
end

TensorIndices(S::Tuple{Vararg{Union{Int, Symmetry}}}) = TensorIndices(flatten_tuple(map(Dims, S)), S)
TensorIndices(S::Vararg{Union{Int, Symmetry}}) = TensorIndices(S)

Base.size(x::TensorIndices) = x.dims

function Base.getindex(t::TensorIndices{order}, I::Vararg{Int, order}) where {order}
    @boundscheck checkbounds(t, I...)
    S = t.size
    st = 1
    inds = Vector{Int}(undef, length(S))
    @inbounds begin
        for (i, s) in enumerate(S)
            x = toindices(s)
            n = ndims(x)
            inds[i] = x[I[st:(st+=n)-1]...]
        end
        LinearIndices(prod.(Dims.(S)))[inds...]
    end
end

function _independent_indices(inds::TensorIndices)
    dict = Dict{Int, Int}()
    map(inds) do i
        get!(dict, i, length(dict) + 1)
    end
end

function _indices(inds::TensorIndices)
    unique(inds)
end

function _duplicates(inds::TensorIndices)
    dups = Dict{Int, Int}()
    for i in inds
        if !haskey(dups, i)
            dups[i] = 1
        else
            dups[i] += 1
        end
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
