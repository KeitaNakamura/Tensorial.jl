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
    sorted = tuple_sort(I, rev = true) # `reverse` is for column-major order
    @inbounds LinearIndices(size(s))[CartesianIndex{order}(sorted...)]
end

struct SkewIndices{order, dim} <: AbstractArray{Int, order}
    skew::Skew{NTuple{order, dim}}
end
SkewIndices(dims::Int...) = SkewIndices(Skew(dims))
SkewIndices(dims::Tuple{Vararg{Int}}) = SkewIndices(Skew(dims))

toindices(skew::Skew) = SkewIndices(skew)
ncomponents(x::SkewIndices) = ncomponents(x.skew)

Base.size(x::SkewIndices) = Dims(x.skew)

function sgn(x::Tuple{Vararg{Int, N}}) where N
    even = true
    @inbounds for i in 1:N, j in i+1:N
        x[i] == x[j] && return 0
        even ⊻= x[i] > x[j]
    end
    even ? 1 : -1
end

@inline function Base.getindex(s::SkewIndices{order}, I::Vararg{Int, order}) where {order}
    # A linear index of skew symmetry is set to satisfy following:
    #   (1) The linear index of stored value in tensor should be the smallest number
    #       in permutations of its cartesian index.
    #   (2) The cartesian index of the linear index should be always odd permutation.
    sorted = tuple_sort(I, rev = true) # `reverse` is for column-major order
    linear = LinearIndices(s)
    ϵ = sgn(sorted)
    ϵ == 0 && return 0
    if ϵ == -1 # sorted is odd permutation
        refindex = linear[CartesianIndex(sorted)]
    else # When sorted inds is even permutation, change it to odd permutation
        refindex = linear[sorted[2], sorted[1], sorted[3:order]...] # to satisfy (1)
    end
    -sgn(I) * refindex
end


struct TensorIndices{order, S <: Tuple{Vararg{Union{Int, Symmetry, Skew}}}} <: AbstractArray{Int, order}
    dims::NTuple{order, Int}
    size::S
end

TensorIndices(S::Tuple{Vararg{Union{Int, Symmetry, Skew}}}) = TensorIndices(flatten_tuple(map(Dims, S)), S)
TensorIndices(S::Vararg{Union{Int, Symmetry, Skew}}) = TensorIndices(S)

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
        sgn = sign(prod(inds))
        sgn == 0 ? 1 : sgn * LinearIndices(prod.(Dims.(S)))[abs.(inds)...]
    end
end

function _independent_indices(inds::TensorIndices)
    dict = Dict{Int, Int}()
    map(inds) do i
        sign(i) * get!(dict, abs(i), length(dict) + 1)
    end
end

function _indices(x::TensorIndices)
    unique(abs.(x))
end

function _duplicates(inds::TensorIndices)
    dups = Dict{Int, Int}()
    for i in inds
        i = abs(i)
        if !haskey(dups, i)
            dups[i] = 1
        else
            dups[i] += 1
        end
    end
    map(x -> x.second, sort(collect(dups), by = x->x[1]))
end

for func in (:independent_indices, :indices, :duplicates)
    @eval @generated function $func(::Space{S}) where {S}
        arr = $(Symbol(:_, func))(TensorIndices(S))
        quote
            SArray{Tuple{$(size(arr)...)}, Int}($arr)
        end
    end
end
