struct Space{S}
    function Space{S}() where {S}
        new{S::Tuple{Vararg{Union{Int, Symmetry}}}}()
    end
end

@pure Space(dims::Vararg{Union{Int, Symmetry}}) = Space{dims}()
@pure Space(dims::Tuple) = Space{dims}()

_construct(x::Int) = x
@pure _construct(::Type{Symmetry{S}}) where {S} = Symmetry{S}()
@pure function Space(::Type{S}) where {S <: Tuple}
    check_size_parameters(S)
    dims = map(_construct, tuple(S.parameters...))
    Space(dims)
end

_ncomponents(x::Int) = x
_ncomponents(x::Symmetry) = ncomponents(x)
@pure ncomponents(::Space{S}) where {S} = prod(_ncomponents, S)

@pure Base.Dims(::Space{S}) where {S} = flatten_tuple(map(Dims, S))
@pure Base.Tuple(::Space{S}) where {S} = S

@pure Base.length(s::Space) = length(Dims(s))
Base.getindex(s::Space, i::Int) = Tuple(s)[i]

function Base.show(io::IO, ::Space{S}) where {S}
    print(io, "Space", S)
end


# dropfirst/droplast
dropfirst() = error()
dropfirst(x::Int) = ()
@pure dropfirst(::Symmetry{NTuple{2, dim}}) where {dim} = (dim,)
@pure dropfirst(::Symmetry{NTuple{order, dim}}) where {order, dim} = (Symmetry{NTuple{order-1, dim}}(),)
_dropfirst(x, ys...) = (dropfirst(x)..., ys...)
@pure dropfirst(::Space{S}) where {S} = Space(_dropfirst(S...))
@pure droplast(::Space{S}) where {S} = Space(reverse(_dropfirst(reverse(S)...)))
for op in (:dropfirst, :droplast)
    @eval begin
        $op(x::Space, ::Val{0}) = x
        $op(x::Space, ::Val{N}) where {N} = $op($op(x), Val(N-1))
    end
end

# otimes/contraction
@pure otimes(x::Space, y::Space) = Space(Tuple(x)..., Tuple(y)...)
@pure function contraction(x::Space, y::Space, ::Val{N}) where {N}
    if !(0 ≤ N ≤ length(x) && 0 ≤ N ≤ length(y) && Dims(x)[end-N+1:end] === Dims(y)[1:N])
        throw(DimensionMismatch("dimensions must match"))
    end
    otimes(droplast(x, Val(N)), dropfirst(y, Val(N)))
end

# promote_space
promote_space(x::Space) = x
function promote_space(x::Space, y::Space)
    Dims(x) == Dims(y) || throw(DimensionMismatch("dimensions must match"))
    Space(_promote_space(Tuple(x), Tuple(y), ()))
end
promote_space(x::Space, y::Space, z::Space...) = promote_space(promote_space(x, y), z...)
## helper functions
_promote_space(x::Tuple{}, y::Tuple{}, promoted::Tuple) = promoted
function _promote_space(x::Tuple, y::Tuple, promoted::Tuple)
    if x[1] == y[1]
        _promote_space(Base.tail(x), Base.tail(y), (promoted..., x[1]))
    else
        _promote_space(_dropfirst(x...), _dropfirst(y...), (promoted..., Dims(x[1])[1]))
    end
end

# tensortype
_typeof(x::Int) = x
_typeof(x::Symmetry) = typeof(x)
@pure function tensortype(x::Space)
    Tensor{Tuple{map(_typeof, Tuple(x))...}, T, length(x), ncomponents(x)} where {T}
end

# LinearIndices/CartesianIndices
for IndicesType in (LinearIndices, CartesianIndices)
    @eval (::Type{$IndicesType})(x::Space) = $IndicesType(Dims(x))
end
