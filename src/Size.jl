struct Size{S}
    function Size{S}() where {S}
        new{S::Tuple{Vararg{Union{Int, Symmetry}}}}()
    end
end

@pure Size(dims::Vararg{Union{Int, Symmetry}}) = Size{dims}()
@pure Size(dims::Tuple) = Size{dims}()

_construct(x::Int) = x
@pure _construct(::Type{Symmetry{S}}) where {S} = Symmetry{S}()
@pure function Size(::Type{S}) where {S <: Tuple}
    check_size_parameters(S)
    dims = map(_construct, tuple(S.parameters...))
    Size(dims)
end

_ncomponents(x::Int) = x
_ncomponents(x::Symmetry) = ncomponents(x)
@pure ncomponents(::Size{S}) where {S} = prod(_ncomponents, S)

@pure Base.length(s::Size) = length(Dims(s))
Base.getindex(s::Size, i::Int) = Tuple(s)[i]

_dims(x::Int) = (x,)
_dims(x::Symmetry) = Dims(x)
@pure Base.Dims(::Size{S}) where {S} = flatten_tuple(map(_dims, S))

@pure Base.Tuple(::Size{S}) where {S} = S

function Base.show(io::IO, ::Size{S}) where {S}
    print(io, "Size", S)
end


# dropfirst/droplast
dropfirst() = error()
dropfirst(x::Int) = ()
@pure dropfirst(::Symmetry{NTuple{2, dim}}) where {dim} = (dim,)
@pure dropfirst(::Symmetry{NTuple{order, dim}}) where {order, dim} = (Symmetry{NTuple{order-1, dim}}(),)
_dropfirst(x, ys...) = (dropfirst(x)..., ys...)
@pure dropfirst(::Size{S}) where {S} = Size(_dropfirst(S...))
@pure droplast(::Size{S}) where {S} = Size(reverse(_dropfirst(reverse(S)...)))
for op in (:dropfirst, :droplast)
    @eval begin
        $op(x::Size, ::Val{0}) = x
        $op(x::Size, ::Val{N}) where {N} = $op($op(x), Val(N-1))
    end
end

# otimes/contraction
@pure otimes(x::Size, y::Size) = Size(Tuple(x)..., Tuple(y)...)
@pure function contraction(x::Size, y::Size, ::Val{N}) where {N}
    if !(0 ≤ N ≤ length(x) && 0 ≤ N ≤ length(y) && Dims(x)[end-N+1:end] === Dims(y)[1:N])
        throw(DimensionMismatch("dimensions must match"))
    end
    otimes(droplast(x, Val(N)), dropfirst(y, Val(N)))
end

# promote_size
promote_size(x::Size) = x
function promote_size(x::Size, y::Size)
    Dims(x) == Dims(y) || throw(DimensionMismatch("dimensions must match"))
    Size(_promote_size(Tuple(x), Tuple(y), ()))
end
promote_size(x::Size, y::Size, z::Size...) = promote_size(promote_size(x, y), z...)
## helper functions
_promote_size(x::Tuple{}, y::Tuple{}, promoted::Tuple) = promoted
function _promote_size(x::Tuple, y::Tuple, promoted::Tuple)
    if x[1] == y[1]
        _promote_size(Base.tail(x), Base.tail(y), (promoted..., x[1]))
    else
        _promote_size(_dropfirst(x...), _dropfirst(y...), (promoted..., Dims(x[1])[1]))
    end
end

# tensortype
_typeof(x::Int) = x
_typeof(x::Symmetry) = typeof(x)
@pure function tensortype(x::Size)
    Tensor{Tuple{map(_typeof, Tuple(x))...}, T, length(x), ncomponents(x)} where {T}
end

# LinearIndices/CartesianIndices
for IndicesType in (LinearIndices, CartesianIndices)
    @eval (::Type{$IndicesType})(x::Size) = $IndicesType(Dims(x))
end
