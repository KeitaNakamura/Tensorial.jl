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

@pure Base.length(s::Space) = length(Tuple(s))
Base.getindex(s::Space, i::Int) = Tuple(s)[i]

@pure tensorsize(s::Space) = Dims(s)
@pure tensororder(s::Space) = length(tensorsize(s))
# don't allow to use `size` and `ndims` because their names are confusing.
Base.size(s::Space) = throw(ArgumentError("use `tensorsize` to get size of a tensor instead of `size`"))
Base.ndims(s::Space) = throw(ArgumentError("use `tensororder` to get order of a tensor instead of `ndims`"))

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
    if !(0 ≤ N ≤ tensororder(x) && 0 ≤ N ≤ tensororder(y) && tensorsize(x)[end-N+1:end] === tensorsize(y)[1:N])
        throw(DimensionMismatch("dimensions must match"))
    end
    otimes(droplast(x, Val(N)), dropfirst(y, Val(N)))
end

# promote_space
promote_space(x::Space) = x
@generated function promote_space(x::Space{S1}, y::Space{S2}) where {S1, S2}
    S = _promote_space(S1, S2, ())
    quote
        tensorsize(x) == tensorsize(y) || throw(DimensionMismatch("dimensions must match"))
        Space($S)
    end
end
promote_space(x::Space, y::Space, z::Space...) = promote_space(promote_space(x, y), z...)
## helper functions
_promote_space(x::Tuple{}, y::Tuple{}, promoted::Tuple) = promoted
function _promote_space(x::Tuple, y::Tuple, promoted::Tuple)
    x1 = x[1]
    y1 = y[1]
    if x1 == y1
        _promote_space(Base.tail(x), Base.tail(y), (promoted..., x1))
    else
        x1_len = length(x1)
        y1_len = length(y1)
        if x1_len < y1_len
            common = promote_space(Space(x1),
                                   droplast(Space(y1), Val(y1_len - x1_len))) |> Tuple
            _promote_space(Base.tail(x),
                           Tuple(dropfirst(Space(y), Val(x1_len))),
                           (promoted..., only(common)))
        elseif length(x1) > length(y1)
            common = promote_space(droplast(Space(x1), Val(x1_len - y1_len)),
                                   Space(y1)) |> Tuple
            _promote_space(Tuple(dropfirst(Space(x), Val(y1_len))),
                           Base.tail(y),
                           (promoted..., only(common)))
        else
            error()
        end
    end
end

# tensortype
_typeof(x::Int) = x
_typeof(x::Symmetry) = typeof(x)
@pure function tensortype(x::Space)
    Tensor{Tuple{map(_typeof, Tuple(x))...}, T, tensororder(x), ncomponents(x)} where {T}
end

# LinearIndices/CartesianIndices
for IndicesType in (LinearIndices, CartesianIndices)
    @eval (::Type{$IndicesType})(x::Space) = $IndicesType(tensorsize(x))
end
