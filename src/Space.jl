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
@pure tensoraxes(s::Space) = map(Base.OneTo, tensorsize(s))
# don't allow to use `size` and `ndims` because their names are confusing.
Base.size(s::Space) = throw(ArgumentError("use `tensorsize` to get size of a tensor instead of `size`"))
Base.ndims(s::Space) = throw(ArgumentError("use `tensororder` to get order of a tensor instead of `ndims`"))

@inline Base.checkbounds(::Type{Bool}, space::Space, I...) = Base.checkbounds_indices(Bool, tensoraxes(space), I)
@inline Base.checkbounds(::Type{Bool}, space::Space, I) = checkindex(Bool, Base.OneTo(prod(tensorsize(space))), I)
@inline function Base.checkbounds(space::Space, I...)
    checkbounds(Bool, space, I...) || Base.throw_boundserror(space, I)
    nothing
end

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

# contractions
@pure function contraction(x::Space, y::Space, ::Val{N}) where {N}
    if !(0 ≤ N ≤ tensororder(x) && 0 ≤ N ≤ tensororder(y) && tensorsize(x)[end-N+1:end] === tensorsize(y)[1:N])
        throw(DimensionMismatch("dimensions must match"))
    end
    otimes(droplast(x, Val(N)), dropfirst(y, Val(N)))
end
@pure otimes(x::Space) = x
@pure otimes(x::Space, y::Space) = Space(Tuple(x)..., Tuple(y)...)
@pure otimes(x::Space, y::Space, z::Space...) = otimes(otimes(x, y), z...)
@pure dot(x::Space, y::Space) = contraction(x, y, Val(1))
@pure double_contraction(x::Space, y::Space) = contraction(x, y, Val(2))

# promote_space
promote_space(x::Space) = x
@pure function promote_space(x::Space{S1}, y::Space{S2}) where {S1, S2}
    tensorsize(Space(S1)) == tensorsize(Space(S2)) || throw(DimensionMismatch("dimensions must match"))
    Space(_promote_space(S1, S2, ()))
end
@pure promote_space(x::Space, y::Space, z::Space...) = promote_space(promote_space(x, y), z...)
## helper functions
@pure _promote_space(x::Tuple{}, y::Tuple{}, promoted::Tuple) = promoted
@pure function _promote_space(x::Tuple{Vararg{Union{Int, Symmetry}}}, y::Tuple{Vararg{Union{Int, Symmetry}}}, promoted::Tuple)
    x1 = x[1]
    y1 = y[1]
    if x1 == y1
        # just use `x1`
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
        elseif x1_len > y1_len
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


#############################
# Static getindex interface #
#############################

abstract type AbstractDynamicIndex end
struct DynamicIndex   <: AbstractDynamicIndex;           end
struct DynamicIndices <: AbstractDynamicIndex; len::Int; end
# Base.length(::DynamicIndex) = 1
Base.length(x::DynamicIndices) = x.len

static_dynamic_index(n::Int, ::Type{Int}) = DynamicIndex()
static_dynamic_index(n::Int, ::Type{Colon}) = 1:n
static_dynamic_index(n::Int, ::Type{<: StaticVector{len}}) where {len} = DynamicIndices(len)
static_dynamic_index(n::Int, ::Type{Val{x}}) where {x} = x

same_index(a::AbstractDynamicIndex, b::AbstractDynamicIndex) = false
same_index(a::AbstractDynamicIndex, b::Any) = false
same_index(a::Any, b::AbstractDynamicIndex) = false
same_index(a::Any, b::Any) = _collect(a) == _collect(b)
_collect(x::Any) = collect(x)
_collect(::Nothing) = nothing

_remove_val(x) = x
_remove_val(::Val{x}) where {x} = SVector{length(x), Int}(x...)
@generated function Base.getindex(space::Space{S}, inds::Union{Int, StaticVector{<: Any, Int}, Colon, Val}...) where {S}
    dims = tensorsize(Space(S))
    indices = map(static_dynamic_index, dims, inds)
    quote
        @boundscheck checkbounds(space, map(_remove_val, inds)...)
        _getindex(space, Val(tuple($(indices...))))
    end
end

@generated function Base.getindex(space::Space{S}, index::Union{Int, StaticVector{<: Any, Int}, Colon, Val}) where {S}
    n = prod(tensorsize(Space(S)))
    index = static_dynamic_index(n, index)
    quote
        @boundscheck checkbounds(space, _remove_val(index))
        _getindex(space, Val(tuple($index)))
    end
end

@generated function _getindex(::Space{S}, ::Val{indices}) where {S, indices}
    # helper functions
    isskipped(count) = count > length(indices) || indices[count] isa Union{Int, DynamicIndex}
    newspace(x) = Any[x]
    space_lastentry(x::Any) = x
    space_lastentry(space::Vector) = isempty(space) ? nothing : space_lastentry(space[end])
    spacesize(x) = length(x)

    count = 0
    new_spaces = []
    for space in S
        if space isa Symmetry
            spaces = []
            for i in 1:length(space)
                isskipped(count += 1) && continue
                if same_index(indices[count], space_lastentry(spaces))
                    push!(spaces[end], indices[count])
                else
                    push!(spaces, newspace(indices[count]))
                end
            end
            # symmetry wrap
            for i in 1:length(spaces)
                if length(spaces[i]) == 1
                    spaces[i] = spacesize(only(spaces[i]))
                else
                    spaces[i] = Symmetry(spacesize.(spaces[i])...)
                end
            end
            append!(new_spaces, spaces)
        else
            isskipped(count += 1) && continue
            push!(new_spaces, length(indices[count]))
        end
    end
    quote
        Base.@_pure_meta
        Space($(new_spaces...))
    end
end
