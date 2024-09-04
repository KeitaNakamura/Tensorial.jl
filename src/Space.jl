struct Space{S}
    function Space{S}() where {S}
        new{S::Tuple{Vararg{Union{Int, Symmetry}}}}()
    end
end

@pure Space(dims::Vararg{Union{Int, Symmetry}}) = Space{dims}()
@pure Space(dims::Tuple) = Space{dims}()

_construct(x::Int) = x
_construct(::Type{Symmetry{S}}) where {S} = Symmetry{S}()
@generated function Space(::Type{S}) where {S <: Tuple}
    check_size_parameters(S)
    dims = map(_construct, S.parameters)
    Space(dims...)
end

_ncomponents(x::Int) = x
_ncomponents(x::Symmetry) = ncomponents(x)
@pure ncomponents(::Space{S}) where {S} = prod(_ncomponents, S)
@pure ncomponents(::Space{()}) = 1

Base.Tuple(::Space{S}) where {S} = S

# tensorsize
tensorsize(x::Int) = (x,)
@pure tensorsize(::Space{S}) where {S} = flatten_tuple(map(tensorsize, S))
# tensororder
@pure tensororder(::Int) = 1
@pure tensororder(s::Space) = length(tensorsize(s))
@pure tensoraxes(s::Space) = map(Base.OneTo, tensorsize(s))
# don't allow to use `size` and `ndims` because their names are confusing.
Base.size(s::Space) = throw(ArgumentError("use `tensorsize` to get size of a tensor instead of `size`"))
Base.ndims(s::Space) = throw(ArgumentError("use `tensororder` to get order of a tensor instead of `ndims`"))

###############
# checkbounds #
###############

struct StaticIndex{index} # index is completely known in compile-time
    StaticIndex{index}() where {index} = new{index::Union{Int, AbstractVector{Int}}}()
end
@pure StaticIndex(index) = StaticIndex{index}()
@pure Base.length(::Type{StaticIndex{index}}) where {index} = length(index)
@pure Base.length(index::StaticIndex) = length(typeof(index))

struct Len{n}
    Len{n}() where {n} = new{n::Int}()
end
@pure Len(n) = Len{n}()

_checkindex(::Type{Bool}, ::Len{n}, i) where {n} = checkindex(Bool, Base.OneTo(n), i)
@generated _checkindex(::Type{Bool}, ::Len{n}, ::StaticIndex{index}) where {n, index} = # static checkindex
    checkindex(Bool, Base.OneTo(n), index)
# linear index
@inline function Base.checkbounds(::Type{Bool}, space::Space, I)
    _checkindex(Bool, Len(prod(tensorsize(space))), I)
end
# cartesin index
@generated function Base.checkbounds(::Type{Bool}, S::Space{spaces}, I...) where {spaces}
    S = Space(spaces)
    tensororder(S) != length(I) && return :(throw(BoundsError($S, I)))
    dims = tensorsize(S)
    exps = [:(_checkindex(Bool, Len($(dims[i])), I[$i])) for i in 1:length(I)]
    quote
        @_inline_meta
        all(tuple($(exps...)))
    end
end
@inline function Base.checkbounds(space::Space, I...)
    checkbounds(Bool, space, I...) || Base.throw_boundserror(space, I)
    nothing
end

function Base.show(io::IO, ::Space{S}) where {S}
    print(io, "Space", S)
end

##############
# operations #
##############

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
@pure function contract(x::Space, y::Space, ::Val{N}) where {N}
    if !(0 ≤ N ≤ tensororder(x) && 0 ≤ N ≤ tensororder(y) && tensorsize(x)[end-N+1:end] === tensorsize(y)[1:N])
        throw(DimensionMismatch("dimensions must match"))
    end
    otimes(droplast(x, Val(N)), dropfirst(y, Val(N)))
end
@pure otimes(x::Space) = x
@pure otimes(x::Space, y::Space) = Space(Tuple(x)..., Tuple(y)...)
@pure otimes(x::Space, y::Space, z::Space...) = otimes(otimes(x, y), z...)
@pure dot(x::Space, y::Space) = contract(x, y, Val(1))
@pure contract2(x::Space, y::Space) = contract(x, y, Val(2))

# promote_space
promote_space(x::Space) = x
@pure function promote_space(x::Space{S1}, y::Space{S2}) where {S1, S2}
    tensorsize(Space(S1)) == tensorsize(Space(S2)) || throw(DimensionMismatch("dimensions must match"))
    Space(_promote_space(S1, S2, ()))
end
@pure promote_space(x::Space, y::Space, z::Space...) = promote_space(promote_space(x, y), z...)
## helper functions
_promote_space(x::Tuple{}, y::Tuple{}, promoted::Tuple) = promoted
@pure function _promote_space(x::Tuple{Vararg{Union{Int, Symmetry}}}, y::Tuple{Vararg{Union{Int, Symmetry}}}, promoted::Tuple)
    x1 = x[1]
    y1 = y[1]
    if x1 == y1
        # just use `x1`
        _promote_space(Base.tail(x), Base.tail(y), (promoted..., x1))
    else
        x1_len = tensororder(x1)
        y1_len = tensororder(y1)
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

#############################
# Static getindex interface #
#############################

@pure _toindex(size::Int, ::Colon) = StaticIndex(1:size)
@pure _toindex(size::Int, index::Any) = index

@generated function Base.getindex(space::Space{spaces}, indices::Union{Int, StaticVector{<: Any, Int}, StaticIndex, Colon}...) where {spaces}
    dims = tensorsize(Space(spaces))
    exps = [:(_toindex($(dims[i]), indices[$i])) for i in 1:length(indices)]
    quote
        @_inline_meta
        @boundscheck checkbounds(space, indices...)
        _getindex(space, $(exps...))
    end
end

@inline function Base.getindex(space::Space, index::Union{Int, StaticVector{<: Any, Int}, StaticIndex, Colon})
    @boundscheck checkbounds(space, index)
    _getindex(space, _toindex(prod(tensorsize(space)), index))
end

@generated function _getindex(::Space{spaces}, indices::Union{Int, StaticVector{<: Any, Int}, StaticIndex}...) where {spaces}
    isnospace(count) = count>length(indices) || isintindex(indices[count]) # integer index will not create space
    getlast(x::Any) = x
    getlast(space::Vector) = isempty(space) ? nothing : getlast(space[end])

    dims = tensorsize(Space(spaces))
    cnt = 0 # counter to walk `indices`
    new_spaces = []
    for space in spaces
        if space isa Symmetry
            tmp_spaces = []
            for i in 1:tensororder(space)
                isnospace(cnt+=1) && continue
                if sameindex(indices[cnt], getlast(tmp_spaces))
                    push!(tmp_spaces[end], indices[cnt]) # push! to existing space
                else
                    push!(tmp_spaces, Any[indices[cnt]]) # create newspace
                end
            end
            # replace index into space
            for i in 1:length(tmp_spaces)
                if length(tmp_spaces[i]) == 1 # single entry -> not symmetric space
                    tmp_spaces[i] = length(only(tmp_spaces[i]))
                else
                    # should keep Symmetry if index is completely the same
                    tmp_spaces[i] = Symmetry(length.(tmp_spaces[i])...)
                end
            end
            # register
            append!(new_spaces, tmp_spaces)
        else
            isnospace(cnt+=1) && continue
            push!(new_spaces, length(indices[cnt]))
        end
    end

    Space(new_spaces...)
end
# isintindex
isintindex(::Type{Int}) = true
isintindex(::Type{StaticIndex{index}}) where {index} = index isa Int
isintindex(::Any) = false
# sameindex
sameindex(::Any, ::Any) = false
sameindex(::Type{StaticIndex{a}}, ::Type{StaticIndex{b}}) where {a, b} = collect(a) == collect(b)
