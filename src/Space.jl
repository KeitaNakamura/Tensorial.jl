const SubSpace = Union{Int, Symmetry}

space_parameter_error() = throw(ArgumentError("Space parameters must be a tuple of `Int`s or `Symmetry`s."))
check_space_parameters(::Tuple{Vararg{SubSpace}}) = nothing
check_space_parameters(::Any) = space_parameter_error()

struct Space{S}
    function Space{S}() where {S}
        check_space_parameters(S)
        new{S}()
    end
end

Space() = Space{()}()
Space(subspaces::Vararg{SubSpace}) = Space{subspaces}()
Space(subspaces::Tuple{Vararg{SubSpace}}) = Space{subspaces}()
Space(dims::Tuple) = space_parameter_error()

function _size_parameter_error(::Type{S}) where {S}
    message = "Tensor's size parameter must be a tuple of `Int`s or `Symmetry` (e.g. `Tensor{Tuple{3,3}}` or `Tensor{Tuple{3, @Symmetry{3,3}}}`)."
    S <: Tuple || return :(throw(ArgumentError($message)))
    for s in S.parameters
        s isa Int && continue
        if s isa Type && s <: Symmetry
            err = _symmetry_size_parameter_error(only(s.parameters))
            err === nothing && continue
            return err
        end
        return :(throw(ArgumentError($message)))
    end
    nothing
end

function _symmetry_size_parameter_error(::Type{S}) where {S}
    symmetry_message = "Symmetry parameter must be a tuple of Ints (e.g., `Symmetry{Tuple{3,3}}` or `@Symmetry{3,3}`)."
    S <: Tuple || return :(throw(ArgumentError($symmetry_message)))
    if length(S.parameters) < 2
        return :(throw(ArgumentError("The number of symmetric indices must be ≥ 2, got $(length(S.parameters)).")))
    end
    all(x -> x isa Int, S.parameters) ||
        return :(throw(ArgumentError($symmetry_message)))
    length(unique(S.parameters)) == 1 ||
        return :(throw(ArgumentError("Dimensions must be all the same number, got $S.")))
    nothing
end

@generated function check_size_parameters(::Type{S}) where {S}
    err = _size_parameter_error(S)
    err === nothing ? :(nothing) : err
end

_construct_subspace(x::Int) = x
_construct_subspace(::Type{Symmetry{S}}) where {S} = Symmetry{S}()
@generated function Space(::Type{S}) where {S <: Tuple}
    err = _size_parameter_error(S)
    err === nothing || return err
    subspaces = map(_construct_subspace, S.parameters)
    Space(subspaces...)
end

Base.Tuple(::Space{S}) where {S} = S

_ncomponents(x::Int) = x
_ncomponents(x::Symmetry) = ncomponents(x)
ncomponents(::Space{S}) where {S} = prod(_ncomponents, S)
ncomponents(::Space{()}) = 1

tensorsize(x::Int) = (x,)
tensorsize(::Space{S}) where {S} = flatten_tuple(map(tensorsize, S))

tensororder(::Int) = 0
tensororder(s::Space) = length(tensorsize(s))
tensoraxes(s::Space) = map(Base.OneTo, tensorsize(s))

# don't allow to use `size` and `ndims` because their names are confusing.
Base.size(s::Space) = throw(ArgumentError("use `tensorsize` to get size of a tensor instead of `size`"))
Base.ndims(s::Space) = throw(ArgumentError("use `tensororder` to get order of a tensor instead of `ndims`"))

space_order(x) = tensororder(x)
space_order(::Int) = 1

###############
# checkbounds #
###############

static_index_parameter_error() = throw(ArgumentError("StaticIndex parameter must be an `Int` or an `AbstractVector{Int}`."))
check_static_index(::Union{Int, AbstractVector{Int}}) = nothing
check_static_index(::Any) = static_index_parameter_error()

struct StaticIndex{index} # index is completely known in compile-time
    function StaticIndex{index}() where {index}
        check_static_index(index)
        new{index}()
    end
end
StaticIndex(index) = StaticIndex{index}()
_static_index_length(::Int) = 1
_static_index_length(index) = length(index)
Base.length(::Type{StaticIndex{index}}) where {index} = _static_index_length(index)
Base.length(index::StaticIndex) = length(typeof(index))

const SpaceIndex = Union{Int, StaticVector{<: Any, Int}, StaticIndex}
const SpaceIndexOrColon = Union{SpaceIndex, Colon}

struct Len{n}
    Len{n}() where {n} = new{n::Int}()
end
Len(n) = Len{n}()

_checkindex(::Type{Bool}, ::Len{n}, i) where {n} = checkindex(Bool, Base.OneTo(n), i)
@generated function _checkindex(::Type{Bool}, ::Len{n}, ::StaticIndex{index}) where {n, index}
    checkindex(Bool, Base.OneTo(n), index) ? :(true) : :(false)
end

# linear index
@inline function Base.checkbounds(::Type{Bool}, space::Space, I)
    _checkindex(Bool, Len(prod(tensorsize(space))), I)
end
# cartesian index
@generated function Base.checkbounds(::Type{Bool}, S::Space{spaces}, I...) where {spaces}
    S = Space(spaces)
    tensororder(S) == length(I) || return :(false)
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
dropfirst() = throw(ArgumentError("cannot drop a dimension from an empty space"))
dropfirst(x::SubSpace) = _dropfirst_subspace(x, 1)
dropfirst(x::Space) = dropfirst(x, Val(1))
droplast(x::Space) = droplast(x, Val(1))

function _drop_error(subspaces::Tuple, n, op::Symbol)
    if !(n isa Int)
        message = string(op, " expects an integer Val, got Val(", n, ").")
        return :(throw(ArgumentError($message)))
    end
    order = sum(space_order, subspaces; init=0)
    if !(0 ≤ n ≤ order)
        message = string("cannot ", op, " ", n, " dimensions from a space of order ", order, ".")
        return :(throw(ArgumentError($message)))
    end
    nothing
end

_subspace_with_order(dim::Int, order::Int) =
    order == 0 ? () :
    order == 1 ? (dim,) :
    (Symmetry{NTuple{order, dim}}(),)

function _dropfirst_subspace(x::Int, n::Int)
    n == 0 ? (x,) :
    n == 1 ? () :
    throw(ArgumentError("cannot drop $n dimensions from a dimension of order 1."))
end

function _dropfirst_subspace(s::Symmetry, n::Int)
    order = space_order(s)
    0 ≤ n ≤ order || throw(ArgumentError("cannot drop $n dimensions from a symmetry of order $order."))
    _subspace_with_order(s[1], order - n)
end

function _dropfirst_subspaces(subspaces::Tuple, n::Int)
    kept = Any[]
    remaining = n
    for subspace in subspaces
        order = space_order(subspace)
        if remaining == 0
            push!(kept, subspace)
        elseif remaining < order
            append!(kept, _dropfirst_subspace(subspace, remaining))
            remaining = 0
        else
            remaining -= order
        end
    end
    Tuple(kept)
end

_droplast_subspaces(subspaces::Tuple, n::Int) =
    reverse(_dropfirst_subspaces(reverse(subspaces), n))

@generated function dropfirst(::Space{S}, ::Val{N}) where {S, N}
    err = _drop_error(S, N, :dropfirst)
    err === nothing || return err
    Space(_dropfirst_subspaces(S, N))
end

@generated function droplast(::Space{S}, ::Val{N}) where {S, N}
    err = _drop_error(S, N, :droplast)
    err === nothing || return err
    Space(_droplast_subspaces(S, N))
end

# contractions
function _check_contract_spaces(x::Space, y::Space, n)
    n isa Int || throw(ArgumentError("contract expects an integer Val, got Val($n)."))
    if !(0 ≤ n ≤ tensororder(x) && 0 ≤ n ≤ tensororder(y))
        throw(DimensionMismatch("cannot contract $n dimensions between spaces of order $(tensororder(x)) and $(tensororder(y))."))
    end
    if tensorsize(x)[end-n+1:end] != tensorsize(y)[1:n]
        throw(DimensionMismatch("contracted dimensions must match"))
    end
    nothing
end

function contract(x::Space, y::Space, ::Val{N}) where {N}
    _check_contract_spaces(x, y, N)
    ⊗(droplast(x, Val(N)), dropfirst(y, Val(N)))
end
tensor(x::Space) = x
tensor(x::Space, y::Space) = Space(Tuple(x)..., Tuple(y)...)
tensor(x::Space, y::Space, z::Space...) = tensor(tensor(x, y), z...)
contract1(x::Space, y::Space) = contract(x, y, Val(1))
contract2(x::Space, y::Space) = contract(x, y, Val(2))

# promote_space
promote_space() = Space()
promote_space(x::Space) = x
@generated function promote_space(::Space{S1}, ::Space{S2}) where {S1, S2}
    tensorsize(Space(S1)) == tensorsize(Space(S2)) || return :(throw(DimensionMismatch("dimensions must match")))
    Space(_promote_subspaces(S1, S2))
end
promote_space(x::Space, y::Space, z::Space...) = promote_space(promote_space(x, y), z...)

## helper functions
_promote_subspaces(x::Tuple, y::Tuple) = _promote_subspaces(x, y, ())
_promote_subspaces(::Tuple{}, ::Tuple{}, promoted::Tuple) = promoted
function _promote_subspaces(x::Tuple{Vararg{SubSpace}}, y::Tuple{Vararg{SubSpace}}, promoted::Tuple)
    x1 = x[1]
    y1 = y[1]
    if x1 == y1
        _promote_subspaces(Base.tail(x), Base.tail(y), (promoted..., x1))
    else
        x1_len = space_order(x1)
        y1_len = space_order(y1)
        if x1_len < y1_len
            y_prefix = Tuple(droplast(Space(y1), Val(y1_len - x1_len)))
            common = only(_promote_subspaces((x1,), y_prefix))
            _promote_subspaces(Base.tail(x),
                               Tuple(dropfirst(Space(y), Val(x1_len))),
                               (promoted..., common))
        elseif x1_len > y1_len
            x_prefix = Tuple(droplast(Space(x1), Val(x1_len - y1_len)))
            common = only(_promote_subspaces(x_prefix, (y1,)))
            _promote_subspaces(Tuple(dropfirst(Space(x), Val(y1_len))),
                               Base.tail(y),
                               (promoted..., common))
        else
            throw(DimensionMismatch("spaces have incompatible subspace decomposition"))
        end
    end
end

# tensortype
_size_parameter(x::Int) = x
_size_parameter(x::Symmetry) = typeof(x)
@generated function tensortype(::Space{S}) where {S}
    params = map(_size_parameter, S)
    quote
        @_inline_meta
        Tensor{Tuple{$(params...)}}
    end
end

#############################
# Static getindex interface #
#############################

_toindex(size::Int, ::Colon) = StaticIndex(1:size)
_toindex(size::Int, index::Any) = index

@generated function Base.getindex(space::Space{spaces}, indices::SpaceIndexOrColon...) where {spaces}
    dims = tensorsize(Space(spaces))
    if length(indices) != length(dims)
        return quote
            @_inline_meta
            @boundscheck checkbounds(space, indices...)
            Base.throw_boundserror(space, indices)
        end
    end
    exps = [:(_toindex($(dims[i]), indices[$i])) for i in 1:length(indices)]
    quote
        @_inline_meta
        @boundscheck checkbounds(space, indices...)
        _getindex(space, $(exps...))
    end
end

@inline function Base.getindex(space::Space, index::Int)
    @boundscheck checkbounds(space, index)
    Space()
end

@generated function Base.getindex(space::Space, index::StaticIndex{static_index}) where {static_index}
    if static_index isa Int
        return quote
            @_inline_meta
            @boundscheck checkbounds(space, index)
            Space()
        end
    end
    quote
        @_inline_meta
        @boundscheck checkbounds(space, index)
        _getindex(space, index)
    end
end

@inline function Base.getindex(space::Space, index::Union{StaticVector{<: Any, Int}, Colon})
    @boundscheck checkbounds(space, index)
    _getindex(space, _toindex(prod(tensorsize(space)), index))
end

@generated function _getindex(::Space{spaces}, indices::SpaceIndex...) where {spaces}
    Space(_indexed_subspaces(spaces, indices))
end

function _indexed_subspaces(spaces::Tuple, indices::Tuple)
    subspaces = Any[]
    index_position = 0
    for subspace in spaces
        if subspace isa Symmetry
            group = Any[]
            for _ in 1:space_order(subspace)
                index_position += 1
                _index_removes_axis(indices, index_position) && continue

                index = indices[index_position]
                if !isempty(group) && sameindex(index, group[end])
                    push!(group, index)
                else
                    _append_index_group!(subspaces, group)
                    group = Any[index]
                end
            end
            _append_index_group!(subspaces, group)
        else
            index_position += 1
            _index_removes_axis(indices, index_position) && continue
            push!(subspaces, _index_type_length(indices[index_position]))
        end
    end
    Tuple(subspaces)
end

_index_removes_axis(indices::Tuple, pos::Int) =
    pos > length(indices) || isintindex(indices[pos])

function _append_index_group!(subspaces::Vector, group::Vector)
    isempty(group) && return subspaces
    if length(group) == 1
        push!(subspaces, _index_type_length(only(group)))
    else
        push!(subspaces, Symmetry(fill(_index_type_length(first(group)), length(group))...))
    end
    subspaces
end

_index_type_length(::Type{<:StaticVector{N, Int}}) where {N} = N
_index_type_length(::Type{StaticIndex{index}}) where {index} = _static_index_length(index)

# isintindex
isintindex(::Type{Int}) = true
isintindex(::Type{StaticIndex{index}}) where {index} = index isa Int
isintindex(::Any) = false
# sameindex
sameindex(::Any, ::Any) = false
_static_index_values(index::Int) = (index,)
_static_index_values(index) = Tuple(index)
sameindex(::Type{StaticIndex{a}}, ::Type{StaticIndex{b}}) where {a, b} =
    _static_index_values(a) == _static_index_values(b)
