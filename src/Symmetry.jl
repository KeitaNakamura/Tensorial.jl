struct Symmetry{S <: Tuple}
    function Symmetry{S}() where {S}
        check_symmetry_parameters(S)
        new{S}()
    end
end
@pure Symmetry(::Type{S}) where {S} = Symmetry{S}()
@pure Symmetry(dims::NTuple{N, Int}) where {N} = Symmetry{Tuple{dims...}}()
@pure Symmetry(dims::Vararg{Int, N}) where {N} = Symmetry{Tuple{dims...}}()

@generated function check_symmetry_parameters(::Type{S}) where {S <: Tuple}
    if !all(x -> isa(x, Int), S.parameters)
        return :(throw(ArgumentError("Symmetry parameter must be a tuple of Ints (e.g., `Symmetry{Tuple{3,3}}` or `@Symmetry{3,3}`).")))
    end
    if length(unique(S.parameters)) != 1
        return :(throw(ArgumentError("Dimensions must be all the same number, got $S.")))
    end
end

@pure Base.Dims(::Symmetry{S}) where {S} = tuple(S.parameters...)
@pure ncomponents(::Symmetry{NTuple{order, dim}}) where {order, dim} = binomial(dim + order - 1, order)

@pure Base.length(s::Symmetry) = length(Dims(s))
Base.getindex(s::Symmetry, i::Int) = Dims(s)[i]

macro Symmetry(ex::Expr)
    @assert ex.head == :braces
    esc(:($Symmetry{Tuple{$(ex.args...)}}))
end

function Base.show(io::IO, ::Symmetry{S}) where {S}
    print(io, "Symmetry", tuple(S.parameters...))
end


struct Skew{S <: Tuple}
    function Skew{S}() where {S}
        check_skew_parameters(S)
        new{S}()
    end
end
@pure Skew(::Type{S}) where {S} = Skew{S}()
@pure Skew(dims::NTuple{N, Int}) where {N} = Skew{Tuple{dims...}}()
@pure Skew(dims::Vararg{Int, N}) where {N} = Skew{Tuple{dims...}}()

@generated function check_skew_parameters(::Type{S}) where {S <: Tuple}
    if !all(x -> isa(x, Int), S.parameters)
        return :(throw(ArgumentError("Skew parameter must be a tuple of Ints (e.g., `Skew{Tuple{3,3}}` or `@Skew{3,3}`).")))
    end
    if length(unique(S.parameters)) != 1
        return :(throw(ArgumentError("Dimensions must be all the same number, got $S.")))
    end
end

@pure Base.Dims(::Skew{S}) where {S} = tuple(S.parameters...)
@pure ncomponents(::Skew{NTuple{order, dim}}) where {order, dim} = binomial(dim, order)

@pure Base.length(s::Skew) = length(Dims(s))
Base.getindex(s::Skew, i::Int) = Dims(s)[i]

macro Skew(ex::Expr)
    @assert ex.head == :braces
    esc(:($Skew{Tuple{$(ex.args...)}}))
end

function Base.show(io::IO, ::Skew{S}) where {S}
    print(io, "Skew", tuple(S.parameters...))
end
