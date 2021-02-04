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
        return :(throw(ArgumentError("Symmetry parameter must be unique, got $S.")))
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
