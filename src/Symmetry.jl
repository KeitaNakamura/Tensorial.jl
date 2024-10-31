struct Symmetry{S <: Tuple}
    function Symmetry{S}() where {S}
        check_symmetry_parameters(S)
        new{S}()
    end
end
@generated Symmetry(::Type{S}) where {S} = Symmetry{S}()
Symmetry(dims::NTuple{N, Int}) where {N} = Symmetry{Tuple{dims...}}()
Symmetry(dims::Vararg{Int, N}) where {N} = Symmetry{Tuple{dims...}}()

@generated function check_symmetry_parameters(::Type{S}) where {S <: Tuple}
    if length(S.parameters) < 2
        return :(throw(ArgumentError("The number of symmetric indices must be ≥ 2, got $(length(S.parameters)).")))
    end
    if !all(x -> isa(x, Int), S.parameters)
        return :(throw(ArgumentError("Symmetry parameter must be a tuple of Ints (e.g., `Symmetry{Tuple{3,3}}` or `@Symmetry{3,3}`).")))
    end
    if length(unique(S.parameters)) != 1
        return :(throw(ArgumentError("Dimensions must be all the same number, got $S.")))
    end
end

tensorsize(::Symmetry{S}) where {S} = tuple(S.parameters...)
tensororder(s::Symmetry) = length(tensorsize(s))
ncomponents(::Symmetry{NTuple{order, dim}}) where {order, dim} = binomial(dim + order - 1, order)

Base.getindex(s::Symmetry, i::Int) = tensorsize(s)[i]

macro Symmetry(ex::Expr)
    @assert ex.head == :braces
    esc(:($Symmetry{Tuple{$(ex.args...)}}))
end

function Base.show(io::IO, ::Symmetry{S}) where {S}
    print(io, "Symmetry", tuple(S.parameters...))
end
