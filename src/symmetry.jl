struct Symmetry{Size <: Tuple}
    function Symmetry{Size}() where {Size}
        check_symmetry_parameters(Size)
        new{Size}()
    end
end
@pure Symmetry(::Type{S}) where {S} = Symmetry{S}()

@generated function check_symmetry_parameters(::Type{Size}) where {Size <: Tuple}
    if !all(x -> isa(x, Int), Size.parameters)
        return :(throw(ArgumentError("Symmetry parameter Size must be a tuple of Ints (e.g. `Symmetry{Tuple{3,3}}` or `@Symmetry{3,3}`).")))
    end
    if length(unique(Size.parameters)) != 1
        return :(throw(ArgumentError("Symmetry parameter Size must be unique, got $Size.")))
    end
end

macro Symmetry(ex::Expr)
    @assert ex.head == :braces
    esc(:($Symmetry{Tuple{$(ex.args...)}}))
end

ncomponents(::Symmetry{NTuple{order, dim}}) where {order, dim} = binomial(dim + order - 1, order)
