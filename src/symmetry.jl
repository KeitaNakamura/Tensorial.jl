struct Symmetry{Size <: Tuple}
    function Symmetry{Size}() where {Size}
        check_symmetry_parameters(Size)
        new{Size}()
    end
end
@pure Symmetry(::Type{S}) where {S} = Symmetry{S}()
@pure Symmetry(dims::NTuple{N, Int}) where {N} = Symmetry{Tuple{dims...}}()
@pure Symmetry(dims::Vararg{Int, N}) where {N} = Symmetry{Tuple{dims...}}()

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

ncomponents(x::Symmetry) = ncomponents(size_to_indices(typeof(x)))
