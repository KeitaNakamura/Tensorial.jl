numbering_components(x::Int) = LinearIndices((x,))

@generated function numbering_components(::Symmetry{S}) where {S}
    dims = tensorsize(Symmetry(S))
    vals = map(CartesianIndices(dims)) do I
        sorted = sort(SVector(Tuple(I)), rev=true) # `reverse` is for column-major order
        LinearIndices(dims)[sorted...]
    end
    quote
        SArray{Tuple{$(dims...)}, Int}(tuple($(vals...)))
    end
end

"""
    numbering_components(space)

Return a tensor-shaped table of component labels for `space`.
Components that are equivalent under symmetry are assigned the same label.

# Examples
```jldoctest
julia> Tensorial.numbering_components(Tensorial.Space(Symmetry(3,3)))
3×3 SArray{Tuple{3, 3}, Int, 2, 9} with indices SOneTo(3)×SOneTo(3):
 1  2  3
 2  5  6
 3  6  9

julia> Tensorial.numbering_components(Tensorial.Space(Symmetry(2,2), 3))
2×2×3 SArray{Tuple{2, 2, 3}, Int, 3, 12} with indices SOneTo(2)×SOneTo(2)×SOneTo(3):
[:, :, 1] =
 1  2
 2  4

[:, :, 2] =
 5  6
 6  8

[:, :, 3] =
 9  10
 10  12
 ```
 """
@generated function numbering_components(::Space{subspaces}) where {subspaces}
    space = Space(subspaces) # reconstruct space
    dims = tensorsize(space)

    # Numbering tables for each subspace
    subnumberings = map(numbering_components, subspaces)

    # Tensor order contributed by each subspace
    suborders = map(space_order, subspaces)

    # Linear indexing over the product space of all subspaces
    linear_indices = LinearIndices(map(prod ∘ tensorsize, subspaces))

    vals = map(CartesianIndices(dims)) do tensor_index
        # Current offset in the full tensor index
        offset = 1

        # Convert each subspace block of indices into its local linear index
        subindex = map(suborders, subnumberings) do order, numbering
            idx = numbering[ntuple(k -> tensor_index[offset + k - 1], order)...]
            offset += order
            idx
        end

        # Map the tuple of subspace indices to the global linear index
        linear_indices[subindex...]
    end

    quote
        SArray{Tuple{$(dims...)}, Int}(tuple($(vals...)))
    end
end

"""
    component_to_independent_map(space)

Return a tensor-shaped table mapping each tensor component to its corresponding independent component index.

# Examples
```jldoctest
julia> Tensorial.component_to_independent_map(Tensorial.Space(Symmetry(3,3)))
3×3 StaticArraysCore.SMatrix{3, 3, Int64, 9} with indices SOneTo(3)×SOneTo(3):
 1  2  3
 2  4  5
 3  5  6

julia> Tensorial.component_to_independent_map(Tensorial.Space(Symmetry(2,2), 3))
2×2×3 SArray{Tuple{2, 2, 3}, Int64, 3, 12} with indices SOneTo(2)×SOneTo(2)×SOneTo(3):
[:, :, 1] =
 1  2
 2  3

[:, :, 2] =
 4  5
 5  6

[:, :, 3] =
 7  8
 8  9
```
"""
@generated function component_to_independent_map(::Space{subspaces}) where {subspaces}
    space = Space(subspaces) # reconstruct space
    component_index_table = numbering_components(space)
    dims = tensorsize(space)

    renumber = Dict{Int, Int}()
    independent_index_table = map(component_index_table) do i
        get!(renumber, i, length(renumber) + 1)
    end

    quote
        SArray{Tuple{$(dims...)}, Int}(tuple($(independent_index_table...)))
    end
end

"""
    independent_to_component_map(space)

Return a vector mapping each independent component index to its tensor component index.

# Examples
```jldoctest
julia> Tensorial.independent_to_component_map(Tensorial.Space(Symmetry(3,3)))
6-element StaticArraysCore.SVector{6, Int64} with indices SOneTo(6):
 1
 2
 3
 5
 6
 9
```
"""
@generated function independent_to_component_map(::Space{subspaces}) where {subspaces}
    space = Space(subspaces) # reconstruct space
    component_index_table = numbering_components(space)
    component_indices = unique(component_index_table)

    quote
        SVector{$(length(component_indices)), Int}(tuple($(component_indices...)))
    end
end

"""
    independent_component_multiplicities(space)

Return a vector whose `i`-th entry is the number of tensor components
represented by the `i`-th independent component.

# Examples
```jldoctest
julia> Tensorial.independent_component_multiplicities(Tensorial.Space(Symmetry(3,3)))
6-element StaticArraysCore.SVector{6, Int64} with indices SOneTo(6):
 1
 2
 2
 1
 2
 1
```
"""
@generated function independent_component_multiplicities(::Space{subspaces}) where {subspaces}
    space = Space(subspaces) # reconstruct space
    component_index_table = numbering_components(space)

    counts = Dict{Int, Int}()
    for i in component_index_table
        counts[i] = get(counts, i, 0) + 1
    end

    multiplicities = map(last, sort(collect(counts), by=first))

    quote
        SVector{$(length(multiplicities)), Int}(tuple($(multiplicities...)))
    end
end
