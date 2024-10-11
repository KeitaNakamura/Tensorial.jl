numbering_components(x::Int) = LinearIndices((x,))

@generated function numbering_components(::Symmetry{S}) where {S}
    dims = tensorsize(Symmetry(S))
    SArray{Tuple{dims...}}(map(CartesianIndices(dims)) do I
        sorted = sort!(collect(I.I), rev=true) # `reverse` is for column-major order
        LinearIndices(dims)[sorted...]
    end)
end

@generated function numbering_components(::Space{spaces}) where {spaces}
    S = Space(spaces)
    dims = tensorsize(S)
    spinds = map(numbering_components, spaces)
    sporders = map(space_order, spaces)
    LI = LinearIndices(map(prod, map(tensorsize, spaces)))
    SArray{Tuple{dims...}}(map(CartesianIndices(dims)) do I
        cnt = Ref(1)
        I′ = map(spinds, sporders) do inds, order
            start = cnt[]
            cnt[] = start + order
            i = I.I[start:cnt[]-1] # extract indices corresponding to each space
            inds[i...]::Int
        end
        LI[I′...]
    end)
end

@generated function indices_all(::Space{spaces}) where {spaces}
    # make `inds` sequence
    inds = numbering_components(Space(spaces))
    dict = Dict{Int, Int}()
    map(inds) do i
        get!(dict, i, length(dict)+1)
    end
end

@generated function indices_unique(::Space{spaces}) where {spaces}
    inds = unique(numbering_components(Space(spaces)))
    SVector{length(inds)}(inds)
end

@generated function indices_dup(::Space{spaces}) where {spaces}
    dups = Dict{Int, Int}()
    for i in numbering_components(Space(spaces))
        get!(dups, i, 0)
        dups[i] += haskey(dups, i)
    end
    inds = map(last, sort(collect(dups), by=first))
    SVector{length(inds)}(inds)
end
