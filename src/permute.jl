permute_error() = throw(ArgumentError("invalid permutation of dimensions"))
@generated function check_permute_parameters(::Val{N}, ::Val{perm}) where {N, perm}
    perm isa Tuple{Vararg{Int}} || return :(permute_error())
    N == length(perm) || return :(permute_error())
    for p in perm
        checkbounds(Bool, 1:N, p) || return :(permute_error())
    end
end

# Vec
Base.permutedims(x::Vec{dim}) where {dim} = Mat{1,dim}(Tuple(x))

# Space

@generated function _permutedims(::Space{spaces}, ::Val{perm}) where {spaces, perm}
    # numbering each dimension
    num = Int[]
    for (i, space) in enumerate(spaces)
        append!(num, fill(i, space_order(space)))
    end
    num = num[collect(perm)] # allow to use invalid permutation in `_permutedims`
    # permute!(num, collect(perm))

    # collect the same number of group
    groups = [[num[1]]]
    for i in 2:length(num)
        if groups[end][end] == num[i] # check number
            push!(groups[end], num[i]) # add to the same group
        else
            push!(groups, [num[i]]) # create new group
        end
    end

    # apply `Symmetry` if needed
    newspace = map(groups) do group
        space_num = group[end]
        space = spaces[space_num]
        if length(group) == 1
            # if Symmetry, extract first dimension
            # if Number, just return it
            space[1]
        else
            if space isa Symmetry
                Symmetry(fill(space[1], length(group))...)
            else
                error() # unreachable
            end
        end
    end

    quote
        @_inline_meta
        Space($(newspace...))
    end
end

@inline function Base.permutedims(x::Space, perm::Val)
    check_permute_parameters(Val(tensororder(x)), perm)
    _permutedims(x, perm)
end

# AbstractTensor

@generated function ispermuted(::Val{N}, ::Val{perm}) where {N, perm}
    res = permute!(collect(1:N), collect(perm))
    quote
        @_inline_meta
        check_permute_parameters(Val(N), Val(perm))
        $(res == 1:N)
    end
end

@generated _revperm(::Val{perm}) where {perm} = :(@_inline_meta; tuple($(sortperm(collect(perm))...)))

@inline function Base.permutedims(x::AbstractTensor, perm::Val)
    ispermuted(Val(ndims(x)), perm) && return x
    S = permutedims(Space(x), perm)
    tensortype(S)() do ij...
        @_inline_meta
        @inbounds x[getindex.(Ref(ij), _revperm(perm))...]
    end
end
