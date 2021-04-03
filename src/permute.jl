permute_error() = throw(ArgumentError("invalid permutation of dimensions"))
@generated function check_permute_parameters(::Val{N}, ::Val{perm}) where {N, perm}
    perm isa Tuple{Vararg{Int}} || return :(permute_error())
    N == length(perm) || return :(permute_error())
    for p in perm
        checkbounds(Bool, 1:N, p) || return :(permute_error())
    end
end

# Space

@generated function _permutedims(::Space{S}, ::Val{perm}) where {S, perm}
    # numbering each dimension
    num = Int[]
    for (i, s) in enumerate(S)
        append!(num, fill(i, length(s)))
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
        s = S[space_num]
        if length(group) == 1
            # if Symmetry, extract first dimension
            # if Real, just return it
            s[1]
        else
            if s isa Symmetry
                Symmetry(fill(s[1], length(group))...)
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
