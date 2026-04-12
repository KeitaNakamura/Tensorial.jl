"""
    DirectSumArray{Axes, T, N, L}

Container for an element of a direct-sum space stored in Mandel-flat coordinates.

# Type parameters
- `Axes`: block-space layout along each axis
- `T`: scalar storage type
- `N`: block-array dimension
- `L`: total number of flat stored entries

# Notes
- `size(A)` returns the block size.
- `flatsize(A)` returns the size of the flat Mandel storage.
"""
struct DirectSumArray{Axes <: Tuple{Vararg{Tuple}}, T, N, L}
    data::NTuple{L, T}
end

"""
    DirectSumVector{Axis1, T, L}

One-dimensional `DirectSumArray`.
"""
const DirectSumVector{Axis1, T, L} = DirectSumArray{Tuple{Axis1}, T, 1, L}

"""
    DirectSumMatrix{Axis1, Axis2, T, L}

Two-dimensional `DirectSumArray`.
"""
const DirectSumMatrix{Axis1, Axis2, T, L} = DirectSumArray{Tuple{Axis1, Axis2}, T, 2, L}

# constructors
@generated function DirectSumArray{Axes, T}(data::Tuple{Vararg{Number, L}}) where {Axes, T, L}
    dims = _flatsize(Axes)
    @assert prod(dims) == L
    quote
        DirectSumArray{Axes, T, $(length(dims)), L}(convert_ntuple(T, data))
    end
end
function DirectSumArray{Axes}(data::Tuple{Vararg{Number, L}}) where {Axes, L}
    T = promote_ntuple_eltype(data)
    DirectSumArray{Axes, T}(data)
end
DirectSumVector{Axis}(data::Tuple{Vararg{Number}}) where {Axis} = DirectSumArray{Tuple{Axis}}(data)
DirectSumMatrix{Axis1, Axis2}(data::Tuple{Vararg{Number}}) where {Axis1, Axis2} = DirectSumArray{Tuple{Axis1, Axis2}}(data)

@generated function ofaxes(::DirectSumArray{Axes}) where {Axes}
    axes = map(x -> Tuple{x.parameters...}, Axes.parameters)
    quote
        tuple($(axes...))
    end
end

Base.zero(::Type{<: DirectSumArray{Axes, T}}) where {Axes, T} = DirectSumArray{Axes, T}(Tuple(zero(Vec{prod(_flatsize(Axes)), T})))
Base.zero(A::DirectSumArray) = zero(typeof(A))

@generated function _size(::Type{Axes}) where {Axes}
    dims = map(axis -> length(axis.parameters), Axes.parameters)
    quote
        tuple($(dims...))
    end
end
Base.size(::Type{<: DirectSumArray{Axes}}) where {Axes} = _size(Axes)
Base.size(A::DirectSumArray) = size(typeof(A))

@generated function _flatsize(::Type{Axes}) where {Axes}
    dims = map(Axes.parameters) do Tup
        sum(ncomponents, map(Space, Tup.parameters))
    end
    quote
        tuple($(dims...))
    end
end
flatsize(::Type{<: DirectSumArray{Axes}}) where {Axes} = _flatsize(Axes)
flatsize(A::DirectSumArray) = flatsize(typeof(A))

# create DirectSumArray from blocks
@generated function _pack(::Type{Axes}, xs::Tuple{Vararg{Union{Number, AbstractTensor}}}) where {Axes}
    axes = Tuple(map(x -> x.parameters, Axes.parameters))

    dims = _flatsize(Axes)

    lens_per_axis = map(ax -> Tuple(map(a -> ncomponents(Space(a)), ax)), axes)

    ranges_per_axis = map(lens_per_axis) do lens
        stops = cumsum(lens)
        starts = stops .- lens .+ 1
        UnitRange.(starts, stops)
    end

    exprs = Array{Any}(undef, dims)

    for (blk, I) in enumerate(CartesianIndices(map(length, axes))) # each block index
        space = ⊗(map(Space, getindex.(axes, Tuple(I)))...)
        inds = CartesianIndices(getindex.(ranges_per_axis, Tuple(I)))

        if space === Space()
            exprs[only(inds)] = :(xs[$blk])
        else
            mults = independent_component_multiplicities(space)
            local_dims = size(inds)
            exprs[inds] = map(LinearIndices(local_dims)) do J
                mults[J] == 1 ? :(Tuple(xs[$blk])[$J]) : :(Tuple(xs[$blk])[$J] * $(sqrt(mults[J])))
            end
        end
    end

    quote
        DirectSumArray{Axes}(tuple($(exprs...)))
    end
end

"""
    pack(args...)

Construct a `DirectSumArray` from the blocks of a direct-sum space.

Each argument is interpreted as one block of the direct sum. Scalar blocks are
stored directly. Tensor blocks are flattened and concatenated into the internal
storage. If a block belongs to a symmetric tensor space, Mandel coordinates are
used internally.

`⊕` (typed by `\\oplus<tab>` ) is provided as an alias for `pack`.

See also: [`unpack`](@ref), [`flatview`](@ref)

# Examples
```jldoctest
julia> A = @Mat[1.0 2.0; 3.0 4.0]
2×2 Tensor{Tuple{2, 2}, Float64, 2, 4}:
 1.0  2.0
 3.0  4.0

julia> x = pack(A, 3.0)
2-element DirectSumVector with storage Float64:
 Space(2, 2)
 Space()

julia> As = symmetric(A)
2×2 SymmetricSecondOrderTensor{2, Float64, 3}:
 1.0  2.5
 2.5  4.0

julia> v = @Vec[4.0, 5.0]
2-element Vec{2, Float64}:
 4.0
 5.0

julia> y = As ⊕ 3.0 ⊕ v
3-element DirectSumVector with storage Float64:
 Space(Symmetry(2, 2),)
 Space()
 Space(2,)
```
"""
pack


"""
    unpack(A::DirectSumArray)

Return the blocks stored in a DirectSumArray.

`unpack(A)` returns all direct-sum blocks as a tuple.
`unpack(A, I...)` returns the block at block index `I`.
Tensor blocks are reconstructed from the internal flat storage.
For symmetric tensor blocks, the internally stored Mandel coordinates
are converted back to the corresponding tensor values.

At the level of block values, unpack is the inverse of pack.

!!! info
    `unpack(A)` is type-stable, but `unpack(A, I...)` may be type-unstable
    if the block index is not known at compile time, since different block indices
    may correspond to different return types. It is therefore most suitable when `I`
    is a constant and constant propagation can resolve the selected block.

See also: [`pack`](@ref), [`flatview`](@ref)

# Examples
```jldoctest
julia> A = symmetric(@Mat[1.0 2.0; 3.0 4.0])
2×2 SymmetricSecondOrderTensor{2, Float64, 3}:
 1.0  2.5
 2.5  4.0

julia> x = pack(A, 3.0)
2-element DirectSumVector with storage Float64:
 Space(Symmetry(2, 2),)
 Space()

julia> unpack(x)
([1.0 2.5; 2.5 4.0], 3.0)

julia> unpack(x, 1)
2×2 SymmetricSecondOrderTensor{2, Float64, 3}:
 1.0  2.5
 2.5  4.0

julia> unpack(x, 2)
3.0
```
"""
unpack

# pack: ⊕
_sizeinfo(::Type{<: Tensor{S}}) where {S} = S
_sizeinfo(::Type{T}) where {T <: Number} = Tuple{}
function pack(args::Union{Number, AbstractTensor}...)
    Axis = Tuple{map(_sizeinfo, map(typeof, args))...}
    _pack(Tuple{Axis}, args)
end

pack(A::DirectSumVector, a::Union{Number, AbstractTensor}) = pack(unpack(A)..., a)
pack(a::Union{Number, AbstractTensor}, A::DirectSumVector) = pack(a, unpack(A)...)
pack(A::DirectSumVector, B::DirectSumVector) = pack(unpack(A)..., unpack(B)...)

@generated function pack(A::DirectSumArray{Axes1, T1, N}, B::DirectSumArray{Axes2, T2, N}) where {Axes1, T1, Axes2, T2, N}
    NewAxes = Tuple{(Tuple{Axes1.parameters[i].parameters..., Axes2.parameters[i].parameters...} for i in 1:N)...}
    dimsA = _flatsize(Axes1)
    dimsB = _flatsize(Axes2)
    T = promote_type(T1, T2)

    exps = Expr[]
    for I in CartesianIndices(_flatsize(NewAxes))
        is_in_A = all(I[d] ≤ dimsA[d] for d in 1:N)
        is_in_B = all(dimsA[d] < I[d] && I[d] ≤ dimsA[d] + dimsB[d] for d in 1:N)

        if is_in_A
            ia = LinearIndices(dimsA)[I]
            push!(exps, :(convert($T, A.data[$ia])))
        elseif is_in_B
            ib = LinearIndices(dimsB)[I - CartesianIndex(dimsA)]
            push!(exps, :(convert($T, B.data[$ib])))
        else
            push!(exps, :(zero($T)))
        end
    end

    quote
        DirectSumArray{$NewAxes}(tuple($(exps...)))
    end
end

@generated function unpack(A::DirectSumArray{Axes}) where {Axes}
    axes = Tuple(map(x -> x.parameters, Axes.parameters))

    lens_per_axis = map(ax -> Tuple(map(a -> ncomponents(Space(a)), ax)), axes)

    ranges_per_axis = map(lens_per_axis) do lens
        stops = cumsum(lens)
        starts = stops .- lens .+ 1
        UnitRange.(starts, stops)
    end

    spaces = Any[]
    block_ranges = Any[]
    for I in CartesianIndices(map(length, axes)) # each block index
        push!(spaces, ⊗(map(Space, getindex.(axes, Tuple(I)))...))
        push!(block_ranges, getindex.(ranges_per_axis, Tuple(I)))
    end

    LI = LinearIndices(_flatsize(Axes))
    exps = Expr[]
    for (space, rngs) in zip(spaces, block_ranges)
        inds = LI[rngs...]
        if space === Space() # scalar
            push!(exps, :(A.data[$(only(inds))]))
        else
            TT = tensortype(space)
            mults = independent_component_multiplicities(space)
            data_expr = Expr(:tuple, map(enumerate(inds)) do (j, i)
                mults[j] == 1 ? :(A.data[$i]) :
                                :(A.data[$i] / $(sqrt(mults[j])))
            end...)
            push!(exps, :($TT($data_expr)))
        end
    end

    quote
        tuple($(exps...))
    end
end

function unpack(A::DirectSumArray, I::Tuple)
    i = LinearIndices(size(A))[I...]
    unpack(A)[i]
end
unpack(A::DirectSumArray, i::Int, j::Int...) = unpack(A, (i,j...))

# ops
Base.:+(A::DirectSumArray) = A
Base.:-(A::DirectSumArray{Axes}) where {Axes} = DirectSumArray{Axes}(-1 .* A.data)
Base.:+(A::DirectSumArray{Axes}, B::DirectSumArray{Axes}) where {Axes} = DirectSumArray{Axes}(A.data .+ B.data)
Base.:-(A::DirectSumArray{Axes}, B::DirectSumArray{Axes}) where {Axes} = DirectSumArray{Axes}(A.data .- B.data)
Base.:*(A::DirectSumArray{Axes}, a::Number) where {Axes} = DirectSumArray{Axes}(A.data .* a)
Base.:*(a::Number, A::DirectSumArray{Axes}) where {Axes} = A * a
Base.:/(A::DirectSumArray{Axes}, a::Number) where {Axes} = DirectSumArray{Axes}(A.data ./ a)

Base.:*(A::DirectSumMatrix{Axis1, Axis2}, x::DirectSumVector{Axis2}) where {Axis1, Axis2} = DirectSumVector{Axis1}(Tuple(flatview(A) * flatview(x)))
Base.:*(A::DirectSumMatrix{Axis1, Axis2}, B::DirectSumMatrix{Axis2, Axis3}) where {Axis1, Axis2, Axis3} = DirectSumMatrix{Axis1, Axis3}(Tuple(flatview(A) * flatview(B)))

@generated function contract(A1::DirectSumArray{Axes1}, A2::DirectSumArray{Axes2}, ::Val{N}) where {Axes1, Axes2, N}
    n1 = length(Axes1.parameters)
    n2 = length(Axes2.parameters)

    @assert 0 ≤ N ≤ min(n1, n2)

    contracted1 = Axes1.parameters[n1-N+1:n1]
    contracted2 = Axes2.parameters[1:N]

    @assert contracted1 == contracted2

    left  = Axes1.parameters[1:n1-N]
    right = Axes2.parameters[N+1:n2]
    NewAxes = Tuple{left..., right...}

    if length(left) + length(right) == 0
        return quote
            contract(Tensor(flatview(A1)), Tensor(flatview(A2)), Val(N))
        end
    else
        return quote
            data = Tuple(contract(Tensor(flatview(A1)), Tensor(flatview(A2)), Val(N)))
            DirectSumArray{$NewAxes}(data)
        end
    end
end
contract1(x1::DirectSumArray, x2::DirectSumArray) = contract(x1, x2, Val(1))
contract2(x1::DirectSumArray, x2::DirectSumArray) = contract(x1, x2, Val(2))
contract3(x1::DirectSumArray, x2::DirectSumArray) = contract(x1, x2, Val(3))
contract4(x1::DirectSumArray, x2::DirectSumArray) = contract(x1, x2, Val(4))
contract5(x1::DirectSumArray, x2::DirectSumArray) = contract(x1, x2, Val(5))
contract6(x1::DirectSumArray, x2::DirectSumArray) = contract(x1, x2, Val(6))
contract7(x1::DirectSumArray, x2::DirectSumArray) = contract(x1, x2, Val(7))
contract8(x1::DirectSumArray, x2::DirectSumArray) = contract(x1, x2, Val(8))
contract9(x1::DirectSumArray, x2::DirectSumArray) = contract(x1, x2, Val(9))

@generated function tensor(A::DirectSumArray{Axes1}, B::DirectSumArray{Axes2}) where {Axes1, Axes2}
    NewAxes = Tuple{Axes1.parameters..., Axes2.parameters...}
    quote
        data = Tuple(tensor(Vec(A.data), Vec(B.data)))
        DirectSumArray{$NewAxes}(data)
    end
end

LinearAlgebra.dot(A::DirectSumArray{Axes}, B::DirectSumArray{Axes}) where {Axes} = dot(A.data, B.data)
LinearAlgebra.norm(A::DirectSumArray) = sqrt(dot(A, A))
Base.inv(A::DirectSumMatrix{Axis1, Axis2}) where {Axis1, Axis2} = DirectSumArray{Tuple{Axis2, Axis1}}(Tuple(inv(flatview(A))))
Base.:\(A::DirectSumMatrix{Axis1, Axis2}, b::DirectSumVector{Axis1}) where {Axis1, Axis2} = DirectSumArray{Tuple{Axis2}}((flatview(A) \ flatview(b)).data)

Base.:(==)(A::DirectSumArray{Axes}, B::DirectSumArray{Axes}) where {Axes} = A.data == B.data
Base.isapprox(A::DirectSumArray{Axes}, B::DirectSumArray{Axes}; kwargs...) where {Axes} = isapprox(SVector(A.data), SVector(B.data); kwargs...)

# AD
unpack(x::Number) = x
_sizeinfo(::Type{<: DirectSumArray{Axes}}) where {Axes} = Axes

function ∂{N}(f, A::DirectSumArray{Axes}, ::Symbol) where {N, Axes}
    xs = unpack(A)
    dA = _pack(Axes, dualize(f, xs, Val(N)))
    df = f(dA)
    raws = extract_all(unpack(df), xs, Val(N))
    repack_extract_all(raws, _sizeinfo(typeof(df)), Axes, Val(N))
end

@generated function repack_extract_all(raws::Tuple, ::Type{OutAxes}, ::Type{InAxes}, ::Val{N}) where {OutAxes, InAxes, N}

    block_expr(expr, I) = foldl((ex, i) -> :($ex[$i]), I; init=expr)

    exps = Expr[]
    for k in N:-1:1
        axes_params = collect(OutAxes.parameters)
        for _ in 1:k
            append!(axes_params, InAxes.parameters)
        end
        Axes = Tuple{axes_params...}

        blkinds = CartesianIndices(_size(Axes))
        ex = Expr(:tuple, [block_expr(:(raws[$(N-k+1)]), Tuple(I)) for I in blkinds]...)
        push!(exps, :(_pack($Axes, $ex)))
    end

    if OutAxes == Tuple{} # scalar case
        push!(exps, :(raws[$(N+1)]))
    else
        blkinds = CartesianIndices(_size(OutAxes))
        ex = Expr(:tuple, [block_expr(:(raws[$(N+1)]), Tuple(I)) for I in blkinds]...)
        push!(exps, :(_pack(OutAxes, $ex)))
    end

    quote
        tuple($(exps...))
    end
end

@generated function blockspaces(::Type{Axes}) where {Axes}
    axes = Tuple(map(x -> x.parameters, Axes.parameters))
    dims = map(length, axes)

    S = Array{Any}(undef, dims...)
    for I in CartesianIndices(dims)
        spaces = map(Space, getindex.(axes, Tuple(I)))
        S[I] = ⊗(spaces...)
    end
    quote
        SArray{Tuple{$(dims...)}}(tuple($(S...)))
    end
end

function Base.summary(io::IO, A::DirectSumArray{Axes, T}) where {Axes, T}
    S = blockspaces(Axes)
    nd = ndims(S)
    print(io, Base.dims2string(size(S)), " ")
    if nd == 1
        print(io, "DirectSumVector")
    elseif nd == 2
        print(io, "DirectSumMatrix")
    else
        print(io, "DirectSumArray")
    end
    print(io, " with storage ", T)
end

function Base.show(io::IO, A::DirectSumArray{Axes}) where {Axes}
    S = blockspaces(Axes)
    summary(io, A)
    print(io, ":\n")
    Base.print_array(io, S)
end

"""
    flatview(A::DirectSumArray)

Return `A` as a flat coordinate array. Symmetric tensor blocks are
represented in Mandel coordinates.

See also: [`pack`](@ref), [`unpack`](@ref)

# Examples
```jldoctest
julia> A = @Mat[1.0 2.0; 3.0 4.0]
2×2 Tensor{Tuple{2, 2}, Float64, 2, 4}:
 1.0  2.0
 3.0  4.0

julia> x = pack(A, 3.0)
2-element DirectSumVector with storage Float64:
 Space(2, 2)
 Space()

julia> flatview(x)
5-element StaticArraysCore.SVector{5, Float64} with indices SOneTo(5):
 1.0
 3.0
 2.0
 4.0
 3.0

julia> As = symmetric(A)
2×2 SymmetricSecondOrderTensor{2, Float64, 3}:
 1.0  2.5
 2.5  4.0

julia> y = pack(As, 3.0)
2-element DirectSumVector with storage Float64:
 Space(Symmetry(2, 2),)
 Space()

julia> flatview(y)
4-element StaticArraysCore.SVector{4, Float64} with indices SOneTo(4):
 1.0
 3.5355339059327378
 4.0
 3.0
```
"""
function flatview(A::DirectSumArray)
    SArray{Tuple{flatsize(A)...}}(A.data)
end
