# default voigt order
# not sure appropriate order for dim ≥ 4
@generated function default_voigt_order(::Val{dim}) where {dim}
    inds = Vector{Tuple{Int, Int}}(undef, dim*dim)
    count = 0
    for i in 1:dim
        @inbounds inds[count+=1] = (i,i)
    end
    # upper
    for j in dim:-1:1
        for i in j-1:-1:1
            @inbounds inds[count+=1] = (i,j)
        end
    end
    # lower
    for i in dim:-1:1
        for j in i-1:-1:1
            @inbounds inds[count+=1] = (i,j)
        end
    end
    quote
        SVector{$(length(inds))}($inds)
    end
end

"""
    tovoigt(A::Union{SecondOrderTensor, FourthOrderTensor}; [order])
    tovoigt(A::Union{SymmetricSecondOrderTensor, SymmetricFourthOrderTensor}; [order, offdiagonal])

Convert a tensor to Voigt form.

Keyword arguments:
 - `offdiagscale`: Determines the scaling factor for the offdiagonal elements.
 - `order`: A vector of cartesian indices (`Tuple{Int, Int}`) determining the Voigt order.
   The default order is `[(1,1), (2,2), (3,3), (2,3), (1,3), (1,2), (3,2), (3,1), (2,1)]`

See also [`fromvoigt`](@ref).

```jldoctest
julia> x = Mat{3,3}(1:9...)
3×3 Tensor{Tuple{3, 3}, Int64, 2, 9}:
 1  4  7
 2  5  8
 3  6  9

julia> tovoigt(x)
9-element StaticArrays.SVector{9, Int64} with indices SOneTo(9):
 1
 5
 9
 8
 7
 4
 6
 3
 2

julia> x = SymmetricSecondOrderTensor{3}(1:6...)
3×3 SymmetricSecondOrderTensor{3, Int64, 6}:
 1  2  3
 2  4  5
 3  5  6

julia> tovoigt(x; offdiagscale = 2,
                  order = [(1,1), (2,2), (3,3), (1,2), (1,3), (2,3)])
6-element StaticArrays.SVector{6, Int64} with indices SOneTo(6):
  1
  4
  6
  4
  6
 10
```
"""
function tovoigt end

@generated function tovoigt(x::SecondOrderTensor{dim};
                            order::AbstractVector{Tuple{Int, Int}} = default_voigt_order(Val(dim))) where {dim}
    L = ncomponents(x)
    exps = [:(x[order[$i]...]) for i in 1:L]
    quote
        @_propagate_inbounds_meta
        @assert length(order) == $L
        SVector{$L}($(exps...))
    end
end

@generated function tovoigt(x::FourthOrderTensor{dim};
                            order::AbstractVector{Tuple{Int, Int}} = default_voigt_order(Val(dim))) where {dim}
    L = Int(sqrt(ncomponents(x)))
    exps = [:(x[order[$i][1], order[$i][2], order[$j][1], order[$j][2]]) for i in 1:L, j in 1:L]
    quote
        @_propagate_inbounds_meta
        @assert length(order) == $L
        SMatrix{$L, $L}($(exps...))
    end
end

@generated function tovoigt(x::SymmetricSecondOrderTensor{dim, T};
                            order::AbstractVector{Tuple{Int, Int}} = default_voigt_order(Val(dim)),
                            offdiagscale::T = one(T)) where {dim, T}
    L = ncomponents(x)
    function tovoigt_element(i)
        ex = :(x[i,j])
        quote
            i, j = order[$i]
            (i == j ? $ex : $ex * offdiagscale)
        end
    end
    exps = map(tovoigt_element, 1:L)
    quote
        @_propagate_inbounds_meta
        @assert length(order) ≥ $L
        @inbounds SVector{$L, T}($(exps...))
    end
end

@generated function tovoigt(x::SymmetricFourthOrderTensor{dim, T};
                            order::AbstractVector{Tuple{Int, Int}} = default_voigt_order(Val(dim)),
                            offdiagscale::T = one(T)) where {dim, T}
    L = Int(sqrt(ncomponents(x)))
    function tovoigt_element(i, j)
        ex = :(x[i,j,k,l])
        quote
            i, j = order[$i]
            k, l = order[$j]
            ((i == j && k == l) ? $ex :
             (i == j || k == l) ? $ex * offdiagscale :
                                  $ex * (offdiagscale * offdiagscale))
        end
    end
    exps = [tovoigt_element(i, j) for i in 1:L, j in 1:L]
    quote
        @_propagate_inbounds_meta
        @assert length(order) ≥ $L
        SMatrix{$L, $L, T}($(exps...))
    end
end

"""
    tomandel(A::Union{SymmetricSecondOrderTensor, SymmetricFourthOrderTensor})

Convert a tensor to Mandel form which is equivalent to `tovoigt(A, offdiagscale = √2)`.

See also [`tovoigt`](@ref).
"""
@inline function tomandel(x::Union{SymmetricSecondOrderTensor, SymmetricFourthOrderTensor})
    @_propagate_inbounds_meta
    tovoigt(x, offdiagscale = eltype(x)(√2))
end


"""
    fromvoigt(S::Type{<: Union{SecondOrderTensor, FourthOrderTensor}}, A::AbstractArray{T}; [order])
    fromvoigt(S::Type{<: Union{SymmetricSecondOrderTensor, SymmetricFourthOrderTensor}}, A::AbstractArray{T}; [order, offdiagscale])

Converts an array `A` stored in Voigt format to a Tensor of type `S`.

Keyword arguments:
 - `offdiagscale`: Determines the scaling factor for the offdiagonal elements.
 - `order`: A vector of cartesian indices (`Tuple{Int, Int}`) determining the Voigt order.
   The default order is `[(1,1), (2,2), (3,3), (2,3), (1,3), (1,2), (3,2), (3,1), (2,1)]`

!!! note
    Since `offdiagscale` is the scaling factor for the offdiagonal elements in **Voigt form**,
    they are multiplied by `1/offdiagscale` in `fromvoigt` unlike [`tovoigt`](@ref).
    Thus `fromvoigt(tovoigt(x, offdiagscale = 2), offdiagscale = 2)` returns original `x`.

See also [`tovoigt`](@ref).

```jldoctest
julia> fromvoigt(Mat{3,3}, 1.0:1.0:9.0)
3×3 Tensor{Tuple{3, 3}, Float64, 2, 9}:
 1.0  6.0  5.0
 9.0  2.0  4.0
 8.0  7.0  3.0

julia> fromvoigt(SymmetricSecondOrderTensor{3},
                 1.0:1.0:6.0,
                 offdiagscale = 2.0,
                 order = [(1,1), (2,2), (3,3), (1,2), (1,3), (2,3)])
3×3 SymmetricSecondOrderTensor{3, Float64, 6}:
 1.0  2.0  2.5
 2.0  2.0  3.0
 2.5  3.0  3.0
```
"""
function fromvoigt end

@generated function fromvoigt(::Type{TT},
                              v::AbstractVector;
                              order::AbstractVector{Tuple{Int, Int}} = default_voigt_order(Val(dim))) where {dim, TT <: Tensor{Tuple{dim, dim}}}
    S = Space(TT)
    L = ncomponents(S)
    inds = independent_indices(S)
    T = eltype(TT) == Any ? eltype(v) : eltype(TT)
    quote
        @_propagate_inbounds_meta
        @assert length(v) == length(order) == $L
        data = MVector{$L, $T}(undef)
        for I in 1:$L
            i, j = order[I]
            data[$inds[i,j]] = v[I]
        end
        TT(Tuple(data))
    end
end

@generated function fromvoigt(::Type{TT},
                              v::AbstractMatrix;
                              order::AbstractVector{Tuple{Int, Int}} = default_voigt_order(Val(dim))) where {dim, TT <: Tensor{NTuple{4, dim}}}
    S = Space(TT)
    L = Int(sqrt(ncomponents(S)))
    inds = independent_indices(S)
    T = eltype(TT) == Any ? eltype(v) : eltype(TT)
    quote
        @_propagate_inbounds_meta
        @assert length(v) == $(L*L)
        @assert length(order) == $L
        data = MVector{$(L*L), $T}(undef)
        for J in 1:$L, I in 1:$L
            i, j = order[I]
            k, l = order[J]
            data[$inds[i,j,k,l]] = v[I,J]
        end
        TT(Tuple(data))
    end
end

@generated function fromvoigt(::Type{TT},
                              v::AbstractVector{T};
                              order::AbstractVector{Tuple{Int, Int}} = default_voigt_order(Val(dim)),
                              offdiagscale::T = one(T)) where {dim, T, TT <: Tensor{Tuple{@Symmetry{dim, dim}}}}
    S = Space(TT)
    L = ncomponents(S)
    inds = independent_indices(S)
    T = eltype(TT) == Any ? eltype(v) : eltype(TT)
    quote
        @_propagate_inbounds_meta
        @assert length(v) == $L
        @assert length(order) ≥ $L
        data = MVector{$L, $T}(undef)
        for I in 1:$L
            i, j = order[I]
            if i == j
                data[$inds[i,j]] = v[I]
            else
                data[$inds[i,j]] = v[I] / offdiagscale
            end
        end
        TT(Tuple(data))
    end
end

@generated function fromvoigt(::Type{TT},
                              v::AbstractMatrix{T};
                              order::AbstractVector{Tuple{Int, Int}} = default_voigt_order(Val(dim)),
                              offdiagscale::T = one(T)) where {dim, T, TT <: Tensor{NTuple{2, @Symmetry{dim, dim}}}}
    S = Space(TT)
    L = Int(sqrt(ncomponents(S)))
    inds = independent_indices(S)
    T = eltype(TT) == Any ? eltype(v) : eltype(TT)
    quote
        @_propagate_inbounds_meta
        @assert length(v) == $(L*L)
        @assert length(order) ≥ $L
        data = MVector{$(L*L), $T}(undef)
        for J in 1:$L, I in 1:$L
            i, j = order[I]
            k, l = order[J]
            if i == j && k == l
                data[$inds[i,j,k,l]] = v[I,J]
            elseif i == j || k == l
                data[$inds[i,j,k,l]] = v[I,J] / offdiagscale
            else
                data[$inds[i,j,k,l]] = v[I,J] / (offdiagscale * offdiagscale)
            end
        end
        TT(Tuple(data))
    end
end

"""
    frommandel(S::Type{<: Union{SymmetricSecondOrderTensor, SymmetricFourthOrderTensor}}, A::AbstractArray{T})

Create a tensor of type `S` from Mandel form.
This is equivalent to `fromvoigt(S, A, offdiagscale = √2)`.

See also [`fromvoigt`](@ref).
"""
@inline function frommandel(TT::Type{<: Union{SymmetricSecondOrderTensor, SymmetricFourthOrderTensor}}, v::AbstractArray{T}) where T
    @_propagate_inbounds_meta
    fromvoigt(TT, v, offdiagscale = T(√2))
end
