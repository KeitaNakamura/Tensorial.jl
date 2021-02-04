const SIMDTypes = Union{Float16, Float32, Float64}

# TODO: implement more efficient computations for symmetric case
@generated function contraction(x::Tensor{<: Any, T}, y::Tensor{<: Any, T}, ::Val{N}) where {T <: SIMDTypes, N}
    S1 = Size(x)
    S2 = Size(y)
    S = contraction(S1, S2, Val(N))
    s1 = map(i -> EinsumIndex(:(Tuple(x)), i), independent_indices(S1))
    s2 = map(i -> EinsumIndex(:(Tuple(y)), i), independent_indices(S2))
    J = prod(size(s2)[1:N])
    I = length(s1) ÷ J
    K = length(s2) ÷ J
    s1′ = reshape(s1, I, J)
    s2′ = reshape(s2, J, K)
    # Create slices
    # [a c
    #  b d] => [[a,b], [c,d]]
    slices = map(axes(s1′, 2)) do j
        :($(Symbol(:v, j)) = SIMD.Vec($(map(construct_expr, s1′[:,j])...)))
    end
    columns = map(axes(s2′, 2)) do j
        coefs = map(axes(s2′, 1)) do i
            construct_expr(s2′[i, j])
        end
        code = :(v1 * $(coefs[1]))
        for i in 2:length(coefs)
            code = :(muladd($(Symbol(:v, i)), $(coefs[i]), $code))
        end
        code
    end
    exps = map(indices(S)) do i
        d, r = divrem(i-1, I) .+ 1
        :(columns[$d][$r])
    end
    if length(S) == 0
        TT = T
    else
        TT = tensortype(S){T}
    end
    quote
        @_inline_meta
        @inbounds begin
            $(slices...)
            columns = tuple($(columns...))
            $TT($(exps...))
        end
    end
end

for op in (:+, :-)
    @eval @inline function Base.$op(x::TT, y::TT) where {T <: SIMDTypes, TT <: Tensor{<: Any, T}}
        TT(Tuple($op(SIMD.Vec(Tuple(x)), SIMD.Vec(Tuple(y)))))
    end
end

for op in (:*, :/)
    @eval @inline function Base.$op(x::TT, a::T) where {T <: SIMDTypes, TT <: Tensor{<: Any, T}}
        TT(Tuple($op(SIMD.Vec(Tuple(x)), a)))
    end
end
@inline function Base.:*(a::T, x::TT) where {T <: SIMDTypes, TT <: Tensor{<: Any, T}}
    x * a
end
