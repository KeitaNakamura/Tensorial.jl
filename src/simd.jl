const SIMDTypes = Union{Float16, Float32, Float64}

@generated function contraction(x::Tensor{<: Any, T, order1}, y::Tensor{<: Any, T, order2}, ::Val{N}) where {T <: SIMDTypes, N, order1, order2}
    S1 = Space(x)
    S2 = Space(y)
    S_Inner = Space((Dims(S2)[i] for i in 1:N)...)
    S1 = otimes(droplast(S1, Val(N)), S_Inner)
    S2 = otimes(S_Inner, dropfirst(S2, Val(N)))
    s1 = [:(Tuple(x)[$i]) for i in 1:ncomponents(S1)]
    s2 = [:(Tuple(y)[$i]) for i in 1:ncomponents(S2)]
    K = prod(Dims(S_Inner))
    I = length(s1) ÷ K
    J = length(s2) ÷ K
    s1′ = reshape(s1, I, K)
    s2′ = reshape(s2, K, J)
    # column slices of s1′
    slices = map(1:K) do k
        :($(Symbol(:v, k)) = SIMD.Vec($(s1′[:,k]...)))
    end
    # parallel computations along with columns
    columns = map(1:J) do j
        coefs = [s2′[k, j] for k in 1:K]
        ex = :(v1 * $(coefs[1]))
        for i in 2:length(coefs)
            ex = :(muladd($(Symbol(:v, i)), $(coefs[i]), $ex))
        end
        ex
    end
    # flatten data
    exps = Expr[]
    for j in 1:J, i in 1:I
        push!(exps, :($(columns[j])[$i]))
    end
    S = contraction(S1, S2, Val(N))
    if length(S) == 0
        TT = T
    else
        TT = tensortype(S){T}
    end
    quote
        @_inline_meta
        x = convert($(tensortype(S1)){T}, x)
        y = convert($(tensortype(S2)){T}, y)
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
