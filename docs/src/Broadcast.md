```@meta
DocTestSetup = :(using Tensorial)
```

# Broadcast

In Tensorial.jl, subtypes of `AbstractTensor` basically behave like scalars rather than `Array`.
For example, broadcasting operations on tensors and arrays of tensors will be performed as

```jldoctest
julia> x = Vec(1,2,3)
3-element Vec{3, Int64}:
 1
 2
 3

julia> V = [Vec{3}(i:i+2) for i in 1:4]
4-element Vector{Vec{3, Int64}}:
 [1, 2, 3]
 [2, 3, 4]
 [3, 4, 5]
 [4, 5, 6]

julia> x .+ V
4-element Vector{Vec{3, Int64}}:
 [2, 4, 6]
 [3, 5, 7]
 [4, 6, 8]
 [5, 7, 9]

julia> V .= zero(x)
4-element Vector{Vec{3, Int64}}:
 [0, 0, 0]
 [0, 0, 0]
 [0, 0, 0]
 [0, 0, 0]
```

On the other hand, broadcasting itself or with scalars and tuples behave the same as built-in `Array` as

```jldoctest
julia> x = Vec(1,2,3)
3-element Vec{3, Int64}:
 1
 2
 3

julia> sqrt.(x)
3-element Vec{3, Float64}:
 1.0
 1.4142135623730951
 1.7320508075688772

julia> x .+ 2
3-element Vec{3, Int64}:
 3
 4
 5

julia> x .+ (2,3,4)
3-element Vec{3, Int64}:
 3
 5
 7
```
