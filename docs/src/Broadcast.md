# Broadcast

In Tensorial.jl, `Tensor` behaves like a scalar rather than an `Array` when performing broadcasting, as follows:

```@setup broadcast1
using Tensorial
```

```@repl broadcast1
x = Vec(1,2,3)
V = [Vec{3}(i:i+2) for i in 1:4]
x .+ V
V .= zero(x)
```

Conversely, broadcasting a `Tensor` itself or with scalars and tuples behaves the same as built-in `Array`, as shown below:

```@setup broadcast2
using Tensorial
```

```@repl broadcast2
x = Vec(1,2,3)
sqrt.(x)
x .+ 2
x .+ (2,3,4)
```
