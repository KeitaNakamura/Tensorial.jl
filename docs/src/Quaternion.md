```@meta
CurrentModule = Tensorial
DocTestSetup = :(using Tensorial)
```

# Quaternion

Quaternions are mainly useful for 3D rotations. Tensorial provides a small
`Quaternion` type, angle-axis constructors, conversion to rotation matrices, and
vector rotation.

!!! note
    Quaternion support is experimental and may change in a future version.

```@setup quaternion
using Tensorial
```

## Construction and access

A quaternion stores a scalar part and a 3D vector part:

```@repl quaternion
q = Quaternion(1.0, 2.0, 3.0, 4.0)
q.scalar
q.vector
Tuple(q)
```

You can also construct a pure-vector quaternion from a `Vec`:

```@repl quaternion
Quaternion(Vec(1.0, 2.0, 3.0))
```

## Angle-axis rotations

Use [`quaternion`](@ref) to construct a unit quaternion from an angle and a unit
axis:

```@repl quaternion
axis = Vec(0.0, 0.0, 1.0)
q = quaternion(π/4, axis)
θ, n = angleaxis(q)
θ ≈ π/4
n ≈ axis
```

The quaternion and the corresponding rotation matrix represent the same
rotation:

```@repl quaternion
v = @Vec [1.0, 0.0, 0.0]
rotate(v, q) ≈ rotmat(q) * v
```

For quaternion docstrings, see [Rotations and quaternions](@ref).
