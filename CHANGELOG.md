# Change log

## v0.20

### New features

* Added rotation support for fourth-order tensors.
* Added `sqrt` for symmetric second-order tensors.
* Added stable automatic differentiation support for spectral `sqrt`, `exp`, and `log`.

### Improvements

* Fixed `:XZX` and `:xzx` Euler rotation sequences.
* Improved `@einsum` validation errors.
* Improved Tensorial component broadcasting, including faster tensor broadcast `copyto!`.

### Breaking changes

* `rotmat(a => b)` now returns the minimal rotation from `a` to `b`, instead of the previous Householder-style matrix based on `a + b`.

## v0.19

### New features

* `∂` is now the standard API for automatic differentiation. `gradient` and `hessian` remain available as aliases for `∂{1}` and `∂{2}`, respectively, so existing code using them continues to work.
* Automatic differentiation now supports multiple inputs and multiple outputs (see [docs](https://keitanakamura.github.io/Tensorial.jl/stable/Automatic%20differentiation/)).
* Repeated differentiation with respect to `Vec` now accounts for symmetry. In particular, Hessians and higher-order derivatives with respect to `Vec` are returned as symmetric tensors when appropriate.
* Added support for direct sums of mixed tensor and scalar variables, preserving block structure in differentiation and linear algebra.

### Breaking changes

* The Hessian with respect to `Vec` is now returned as a symmetric tensor type rather than a non-symmetric tensor type.

## v0.18

### Breaking changes

* Tensorial.jl is now built on [TensorCore.jl](https://github.com/JuliaMath/TensorCore.jl).
* Single contraction: `⋅` has been replaced by `⊡` (`⋅` now behaves as in `LinearAlgebra`).
* Double contraction: `⊡` has been replaced by `⊡₂` (which can be typed by `\boxdot<tab>\_2<tab>`).
* Broadcasting: Scalar-like behavior has been removed. Broadcasting now behaves the same as with other `AbstractArray`s.
* `mean`: The specialized `mean` definition in `Statistics` has been removed.
