# Automatic differentiation

!!! warn
    The user must provide the appropriate tensor symmetry information;
    otherwise, automatic differentiation may return unexpected values.
    In the following example, even with identical tensor values,
    the results vary depending on the `Tensor` type.
    ```@setup automatic-differentiation
    using Tensorial
    ```
    ```@repl automatic-differentiation
    A = rand(Mat{3,3})
    S = A ⋅ A' # `S` is symmetric but not of the type `SymmetricSecondOrderTensor`
    gradient(identity, S) ≈ one(FourthOrderTensor{3})
    gradient(identity, symmetric(S)) ≈ one(SymmetricFourthOrderTensor{3})
    ```

```@docs
gradient
hessian
```
