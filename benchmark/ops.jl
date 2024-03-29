for T in (Float64, #=Float32=#)
    A3 = tensors[(Space(3), T)]
    A3x3 = tensors[(Space(3,3), T)]
    A3x3s = tensors[(Space(Symmetry(3,3)), T)]
    A3x3x3 = tensors[(Space(3,3,3), T)]
    A3x3x3x3 = tensors[(Space(3,3,3,3), T)]
    A3x3x3x3s = tensors[(Space(Symmetry(3,3),Symmetry(3,3)), T)]
    if run_array
        B3 = Array(A3)
        B3x3 = Array(A3x3)
        B3x3s = Array(A3x3s)
        B3x3x3 = Array(A3x3x3)
        B3x3x3x3 = Array(A3x3x3x3)
        B3x3x3x3s = Array(A3x3x3x3s)
    end

    # dot
    suite["Tensor"]["Space(3) ⋅ Space(3)"] = @benchmarkable $(Ref(A3))[] ⋅ $(Ref(A3))[]
    suite["Tensor"]["Space(3,3) ⋅ Space(3)"] = @benchmarkable $(Ref(A3x3))[] ⋅ $(Ref(A3))[]
    suite["Tensor"]["Space(Symmetry(3,3)) ⋅ Space(3)"] = @benchmarkable $(Ref(A3x3s)[]) ⋅ $(Ref(A3))[]
    if run_array
        suite["Array"]["Space(3) ⋅ Space(3)"] = @benchmarkable $(Ref(B3))[] ⋅ $(Ref(B3))[]
        suite["Array"]["Space(3,3) ⋅ Space(3)"] = @benchmarkable $(Ref(B3x3))[] * $(Ref(B3))[]
        suite["Array"]["Space(Symmetry(3,3)) ⋅ Space(3)"] = @benchmarkable $(Ref(B3x3s))[] * $(Ref(B3))[]
    end

    # double_contraction
    suite["Tensor"]["Space(3,3) ⊡ Space(3,3)"] = @benchmarkable $(Ref(A3x3))[] ⊡ $(Ref(A3x3))[]
    suite["Tensor"]["Space(Symmetry(3,3)) ⊡ Space(Symmetry(3,3))"] = @benchmarkable $(Ref(A3x3s))[] ⊡ $(Ref(A3x3s))[]
    suite["Tensor"]["Space(3,3,3) ⊡ Space(3,3)"] = @benchmarkable $(Ref(A3x3x3))[] ⊡ $(Ref(A3x3))[]
    suite["Tensor"]["Space(3,3,3,3) ⊡ Space(3,3)"] = @benchmarkable $(Ref(A3x3x3x3))[] ⊡ $(Ref(A3x3))[]
    suite["Tensor"]["Space(Symmetry(3,3),Symmetry(3,3)) ⊡ Space(Symmetry(3,3))"] = @benchmarkable $(Ref(A3x3x3x3s))[] ⊡ $(Ref(A3x3s))[]
    if run_array
        suite["Array"]["Space(3,3) ⊡ Space(3,3)"] = @benchmarkable $(Ref(B3x3))[] ⋅ $(Ref(B3x3))[]
        suite["Array"]["Space(Symmetry(3,3)) ⊡ Space(Symmetry(3,3))"] = @benchmarkable $(Ref(B3x3s))[] ⋅ $(Ref(B3x3s))[]
        suite["Array"]["Space(3,3,3) ⊡ Space(3,3)"] = @benchmarkable $(Ref(reshape(B3x3x3, 3, 9)))[] * $(Ref(reshape(B3x3, 9, 1)))[]
        suite["Array"]["Space(3,3,3,3) ⊡ Space(3,3)"] = @benchmarkable $(Ref(reshape(B3x3x3x3, 9, 9)))[] * $(Ref(reshape(B3x3, 9, 1)))[]
        suite["Array"]["Space(Symmetry(3,3),Symmetry(3,3)) ⊡ Space(Symmetry(3,3))"] = @benchmarkable $(Ref(reshape(B3x3x3x3s, 9, 9)))[] * $(Ref(reshape(B3x3s, 9, 1)))[]
    end

    # otimes
    suite["Tensor"]["Space(3) ⊗ Space(3)"] = @benchmarkable $(Ref(A3))[] ⊗ $(Ref(A3))[]
    if run_array
        suite["Array"]["Space(3) ⊗ Space(3)"] = @benchmarkable $(Ref(B3))[] * $(Ref(B3))[]'
    end

    # cross
    suite["Tensor"]["Space(3) × Space(3)"] = @benchmarkable $(Ref(A3))[] × $(Ref(A3))[]
    if run_array
        suite["Array"]["Space(3) × Space(3)"] = @benchmarkable $(Ref(B3))[] × $(Ref(B3))[]
    end

    # det
    suite["Tensor"]["det(Space(3,3))"] = @benchmarkable det($(Ref(A3x3))[])
    suite["Tensor"]["det(Space(Symmetry(3,3)))"] = @benchmarkable det($(Ref(A3x3s))[])
    if run_array
        suite["Array"]["det(Space(3,3))"] = @benchmarkable det($(Ref(B3x3))[])
        suite["Array"]["det(Space(Symmetry(3,3)))"] = @benchmarkable det($(Ref(B3x3s))[])
    end

    # inv
    suite["Tensor"]["inv(Space(3,3))"] = @benchmarkable inv($(Ref(A3x3))[])
    suite["Tensor"]["inv(Space(Symmetry(3,3)))"] = @benchmarkable inv($(Ref(A3x3s))[])
    suite["Tensor"]["inv(Space(3,3,3,3))"] = @benchmarkable inv($(Ref(A3x3x3x3))[])
    suite["Tensor"]["inv(Space(Symmetry(3,3),Symmetry(3,3)))"] = @benchmarkable inv($(Ref(A3x3x3x3s))[])
    if run_array
        suite["Array"]["inv(Space(3,3))"] = @benchmarkable inv($(Ref(B3x3))[])
        suite["Array"]["inv(Space(Symmetry(3,3)))"] = @benchmarkable inv($(Ref(B3x3s))[])
        suite["Array"]["inv(Space(3,3,3,3))"] = @benchmarkable inv($(Ref(reshape(B3x3x3x3, 9, 9)))[])
        suite["Array"]["inv(Space(Symmetry(3,3),Symmetry(3,3)))"] = @benchmarkable inv($(Ref(reshape(B3x3x3x3, 9, 9)))[]) # B3x3x3x3s is not invertible
    end
end
