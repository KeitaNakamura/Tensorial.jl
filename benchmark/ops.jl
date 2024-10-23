for T in (Float64, #=Float32=#)
    A3 = tensors[(Space(3), T)]
    A3x3 = tensors[(Space(3,3), T)]
    A3x3s = tensors[(Space(Symmetry(3,3)), T)]
    A3x3x3x3 = tensors[(Space(3,3,3,3), T)]
    A3x3x3x3s = tensors[(Space(Symmetry(3,3),Symmetry(3,3)), T)]
    # Array
    B3 = Array(A3)
    B3x3 = Array(A3x3)
    B3x3s = Array(A3x3s)
    B3x3_voigt = Array(tovoigt(A3x3))
    B3x3s_voigt = Array(tovoigt(A3x3s))
    B3x3x3x3_voigt = Array(tovoigt(A3x3x3x3))
    B3x3x3x3s_voigt = Array(tovoigt(A3x3x3x3s))
    # SArray
    C3 = SArray(A3)
    C3x3 = SArray(A3x3)
    C3x3s = SArray(A3x3s)
    C3x3_voigt = tovoigt(A3x3)
    C3x3s_voigt = tovoigt(A3x3s)
    C3x3x3x3_voigt = tovoigt(A3x3x3x3)
    C3x3x3x3s_voigt = tovoigt(A3x3x3x3s)

    # dot
    suite["Tensor"]["Space(3) ⊡ Space(3)"] = @benchmarkable $(Ref(A3))[] ⊡ $(Ref(A3))[]
    suite["Tensor"]["Space(3,3) ⊡ Space(3)"] = @benchmarkable $(Ref(A3x3))[] ⊡ $(Ref(A3))[]
    suite["Tensor"]["Space(Symmetry(3,3)) ⊡ Space(3)"] = @benchmarkable $(Ref(A3x3s)[]) ⊡ $(Ref(A3))[]
    suite["Array"]["Space(3) ⊡ Space(3)"] = @benchmarkable $(Ref(B3))[] ⋅ $(Ref(B3))[]
    suite["Array"]["Space(3,3) ⊡ Space(3)"] = @benchmarkable $(Ref(B3x3))[] * $(Ref(B3))[]
    suite["Array"]["Space(Symmetry(3,3)) ⊡ Space(3)"] = @benchmarkable $(Ref(B3x3s))[] * $(Ref(B3))[]
    suite["SArray"]["Space(3) ⊡ Space(3)"] = @benchmarkable $(Ref(C3))[] ⋅ $(Ref(C3))[]
    suite["SArray"]["Space(3,3) ⊡ Space(3)"] = @benchmarkable $(Ref(C3x3))[] * $(Ref(C3))[]
    suite["SArray"]["Space(Symmetry(3,3)) ⊡ Space(3)"] = @benchmarkable $(Ref(C3x3s))[] * $(Ref(C3))[]

    # double_contraction
    suite["Tensor"]["Space(3,3) ⊡₂ Space(3,3)"] = @benchmarkable $(Ref(A3x3))[] ⊡₂ $(Ref(A3x3))[]
    suite["Tensor"]["Space(Symmetry(3,3)) ⊡₂ Space(Symmetry(3,3))"] = @benchmarkable $(Ref(A3x3s))[] ⊡₂ $(Ref(A3x3s))[]
    suite["Tensor"]["Space(3,3,3,3) ⊡₂ Space(3,3)"] = @benchmarkable $(Ref(A3x3x3x3))[] ⊡₂ $(Ref(A3x3))[]
    suite["Tensor"]["Space(Symmetry(3,3),Symmetry(3,3)) ⊡₂ Space(Symmetry(3,3))"] = @benchmarkable $(Ref(A3x3x3x3s))[] ⊡₂ $(Ref(A3x3s))[]
    suite["Array"]["Space(3,3) ⊡₂ Space(3,3)"] = @benchmarkable $(Ref(B3x3_voigt))[] ⋅ $(Ref(B3x3_voigt))[]
    suite["Array"]["Space(Symmetry(3,3)) ⊡₂ Space(Symmetry(3,3))"] = @benchmarkable $(Ref(B3x3s_voigt))[] ⋅ $(Ref(B3x3s_voigt))[]
    suite["Array"]["Space(3,3,3,3) ⊡₂ Space(3,3)"] = @benchmarkable $(Ref(B3x3x3x3_voigt))[] * $(Ref(B3x3_voigt))[]
    suite["Array"]["Space(Symmetry(3,3),Symmetry(3,3)) ⊡₂ Space(Symmetry(3,3))"] = @benchmarkable $(Ref(B3x3x3x3s_voigt))[] * $(Ref(B3x3s_voigt))[]
    suite["SArray"]["Space(3,3) ⊡₂ Space(3,3)"] = @benchmarkable $(Ref(C3x3))[] ⋅ $(Ref(C3x3))[]
    suite["SArray"]["Space(Symmetry(3,3)) ⊡₂ Space(Symmetry(3,3))"] = @benchmarkable $(Ref(C3x3s))[] ⋅ $(Ref(C3x3s))[]
    suite["SArray"]["Space(3,3,3,3) ⊡₂ Space(3,3)"] = @benchmarkable $(Ref(C3x3x3x3_voigt))[] * $(Ref(C3x3_voigt))[]
    suite["SArray"]["Space(Symmetry(3,3),Symmetry(3,3)) ⊡₂ Space(Symmetry(3,3))"] = @benchmarkable $(Ref(C3x3x3x3s_voigt))[] * $(Ref(C3x3s_voigt))[]

    # otimes
    suite["Tensor"]["Space(3) ⊗ Space(3)"] = @benchmarkable $(Ref(A3))[] ⊗ $(Ref(A3))[]
    suite["Array"]["Space(3) ⊗ Space(3)"] = @benchmarkable $(Ref(B3))[] * $(Ref(B3))[]'
    suite["SArray"]["Space(3) ⊗ Space(3)"] = @benchmarkable $(Ref(C3))[] * $(Ref(C3))[]'

    # cross
    suite["Tensor"]["Space(3) × Space(3)"] = @benchmarkable $(Ref(A3))[] × $(Ref(A3))[]
    suite["Array"]["Space(3) × Space(3)"] = @benchmarkable $(Ref(B3))[] × $(Ref(B3))[]
    suite["SArray"]["Space(3) × Space(3)"] = @benchmarkable $(Ref(C3))[] × $(Ref(C3))[]

    # det
    suite["Tensor"]["det(Space(3,3))"] = @benchmarkable det($(Ref(A3x3))[])
    suite["Tensor"]["det(Space(Symmetry(3,3)))"] = @benchmarkable det($(Ref(A3x3s))[])
    suite["Array"]["det(Space(3,3))"] = @benchmarkable det($(Ref(B3x3))[])
    suite["Array"]["det(Space(Symmetry(3,3)))"] = @benchmarkable det($(Ref(B3x3s))[])
    suite["SArray"]["det(Space(3,3))"] = @benchmarkable det($(Ref(C3x3))[])
    suite["SArray"]["det(Space(Symmetry(3,3)))"] = @benchmarkable det($(Ref(C3x3s))[])

    # inv
    suite["Tensor"]["inv(Space(3,3))"] = @benchmarkable inv($(Ref(A3x3))[])
    suite["Tensor"]["inv(Space(Symmetry(3,3)))"] = @benchmarkable inv($(Ref(A3x3s))[])
    suite["Tensor"]["inv(Space(3,3,3,3))"] = @benchmarkable inv($(Ref(A3x3x3x3))[])
    suite["Tensor"]["inv(Space(Symmetry(3,3),Symmetry(3,3)))"] = @benchmarkable inv($(Ref(A3x3x3x3s))[])
    suite["Array"]["inv(Space(3,3))"] = @benchmarkable inv($(Ref(B3x3))[])
    suite["Array"]["inv(Space(Symmetry(3,3)))"] = @benchmarkable inv($(Ref(B3x3s))[])
    suite["Array"]["inv(Space(3,3,3,3))"] = @benchmarkable inv($(Ref(B3x3x3x3_voigt))[])
    suite["Array"]["inv(Space(Symmetry(3,3),Symmetry(3,3)))"] = @benchmarkable inv($(Ref(B3x3x3x3s_voigt))[])
    suite["SArray"]["inv(Space(3,3))"] = @benchmarkable inv($(Ref(C3x3))[])
    suite["SArray"]["inv(Space(Symmetry(3,3)))"] = @benchmarkable inv($(Ref(C3x3s))[])
    suite["SArray"]["inv(Space(3,3,3,3))"] = @benchmarkable inv($(Ref(C3x3x3x3_voigt))[])
    suite["SArray"]["inv(Space(Symmetry(3,3),Symmetry(3,3)))"] = @benchmarkable inv($(Ref(C3x3x3x3s_voigt))[])
end
