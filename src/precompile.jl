using SnoopPrecompile

@precompile_all_calls begin
    for dim in (2, 3)
        v = ones(Vec{dim, Float64})
        σ = one(SymmetricSecondOrderTensor{dim, Float64})
        F = one(SecondOrderTensor{dim, Float64})
        E = one(SymmetricFourthOrderTensor{dim, Float64})
        C = one(FourthOrderTensor{dim, Float64})
        v * 1.0
        v ⋅ v
        v ⊗ v
        σ ⋅ v
        F ⋅ v
        σ ⊗ σ
        F ⊗ F
        σ * 1.0
        F * 1.0
        σ ⊡ σ
        F ⊡ F
        E ⊡ σ
        C ⊡ F
        E * 1.0
        C * 1.0
    end
end
