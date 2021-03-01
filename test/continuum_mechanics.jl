@testset "Continuum mechanics" begin
    @testset "mean/vol/dev" begin
        for T in (Float32, Float64)
            x = rand(SecondOrderTensor{3, T})
            y = rand(SymmetricSecondOrderTensor{3, T})
            # mean
            @test (@inferred mean(x))::T ≈ tr(Array(x)) / 3
            @test (@inferred mean(y))::T ≈ tr(Array(y)) / 3
            # vol/dev
            @test (@inferred vol(x))::typeof(x) + (@inferred dev(x))::typeof(x) ≈ x
            @test (@inferred vol(y))::typeof(y) + (@inferred dev(y))::typeof(y) ≈ y
            @test (@inferred vol(x))::typeof(x) ⊡ (@inferred dev(x))::typeof(x) ≈ zero(T)  atol = sqrt(eps(T))
            @test (@inferred vol(y))::typeof(y) ⊡ (@inferred dev(y))::typeof(y) ≈ zero(T)  atol = sqrt(eps(T))
        end
    end
    @testset "stress invariants" begin
        for T in (Float32, Float64), x in (rand(SecondOrderTensor{3, T}),
                                           rand(SymmetricSecondOrderTensor{3, T}))
            # stress invariants
            I₁, I₂, I₃ = (@inferred stress_invariants(x))::NamedTuple{(:I1,:I2,:I3), NTuple{3, T}}
            @test x^3 - I₁*x^2 + I₂*x - I₃*I ≈ zero(x)  atol = sqrt(eps(T))
            # deviatoric stress invariants
            J₁, J₂, J₃ = (@inferred deviatoric_stress_invariants(x))::NamedTuple{(:J1,:J2,:J3), NTuple{3, T}}
            @test dev(x)^3 - J₁*dev(x)^2 - J₂*dev(x) - J₃*I ≈ zero(x)  atol = sqrt(eps(T))
        end
    end
end
