@testset "Continuum mechanics" begin
    @testset "mean/vol/dev" begin
        for T in (Float32, Float64)
            x = rand(SecondOrderTensor{3, T})
            y = rand(SymmetricSecondOrderTensor{3, T})
            # mean
            @test (@inferred mean(x))::T ≈ tr(Array(x)) / 3
            @test (@inferred mean(y))::T ≈ tr(Array(y)) / 3
            # vol/dev for 2nd-order tensors
            @test (@inferred vol(x))::typeof(x) + (@inferred dev(x))::typeof(x) ≈ x
            @test (@inferred vol(y))::typeof(y) + (@inferred dev(y))::typeof(y) ≈ y
            @test (@inferred vol(x))::typeof(x) ⊡ (@inferred dev(x))::typeof(x) ≈ zero(T)  atol = sqrt(eps(T))
            @test (@inferred vol(y))::typeof(y) ⊡ (@inferred dev(y))::typeof(y) ≈ zero(T)  atol = sqrt(eps(T))

            # vol/dev for vectors
            v = rand(Vec{3, T})
            @test (@inferred vol(v)) + (@inferred dev(v)) ≈ v
            @test sum(vol(v)) ≈ sum(v)
            @test dev(vol(v)) ≈ zero(v) atol = sqrt(eps(T))
            @test vol(dev(v)) ≈ zero(v) atol = sqrt(eps(T))

            # vol/dev for 4th-order tensors
            δ = one(SymmetricSecondOrderTensor{3, T})
            I_vol = @einsum (i,j,k,l) -> δ[i,j]*δ[k,l]/3
            I_dev = @einsum (i,j,k,l) -> δ[i,k]*δ[j,l] - δ[i,j]*δ[k,l]/3
            Isym_dev = @einsum (i,j,k,l) -> (δ[i,k]*δ[j,l]+δ[i,l]*δ[j,k])/2 - δ[i,j]*δ[k,l]/3
            @test (@inferred vol(FourthOrderTensor{3}))::FourthOrderTensor{3, Float64}                   ≈ I_vol
            @test (@inferred dev(FourthOrderTensor{3}))::FourthOrderTensor{3, Float64}                   ≈ I_dev
            @test (@inferred vol(FourthOrderTensor{3, T}))::FourthOrderTensor{3, T}                      ≈ I_vol
            @test (@inferred dev(FourthOrderTensor{3, T}))::FourthOrderTensor{3, T}                      ≈ I_dev
            @test (@inferred vol(SymmetricFourthOrderTensor{3}))::SymmetricFourthOrderTensor{3, Float64} ≈ I_vol
            @test (@inferred dev(SymmetricFourthOrderTensor{3}))::SymmetricFourthOrderTensor{3, Float64} ≈ Isym_dev
            @test (@inferred vol(SymmetricFourthOrderTensor{3, T}))::SymmetricFourthOrderTensor{3, T}    ≈ I_vol
            @test (@inferred dev(SymmetricFourthOrderTensor{3, T}))::SymmetricFourthOrderTensor{3, T}    ≈ Isym_dev
            @test (vol(FourthOrderTensor{3}) + dev(FourthOrderTensor{3}))::FourthOrderTensor{3} ≈ one(FourthOrderTensor{3})
            @test (vol(SymmetricFourthOrderTensor{3}) + dev(SymmetricFourthOrderTensor{3}))::SymmetricFourthOrderTensor{3} ≈ one(SymmetricFourthOrderTensor{3})
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
