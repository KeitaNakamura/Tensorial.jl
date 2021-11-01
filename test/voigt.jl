@testset "Voigt form" begin
    for dim in 1:3, T in (Float32, Float64)
        A = rand(Tensor{NTuple{2, dim}, T})
        AA = rand(Tensor{NTuple{4, dim}, T})
        A_sym = rand(Tensor{Tuple{@Symmetry({dim, dim})}, T})
        AA_sym = rand(Tensor{NTuple{2, @Symmetry({dim, dim})}, T})
        # tovoigt
        @test (@inferred tovoigt(AA)) * (@inferred tovoigt(A)) ≈ tovoigt(AA ⊡ A)
        @test (@inferred tovoigt(AA_sym)) * (@inferred tovoigt(A_sym, offdiagscale=T(2))) ≈ tovoigt(AA_sym ⊡ A_sym)
        # fromvoigt
        @test (@inferred fromvoigt(SecondOrderTensor{dim}, tovoigt(A)))::SecondOrderTensor{dim, T} ≈ A
        @test (@inferred fromvoigt(FourthOrderTensor{dim}, tovoigt(AA)))::FourthOrderTensor{dim, T} ≈ AA
        @test (@inferred fromvoigt(SymmetricSecondOrderTensor{dim}, tovoigt(A_sym, offdiagscale=T(2)), offdiagscale=T(2)))::SymmetricSecondOrderTensor{dim, T} ≈ A_sym
        @test (@inferred fromvoigt(SymmetricFourthOrderTensor{dim}, tovoigt(AA_sym, offdiagscale=T(2)), offdiagscale=T(2)))::SymmetricFourthOrderTensor{dim, T} ≈ AA_sym
        @test (@inferred fromvoigt(SecondOrderTensor{dim, Float64}, tovoigt(A)))::SecondOrderTensor{dim, Float64} ≈ A
        @test (@inferred fromvoigt(FourthOrderTensor{dim, Float64}, tovoigt(AA)))::FourthOrderTensor{dim, Float64} ≈ AA
        @test (@inferred fromvoigt(SymmetricSecondOrderTensor{dim, Float64}, tovoigt(A_sym, offdiagscale=T(2)), offdiagscale=T(2)))::SymmetricSecondOrderTensor{dim, Float64} ≈ A_sym
        @test (@inferred fromvoigt(SymmetricFourthOrderTensor{dim, Float64}, tovoigt(AA_sym, offdiagscale=T(2)), offdiagscale=T(2)))::SymmetricFourthOrderTensor{dim, Float64} ≈ AA_sym
        # tomandel
        @test (@inferred tomandel(AA_sym)) * (@inferred tomandel(A_sym)) ≈ tomandel(AA_sym ⊡ A_sym)
        # frommandel
        @test (@inferred frommandel(SymmetricSecondOrderTensor{dim}, tomandel(A_sym)))::SymmetricSecondOrderTensor{dim, T} ≈ A_sym
        @test (@inferred frommandel(SymmetricFourthOrderTensor{dim}, tomandel(AA_sym)))::SymmetricFourthOrderTensor{dim, T} ≈ AA_sym
        @test (@inferred frommandel(SymmetricSecondOrderTensor{dim, Float64}, tomandel(A_sym)))::SymmetricSecondOrderTensor{dim, Float64} ≈ A_sym
        @test (@inferred frommandel(SymmetricFourthOrderTensor{dim, Float64}, tomandel(AA_sym)))::SymmetricFourthOrderTensor{dim, Float64} ≈ AA_sym
        # error
        @test_throws Exception Tensorial.default_voigt_order(Val(4))
        @test_throws Exception tovoigt(rand(Mat{4,4}))
    end
end
