@testset "Inversion of second and fourth order tensors" begin
    for T in (Float32, Float64), dim in 1:4
        Random.seed!(1234)
        # second order
        x = rand(SecondOrderTensor{dim, T})
        y = rand(SymmetricSecondOrderTensor{dim, T})
        @test (@inferred inv(x))::typeof(x) ⋅ x ≈ one(x)
        @test (@inferred inv(y))::typeof(y) ⋅ y ≈ one(y)
        # fourth order
        dim > 3 && continue
        x = rand(FourthOrderTensor{dim, T})
        y = rand(SymmetricFourthOrderTensor{dim, T})
        @test (@inferred inv(x))::typeof(x) ⊡ x ≈ one(x)
        @test (@inferred inv(y))::typeof(y) ⊡ y ≈ one(y)
    end
end

@testset "Solving linear system" begin
    for T in (Float32, Float64), dim in 1:4
        Random.seed!(1234)
        A = rand(SecondOrderTensor{dim, T})
        S = rand(SymmetricSecondOrderTensor{dim, T})
        b = rand(Vec{dim, T})
        @test (@inferred A \ b)::Vec{dim, T} |> Array ≈ Array(A) \ Array(b)
        @test (@inferred S \ b)::Vec{dim, T} |> Array ≈ Array(S) \ Array(b)
    end
end
