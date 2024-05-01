@testset "Inverse" begin
    @testset "Adjugate matrix" begin
        for T in (Float32, Float64), dim in 1:4
            TOL = sqrt(eps(T))
            Random.seed!(1234)
            for A in (rand(SecondOrderTensor{dim, T}), rand(SymmetricSecondOrderTensor{dim, T}))
                c = rand(T)
                (@inferred Tensorial.adj(A))::typeof(A)
                @test Tensorial.adj(A) / det(A) ≈ inv(A)
                @test Tensorial.adj(one(A)) ≈ one(A)
                @test Tensorial.adj(c*A) ≈ c^(dim-1) * Tensorial.adj(A)
                @test Tensorial.adj(A)' ≈ Tensorial.adj(A')
                @test det(Tensorial.adj(A)) ≈ det(A)^(dim-1) atol=TOL
            end
        end
    end
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
        @testset "Fast version using block matrix algorithm" begin
            for T in (Float32, Float64), dim in 1:11
                Random.seed!(1234)
                x = rand(SecondOrderTensor{dim, T})
                y = rand(SymmetricSecondOrderTensor{dim, T})
                xinv = (@inferred Tensorial.fastinv(x))::typeof(x)
                yinv = (@inferred Tensorial.fastinv(y))::typeof(y)
                if T == Float64
                    @test xinv ⋅ x ≈ one(x)
                    @test yinv ⋅ y ≈ one(y)
                    @test xinv ≈ inv(x)
                    @test yinv ≈ inv(y)
                else
                    # not accurate enough to satisfy above conditions
                end
            end
        end
    end
end

@testset "Solving linear system" begin
    for T in (Float32, Float64), dim in 1:4
        Random.seed!(1234)
        A = rand(SecondOrderTensor{dim, T})
        S = rand(SymmetricSecondOrderTensor{dim, T})
        b = rand(Vec{dim, T})
        @test (@inferred A \ b)::Vec{dim, T} ≈ Array(A) \ Array(b)
        @test (@inferred S \ b)::Vec{dim, T} ≈ Array(S) \ Array(b)
    end
    @testset "Fast version using block matrix algorithm" begin
        for T in (Float32, Float64), dim in 1:11
            Random.seed!(1234)
            A = rand(SecondOrderTensor{dim, T})
            S = rand(SymmetricSecondOrderTensor{dim, T})
            b = rand(Vec{dim, T})
            if T == Float64
                @test (@inferred Tensorial.fastsolve(A, b))::Vec{dim, T} ≈ Array(A) \ Array(b)
                @test (@inferred Tensorial.fastsolve(S, b))::Vec{dim, T} ≈ Array(S) \ Array(b)
            else
                @test_throws Exception Tensorial.fastsolve(A, b)
                @test_throws Exception Tensorial.fastsolve(S, b)
            end
        end
    end
end
