@testset "Automatic differentiation" begin
    @testset "API consistency" begin
        x = Vec(2.0, -1.0)
        f(x) = x ⋅ x

        @test ∂(f, x) == first(∂(f, x, :all))
        @test ∂{2}(f, x) == first(∂{2}(f, x, :all))
        @test gradient(f, x) == first(gradient(f, x, :all))

        a = 2.0
        g(a, x) = a * (x ⋅ x)

        @test gradient(g, a, x) == first(gradient(g, a, x, :all))
        @test ∂{2}(g, a, x) == first(∂{2}(g, a, x, :all))
        @test ∂{3}((a, x) -> a^2 * (x ⋅ x), a, x) == first(∂{3}((a, x) -> a^2 * (x ⋅ x), a, x, :all))
    end

    @testset "first-order basics" begin
        @testset "Real -> Real" begin
            for T in (Float32, Float64, Int)
                x = rand(T)
                @test (@inferred gradient(x -> 2x^2 - 3x + 2, x, :all))::Tuple{T, T} == (4x - 3, 2x^2 - 3x + 2)
                @test (@inferred gradient(x -> 3, x, :all))::Tuple{Int, Int} == (zero(T), 3)
            end
        end

        @testset "Real -> Tensor" begin
            for RetType in (Tensor{Tuple{2,3}}, Tensor{Tuple{3, @Symmetry{3,3}}})
                for T in (Float32, Float64)
                    @eval begin
                        x = rand($T)
                        @test (@inferred gradient(x -> $RetType((ij...) -> prod(ij) * x^2), x, :all))::Tuple{$RetType{$T}, $RetType{$T}} ==
                              ($RetType((ij...) -> prod(ij) * 2x), $RetType((ij...) -> prod(ij) * x^2))
                        @test (@inferred gradient(x -> $RetType((ij...) -> prod(ij)), x, :all))::Tuple{$RetType{Int}, $RetType{Int}} ==
                              (zero($RetType), $RetType((ij...) -> prod(ij)))
                    end
                end
            end
        end

        @testset "Tensor -> Real" begin
            I₁ = x -> tr(x)
            I₂ = x -> (tr(x)^2 - tr(x^2)) / 2
            I₃ = x -> det(x)

            for GivenType in (SecondOrderTensor, SymmetricSecondOrderTensor)
                for T in (Float32, Float64), dim in 1:4
                    Random.seed!(1234)
                    x = rand(GivenType{dim, T})

                    y = @inferred gradient(I₁, x, :all)
                    @test y[1]::GivenType{dim, T} ≈ one(x)
                    @test y[2]::T ≈ I₁(x)

                    y = @inferred gradient(I₂, x, :all)
                    @test y[1]::GivenType{dim, T} ≈ I₁(x) * one(x) - x'
                    @test y[2]::T ≈ I₂(x)

                    y = @inferred gradient(I₃, x, :all)
                    @test y[1]::GivenType{dim, T} ≈ det(x) * inv(x)'
                    @test y[2]::T ≈ I₃(x)

                    @test (@inferred gradient(x -> 1, x, :all))::Tuple{GivenType{dim, Int}, Int} == (zero(x), 1)
                end
            end
        end

        @testset "Tensor -> Tensor" begin
            for T in (Float32, Float64), dim in 1:4
                x = rand(SecondOrderTensor{dim, T})
                y = rand(SymmetricSecondOrderTensor{dim, T})

                @test (@inferred gradient(identity, x, :all))::Tuple{FourthOrderTensor{dim, T}, SecondOrderTensor{dim, T}} ==
                      (one(FourthOrderTensor{dim, T}), x)

                @test (@inferred gradient(identity, y, :all))::Tuple{SymmetricFourthOrderTensor{dim, T}, SymmetricSecondOrderTensor{dim, T}} ==
                      (one(SymmetricFourthOrderTensor{dim, T}), y)

                @eval begin
                    @test (@inferred gradient(x -> one(SecondOrderTensor{$dim, Int}), $x, :all))::Tuple{FourthOrderTensor{$dim, Int}, SecondOrderTensor{$dim, Int}} ==
                          (zero(FourthOrderTensor{$dim, Int}), one(SecondOrderTensor{$dim, Int}))

                    @test (@inferred gradient(x -> one(SecondOrderTensor{$dim, Int}), $y, :all))::Tuple{Tensor{Tuple{$dim,$dim,@Symmetry{$dim,$dim}}, Int}, SecondOrderTensor{$dim, Int}} ==
                          (zero(FourthOrderTensor{$dim, Int}), one(SecondOrderTensor{$dim, Int}))
                end
            end
        end
    end

    @testset "nested differentiation" begin
        @test gradient(x -> x * gradient(y -> x + y, 1), 1) == 1
        @test gradient(v -> sum(v) * gradient(y -> y * norm(v), 1), Vec(1,2)) ≈ gradient(v -> sum(v) * norm(v), Vec(1,2))

        f(a::Number, v::Vec{2}) = a^2 * v[1]^2 * 2v[2]^3
        dfda(a::Number, v::Vec{2}) = 2a * v[1]^2 * 2v[2]^3
        dfdv(a::Number, v::Vec{2}) = a^2 * Vec(2v[1] * 2v[2]^3, v[1]^2 * 6v[2]^2)

        a = rand()
        v = rand(Vec{2})

        @test gradient(v -> gradient(a -> f(a, v), a), v) ≈ gradient(v -> dfda(a, v), v)
        @test gradient(a -> gradient(v -> f(a, v), v), a) ≈ gradient(a -> dfdv(a, v), a)
        @test gradient(v -> sum(v) * gradient(a -> f(a, v), a), v) ≈ gradient(v -> sum(v) * dfda(a, v), v)
        @test gradient(a -> a * gradient(v -> f(a, v), v), a) ≈ gradient(a -> a * dfdv(a, v), a)
    end

    @testset "single-input higher-order derivatives" begin
        T = Float64
        x = rand(Vec{2, T})

        f = norm(x)
        ∂f = ∂(norm, x)
        ∂²f = ∂(x -> ∂(norm, x), x)
        ∂³f = ∂(x -> ∂(x -> ∂(norm, x), x), x)
        ∂⁴f = ∂(x -> ∂(x -> ∂(x -> ∂(norm, x), x), x), x)

        @test all((@inferred ∂(norm, x, :all)) .≈ (∂f, f))
        @test all((@inferred ∂{2}(norm, x, :all)) .≈ (∂²f, ∂f, f))
        @test all((@inferred ∂{3}(norm, x, :all)) .≈ (∂³f, ∂²f, ∂f, f))
        @test all((@inferred ∂{4}(norm, x, :all)) .≈ (∂⁴f, ∂³f, ∂²f, ∂f, f))
        @test all((@inferred ∂{0}(norm, x, :all)) .≈ (f,))

        H = @inferred ∂{2}(x -> x ⋅ x, x)
        @test H isa SymmetricSecondOrderTensor{2,T}
        @test Tuple(Tensorial.Space(H)) == (Symmetry(2,2),)

        D3 = @inferred ∂{3}(x -> sum(x .^ 3), x)
        @test Tuple(Tensorial.Space(D3)) == (Symmetry(2,2,2),)
    end

    @testset "multiple-input first-order derivatives" begin
        for T in (Float32, Float64)
            a = rand(T)
            x = rand(SymmetricSecondOrderTensor{2,T})

            y = @inferred gradient((x,y) -> x * tr(y), a, x, :all)
            @test y::Tuple{Tuple{T, SymmetricSecondOrderTensor{2,T}}, T} ==
                  ((tr(x), a * one(SymmetricSecondOrderTensor{2,T})), a * tr(x))

            y = @inferred gradient((x,y) -> x * y, a, x, :all)
            @test y::Tuple{Tuple{SymmetricSecondOrderTensor{2,T}, SymmetricFourthOrderTensor{2,T}}, SymmetricSecondOrderTensor{2,T}} ==
                  ((x, a * one(SymmetricFourthOrderTensor{2,T})), a * x)

            @test gradient((x,z) -> x * gradient(y -> x + y, 1), 1, x, :all) == ((one(T), zero(x)), one(T))
        end
    end

    @testset "higher-order symmetry structure" begin
        T = Float64

        @testset "mixed derivatives with a matrix input" begin
            x = rand(Vec{2,T})
            A = rand(Mat{2,2,T})

            D3 = @inferred ∂{3}((x, A) -> (x ⋅ x) * tr(A), x, A)

            @test Tuple(Tensorial.Space(D3[1][1][2])) == (Symmetry(2,2), 2, 2)
            @test Tuple(Tensorial.Space(D3[1][2][1])) == (2, 2, 2, 2)
            @test Tuple(Tensorial.Space(D3[2][1][1])) == (2, 2, Symmetry(2,2))
        end

        @testset "multiple contiguous Vec runs" begin
            x = rand(Vec{1,T})
            y = rand(Vec{2,T})

            D4 = @inferred ∂{4}((x, y) -> (x ⋅ x) * (y ⋅ y), x, y)

            @test Tuple(Tensorial.Space(D4[1][1][2][2])) == (Symmetry(1,1), Symmetry(2,2))
            @test Tuple(Tensorial.Space(D4[1][2][2][1])) == (1, Symmetry(2,2), 1)
            @test Tuple(Tensorial.Space(D4[2][1][1][2])) == (2, Symmetry(1,1), 2)
        end

        @testset "mixed scalar and tensor inputs" begin
            x = rand(T)
            A = rand(SymmetricSecondOrderTensor{2,T})

            H, G, F = ∂{2}((x, A) -> x * tr(A), x, A, :all)
            @test H[1][1] isa T
            @test H[1][2] isa SymmetricSecondOrderTensor{2,T}
            @test H[2][1] isa SymmetricSecondOrderTensor{2,T}
            @test H[2][2] isa SymmetricFourthOrderTensor{2,T}
            @test G isa Tuple{T, SymmetricSecondOrderTensor{2,T}}
            @test F isa T
        end
    end

    @testset "higher-order derivative values" begin
        @testset "symmetric second-order tensor input" begin
            A = rand(SymmetricSecondOrderTensor{2})
            B = rand(SymmetricSecondOrderTensor{2})

            f(A) = 0.5 * (A ⊡₂ A)
            g(A) = A ⊡₂ A
            h(A) = tr(A)^2

            H, G, F = ∂{2}(f, A, :all)
            @test F ≈ 0.5 * (A ⊡₂ A)
            @test G ≈ A
            @test H ⊡₂ B ≈ B
            @test (H ⊡₂ B) ⊡₂ B ≈ (B ⊡₂ B)

            H, G, F = ∂{2}(g, A, :all)
            @test F ≈ A ⊡₂ A
            @test G ≈ 2A
            @test H ⊡₂ B ≈ 2B
            @test (H ⊡₂ B) ⊡₂ B ≈ 2(B ⊡₂ B)

            H, G, F = ∂{2}(h, A, :all)
            @test F ≈ tr(A)^2
            @test G ≈ 2tr(A) * one(A)
            @test H ⊡₂ B ≈ 2tr(B) * one(A)
            @test (H ⊡₂ B) ⊡₂ B ≈ 2tr(B)^2
        end

        @testset "multiple inputs: scalar and Vec" begin
            T = Float64

            @testset "second derivatives" begin
                a = T(2.0)
                x = Vec{2,T}(3.0, 5.0)

                f(a, x) = a * (x ⋅ x)

                H, G, F = ∂{2}(f, a, x, :all)

                @test F == a * (x ⋅ x)
                @test G == ((x ⋅ x), 2a * x)
                @test H[1][1] == zero(T)
                @test H[1][2] == 2x
                @test H[2][1] == 2x
                @test H[2][2] == 2a * one(SymmetricSecondOrderTensor{2,T})
            end

            @testset "third derivatives" begin
                a = T(2.0)
                x = Vec{2,T}(3.0, 5.0)

                f(a, x) = a^2 * (x ⋅ x)

                D3, H, G, F = ∂{3}(f, a, x, :all)

                @test F == a^2 * (x ⋅ x)
                @test G == (2a * (x ⋅ x), 2a^2 * x)
                @test H[1][1] == 2(x ⋅ x)
                @test H[1][2] == 4a * x
                @test H[2][1] == 4a * x
                @test H[2][2] == 2a^2 * one(SymmetricSecondOrderTensor{2,T})

                @test D3[1][1][1] == zero(T)
                @test D3[1][1][2] == 4x
                @test D3[1][2][1] == 4x
                @test D3[2][1][1] == 4x
                @test D3[1][2][2] == 4a * one(SymmetricSecondOrderTensor{2,T})
                @test D3[2][1][2] == 4a * one(SymmetricSecondOrderTensor{2,T})
                @test D3[2][2][1] == 4a * one(SymmetricSecondOrderTensor{2,T})
                @test D3[2][2][2] == zero(D3[2][2][2])
            end
        end

        @testset "multiple inputs: Vec and Mat" begin
            T = Float64
            x = Vec{2,T}(2.0, -1.0)
            A = Mat{2,2,T}(1.0, 2.0,
                           3.0, 4.0)

            f(x, A) = (x ⋅ x) * tr(A)

            H, G, F = ∂{2}(f, x, A, :all)

            @test F == (x ⋅ x) * tr(A)
            @test G == (2tr(A) * x, (x ⋅ x) * one(SecondOrderTensor{2,T}))
            @test H[1][1] == 2tr(A) * one(SymmetricSecondOrderTensor{2,T})
            @test H[1][2] == 2x ⊗ one(SecondOrderTensor{2,T})
            @test H[2][1] == one(SecondOrderTensor{2,T}) ⊗ 2x
            @test H[2][2] == zero(FourthOrderTensor{2,T})
        end
    end

    @testset "tuple-output support" begin
        @testset "single input, tuple output" begin
            ∇f, f = gradient(x -> (x, 3x^2), 2, :all)
            @test ∇f == (1, 12)
            @test f == (2, 12)
        end

        @testset "multiple inputs, tuple output" begin
            ∇f, f = gradient((x, y) -> (x + y, x), 2, 3, :all)
            @test ∇f == ((1, 1), (1, 0))
            @test f == (5, 2)
        end

        @testset "mixed scalar/tensor outputs" begin
            x = Vec(2.0, -1.0)
            ∇f, f = gradient(x -> (x ⋅ x, 2x), x, :all)
            @test f[1] == 5.0
            @test f[2] == Vec(4.0, -2.0)
            @test ∇f[1] == Vec(4.0, -2.0)
            @test ∇f[2] == 2 * one(Mat{2,2})
        end

        @testset "consider_symmetry_result for tuple outputs" begin
            x = Vec(2.0, -1.0)
            H, G, F = ∂{2}(x -> (x ⋅ x, 3(x ⋅ x)), x, :all)
            @test F == (5.0, 15.0)
            @test G == (Vec(4.0, -2.0), Vec(12.0, -6.0))
            @test H == (2 * one(Mat{2,2}), 6 * one(Mat{2,2}))
            @test H[1] isa SymmetricSecondOrderTensor{2}
            @test H[2] isa SymmetricSecondOrderTensor{2}
        end

        @testset "tuple output with constant tensor branch" begin
            ∇f, f = gradient(x -> (zero(Mat{2,2}), Mat{2,2}(x,x,x,x)), 2.0, :all)
            @test f == (zero(Mat{2,2}), Mat{2,2}(2.0, 2.0, 2.0, 2.0))
            @test ∇f[1] == zero(Mat{2,2})
            @test ∇f[2] == Mat{2,2}(1.0, 1.0, 1.0, 1.0)
        end
    end
end
