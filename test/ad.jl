struct SquareMatrix{n, T, L} <: AbstractTensor{Tuple{n, n}, T, 2}
    data::NTuple{L, T}
end
SquareMatrix{n, T, L}(x::AbstractMatrix) where {n, T, L} = SquareMatrix{n, T, L}(Tuple(SecondOrderTensor{n}(x)))
Base.Tuple(x::SquareMatrix) = x.data
Base.getindex(x::SquareMatrix, i::Int) = x.data[i]
Base.rand(::Type{SquareMatrix{n, T}}) where {n, T} = SquareMatrix{n, T, n*n}(Tensorial.fill_tuple(()->rand(T), Val(n*n)))
Base.zero(::Type{<: SquareMatrix{n, T}}) where {n, T} = SquareMatrix{n, T, n*n}(Tuple(zero(SecondOrderTensor{n, T})))
Base.one(::Type{<: SquareMatrix{n, T}}) where {n, T} = SquareMatrix{n, T, n*n}(Tuple(one(SecondOrderTensor{n, T})))

@testset "Automatic differentiation" begin
    @testset "Real -> Real" begin
        for T in (Float32, Float64, Int)
            x = rand(T)
            @test (@inferred gradient(x -> 2x^2 - 3x + 2, x))::T == 4x - 3
            @test (@inferred gradient(x -> 2x^2 - 3x + 2, x, :all))::Tuple{T, T} == (4x - 3, 2x^2 - 3x + 2)
            @test (@inferred gradient(x -> 3, x))::Int == zero(T)
            @test (@inferred gradient(x -> 3, x, :all))::Tuple{Int, Int} == (zero(T), 3)
        end
    end
    @testset "Real -> Tensor" begin
        for RetType in (Tensor{Tuple{2,3}}, Tensor{Tuple{3, @Symmetry{3,3}}})
            for T in (Float32, Float64)
                @eval begin
                    x = rand($T)
                    @test (@inferred gradient(x -> $RetType((ij...) -> prod(ij) * x^2), x))::$RetType{$T} == $RetType((ij...) -> prod(ij) * 2x)
                    @test (@inferred gradient(x -> $RetType((ij...) -> prod(ij) * x^2), x, :all))::Tuple{$RetType{$T}, $RetType{$T}} == ($RetType((ij...) -> prod(ij) * 2x), $RetType((ij...) -> prod(ij) * x^2))
                    @test (@inferred gradient(x -> $RetType((ij...) -> prod(ij)), x))::$RetType{Int} == zero($RetType)
                    @test (@inferred gradient(x -> $RetType((ij...) -> prod(ij)), x, :all))::Tuple{$RetType{Int}, $RetType{Int}} == (zero($RetType), $RetType((ij...) -> prod(ij)))
                end
            end
        end
    end
    @testset "Tensor -> Real" begin
        I₁ = x -> tr(x)
        I₂ = x -> (tr(x)^2 - tr(x^2)) / 2
        I₃ = x -> det(x)
        for GivenType in (SecondOrderTensor, SymmetricSecondOrderTensor, SquareMatrix)
            for T in (Float32, Float64), dim in 1:4
                Random.seed!(1234)
                x = rand(GivenType{dim, T})
                if GivenType == SquareMatrix
                    RetType{dim, T} = SecondOrderTensor{dim, T}
                else
                    RetType{dim, T} = GivenType{dim, T}
                end
                @test (@inferred gradient(I₁, x))::RetType{dim, T} ≈ one(x)
                @test (@inferred gradient(I₂, x))::RetType{dim, T} ≈ I₁(x)*one(x) - x'
                @test (@inferred gradient(I₃, x))::RetType{dim, T} ≈ det(x)*inv(x)'
                @test (@inferred gradient(I₁, x, :all))[1]::RetType{dim, T} ≈ one(x)
                @test (@inferred gradient(I₂, x, :all))[1]::RetType{dim, T} ≈ I₁(x)*one(x) - x'
                @test (@inferred gradient(I₃, x, :all))[1]::RetType{dim, T} ≈ det(x)*inv(x)'
                @test (@inferred gradient(I₁, x, :all))[2]::T ≈ I₁(x)
                @test (@inferred gradient(I₂, x, :all))[2]::T ≈ I₂(x)
                @test (@inferred gradient(I₃, x, :all))[2]::T ≈ I₃(x)
                @test (@inferred gradient(x -> 1, x))::RetType{dim, Int} == zero(x)
                @test (@inferred gradient(x -> 1, x, :all))::Tuple{RetType{dim, Int}, Int} == (zero(x), 1)
            end
        end
    end
    @testset "Tensor -> Tensor" begin
        for T in (Float32, Float64), dim in 1:4
            x = rand(SecondOrderTensor{dim, T})
            y = rand(SymmetricSecondOrderTensor{dim, T})
            z = rand(SquareMatrix{dim, T})
            @test (@inferred gradient(identity, x))::FourthOrderTensor{dim, T} == one(FourthOrderTensor{dim, T})
            @test (@inferred gradient(identity, y))::SymmetricFourthOrderTensor{dim, T} == one(SymmetricFourthOrderTensor{dim, T})
            @test (@inferred gradient(identity, z))::FourthOrderTensor{dim, T} == one(FourthOrderTensor{dim, T})
            @eval begin
                @test (@inferred gradient(x -> one(SecondOrderTensor{$dim, Int}), $x))::FourthOrderTensor{$dim, Int} == zero(FourthOrderTensor{$dim, Int})
                @test (@inferred gradient(x -> one(SecondOrderTensor{$dim, Int}), $y))::Tensor{Tuple{$dim,$dim,@Symmetry{$dim,$dim}}, Int} == zero(FourthOrderTensor{$dim, Int})
                @test (@inferred gradient(x -> one(SquareMatrix{$dim, Int}), $z))::FourthOrderTensor{$dim, Int} == zero(FourthOrderTensor{$dim, Int})
            end
        end
    end
    @testset "Check nested differentiation" begin
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
    @testset "n-th derivatives" begin
        T = Float64
        x = rand(Vec{2, T})
        f = norm(x)
        ∂f = ∂(norm, x)
        ∂²f = ∂(x -> ∂(norm, x), x)
        ∂³f = ∂(x -> ∂(x -> ∂(norm, x), x), x)
        ∂⁴f = ∂(x -> ∂(x -> ∂(x -> ∂(norm, x), x), x), x)
        @test (@inferred ∂(norm, x))::Tensor{Tuple{2}, T} ≈ ∂f
        @test (@inferred ∂{2}(norm, x))::Tensor{Tuple{@Symmetry{2,2}}, T} ≈ ∂²f
        @test (@inferred ∂{3}(norm, x))::Tensor{Tuple{@Symmetry{2,2,2}}, T} ≈ ∂³f
        @test (@inferred ∂{4}(norm, x))::Tensor{Tuple{@Symmetry{2,2,2,2}}, T} ≈ ∂⁴f
        @test (@inferred ∂{0}(norm, x))::T ≈ f
        @test all((@inferred ∂(norm, x, :all)) .≈ (∂f, f))
        @test all((@inferred ∂{2}(norm, x, :all)) .≈ (∂²f, ∂f, f))
        @test all((@inferred ∂{3}(norm, x, :all)) .≈ (∂³f, ∂²f, ∂f, f))
        @test all((@inferred ∂{4}(norm, x, :all)) .≈ (∂⁴f, ∂³f, ∂²f, ∂f, f))
        @test all((@inferred ∂{0}(norm, x, :all)) .≈ (f,))
    end
    @testset "Multiple arguments" begin
        for T in (Float32, Float64)
            a = rand(T)
            x = rand(SymmetricSecondOrderTensor{2,T})
            @test (@inferred gradient((x,y) -> x * tr(y), a, x))::Tuple{T, SymmetricSecondOrderTensor{2,T}} === (tr(x), a*one(SymmetricSecondOrderTensor{2,T}))
            @test (@inferred gradient((x,y) -> x * tr(y), a, x, :all))::Tuple{Tuple{T, SymmetricSecondOrderTensor{2,T}}, T} === ((tr(x), a*one(SymmetricSecondOrderTensor{2,T})), a*tr(x))
            @test (@inferred gradient((x,y) -> x * y, a, x))::Tuple{SymmetricSecondOrderTensor{2,T}, SymmetricFourthOrderTensor{2,T}} === (x, a*one(SymmetricFourthOrderTensor{2,T}))
            @test (@inferred gradient((x,y) -> x * y, a, x, :all))::Tuple{Tuple{SymmetricSecondOrderTensor{2,T}, SymmetricFourthOrderTensor{2,T}}, SymmetricSecondOrderTensor{2,T}} === ((x, a*one(SymmetricFourthOrderTensor{2,T})), a*x)
            # nested differentiation
            @test gradient((x,z) -> x * gradient(y -> x + y, 1), 1, x) === (one(T), zero(x))
            @test gradient((x,z) -> x * gradient(y -> x + y, 1), 1, x, :all) === ((one(T), zero(x)), one(T))
        end
    end
    @testset "AD tuple-output support" begin
        @testset "single input, tuple output" begin
            ∇f, f = gradient(x -> (x, 3x^2), 2, :all)
            @test ∇f == (1, 12)
            @test f  == (2, 12)
            @test gradient(x -> (x, 3x^2), 2) == (1, 12)
        end
        @testset "multiple inputs, tuple output" begin
            ∇f, f = gradient((x, y) -> (x + y, x), 2, 3, :all)
            @test ∇f == ((1, 1), (1, 0))
            @test f  == (5, 2)
            @test gradient((x, y) -> (x + y, x), 2, 3) == ((1, 1), (1, 0))
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
    @testset "automatic differentiation symmetry" begin
        T = Float64

        @testset "single Vec input" begin
            x = rand(Vec{2,T})

            # Repeated derivatives with respect to a single Vec should be symmetrized.
            H = ∂{2}(x -> x ⋅ x, x)
            @test H isa SymmetricSecondOrderTensor{2,T}
            @test Tuple(Tensorial.Space(H)) == (Symmetry(2,2),)

            D3 = ∂{3}(x -> sum(x .^ 3), x)
            @test Tuple(Tensorial.Space(D3)) == (Symmetry(2,2,2),)
        end

        @testset "mixed derivatives with a matrix input" begin
            x = rand(Vec{2,T})
            A = rand(Mat{2,2,T})

            D3 = ∂{3}((x, A) -> (x ⋅ x) * tr(A), x, A)

            @test Tuple(Tensorial.Space(D3[1][1][2])) == (Symmetry(2,2), 2, 2)
            @test Tuple(Tensorial.Space(D3[1][2][1])) == (2, 2, 2, 2)
            @test Tuple(Tensorial.Space(D3[2][1][1])) == (2, 2, Symmetry(2,2))
        end

        @testset "multiple contiguous Vec runs" begin
            x = rand(Vec{1,T})
            y = rand(Vec{2,T})

            D4 = ∂{4}((x, y) -> (x ⋅ x) * (y ⋅ y), x, y)

            @test Tuple(Tensorial.Space(D4[1][1][2][2])) == (Symmetry(1,1), Symmetry(2,2))
            @test Tuple(Tensorial.Space(D4[1][2][2][1])) == (1, Symmetry(2,2), 1)
            @test Tuple(Tensorial.Space(D4[2][1][1][2])) == (2, Symmetry(1,1), 2)
        end

        @testset "mixed scalar and tensor inputs" begin
            x = rand(T)
            A = rand(SymmetricSecondOrderTensor{2,T})

            H = ∂{2}((x, A) -> x * tr(A), x, A)
            @test H isa Tuple
            @test H[1] isa Tuple
            @test H[2] isa Tuple

            @test H[1][1] isa T
            @test H[1][2] isa SymmetricSecondOrderTensor{2,T}
            @test H[2][1] isa SymmetricSecondOrderTensor{2,T}
            @test H[2][2] isa SymmetricFourthOrderTensor{2,T}

            H, G, F = ∂{2}((x, A) -> x * tr(A), x, A, :all)
            @test H[1][1] isa T
            @test H[1][2] isa SymmetricSecondOrderTensor{2,T}
            @test H[2][1] isa SymmetricSecondOrderTensor{2,T}
            @test H[2][2] isa SymmetricFourthOrderTensor{2,T}
            @test G isa Tuple{T, SymmetricSecondOrderTensor{2,T}}
            @test F isa T
        end
    end
end
