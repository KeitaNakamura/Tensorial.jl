struct SquareMatrix{n, T, L} <: AbstractTensor{Tuple{n, n}, T, 2}
    data::NTuple{L, T}
end
SquareMatrix{n, T, L}(x::AbstractMatrix) where {n, T, L} = SquareMatrix{n, T, L}(Tuple(SecondOrderTensor{n}(x)))
Base.Tuple(x::SquareMatrix) = x.data
Base.getindex(x::SquareMatrix, i::Int) = x.data[i]
Base.rand(::Type{SquareMatrix{n, T}}) where {n, T} = SquareMatrix{n, T, n*n}(Tensorial.fill_tuple(()->rand(T), Val(n*n)))
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
        for RetType in (Tensor{Tuple{2,3}}, Tensor{Tuple{3, @Symmetry({3,3})}})
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
                @test (@inferred gradient(x -> one(SecondOrderTensor{$dim, Int}), $y))::Tensor{Tuple{$dim,$dim,@Symmetry({$dim,$dim})}, Int} == zero(FourthOrderTensor{$dim, Int})
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
end
