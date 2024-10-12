@testset "Permutation" begin
    @testset "permutedims" begin
        @testset "Vec" begin
            for T in (Float64, Float32)
                for dim in 1:10
                    x = rand(Vec{dim, T})
                    @test (@inferred permutedims(x)) === Mat{1,dim,T}(x)
                end
            end
        end
        @testset "Space" begin
            S = Space(4, 3, Symmetry(2, 2))

            @test (@inferred permutedims(S, Val((1,2,3,4)))) == Space(4, 3, Symmetry(2, 2))
            @test (@inferred permutedims(S, Val((1,2,4,3)))) == Space(4, 3, Symmetry(2, 2))
            @test (@inferred permutedims(S, Val((1,3,2,4)))) == Space(4, 2, 3, 2)
            @test (@inferred permutedims(S, Val((1,3,4,2)))) == Space(4, Symmetry(2, 2), 3)
            @test (@inferred permutedims(S, Val((1,4,2,3)))) == Space(4, 2, 3, 2)
            @test (@inferred permutedims(S, Val((1,4,3,2)))) == Space(4, Symmetry(2, 2), 3)

            @test (@inferred permutedims(S, Val((2,1,3,4)))) == Space(3, 4, Symmetry(2, 2))
            @test (@inferred permutedims(S, Val((2,1,4,3)))) == Space(3, 4, Symmetry(2, 2))
            @test (@inferred permutedims(S, Val((2,3,1,4)))) == Space(3, 2, 4, 2)
            @test (@inferred permutedims(S, Val((2,3,4,1)))) == Space(3, Symmetry(2, 2), 4)
            @test (@inferred permutedims(S, Val((2,4,1,3)))) == Space(3, 2, 4, 2)
            @test (@inferred permutedims(S, Val((2,4,3,1)))) == Space(3, Symmetry(2, 2), 4)

            @test (@inferred permutedims(S, Val((3,1,2,4)))) == Space(2, 4, 3, 2)
            @test (@inferred permutedims(S, Val((3,1,4,2)))) == Space(2, 4, 2, 3)
            @test (@inferred permutedims(S, Val((3,2,1,4)))) == Space(2, 3, 4, 2)
            @test (@inferred permutedims(S, Val((3,2,4,1)))) == Space(2, 3, 2, 4)
            @test (@inferred permutedims(S, Val((3,4,1,2)))) == Space(Symmetry(2, 2), 4, 3)
            @test (@inferred permutedims(S, Val((3,4,2,1)))) == Space(Symmetry(2, 2), 3, 4)

            @test (@inferred permutedims(S, Val((4,1,2,3)))) == Space(2, 4, 3, 2)
            @test (@inferred permutedims(S, Val((4,1,3,2)))) == Space(2, 4, 2, 3)
            @test (@inferred permutedims(S, Val((4,2,1,3)))) == Space(2, 3, 4, 2)
            @test (@inferred permutedims(S, Val((4,2,3,1)))) == Space(2, 3, 2, 4)
            @test (@inferred permutedims(S, Val((4,3,1,2)))) == Space(Symmetry(2, 2), 4, 3)
            @test (@inferred permutedims(S, Val((4,3,2,1)))) == Space(Symmetry(2, 2), 3, 4)
        end
        @testset "Tensor" begin
            for T in (Float32, Float64)
                x = rand(Tensorial.tensortype(Space(4, 3, Symmetry(2, 2))){T})

                @test (@inferred permutedims(x, Val((1,2,3,4))))::Tensor{Tuple{4, 3, @Symmetry{2, 2}}} ≈ permutedims(x, [1,2,3,4])
                @test (@inferred permutedims(x, Val((1,2,4,3))))::Tensor{Tuple{4, 3, @Symmetry{2, 2}}} ≈ permutedims(x, [1,2,4,3])
                @test (@inferred permutedims(x, Val((1,3,2,4))))::Tensor{Tuple{4, 2, 3, 2}} ≈ permutedims(x, [1,3,2,4])
                @test (@inferred permutedims(x, Val((1,3,4,2))))::Tensor{Tuple{4, @Symmetry{2, 2}, 3}} ≈ permutedims(x, [1,3,4,2])
                @test (@inferred permutedims(x, Val((1,4,2,3))))::Tensor{Tuple{4, 2, 3, 2}} ≈ permutedims(x, [1,4,2,3])
                @test (@inferred permutedims(x, Val((1,4,3,2))))::Tensor{Tuple{4, @Symmetry{2, 2}, 3}} ≈ permutedims(x, [1,4,3,2])

                @test (@inferred permutedims(x, Val((2,1,3,4))))::Tensor{Tuple{3, 4, @Symmetry{2, 2}}} ≈ permutedims(x, [2,1,3,4])
                @test (@inferred permutedims(x, Val((2,1,4,3))))::Tensor{Tuple{3, 4, @Symmetry{2, 2}}} ≈ permutedims(x, [2,1,4,3])
                @test (@inferred permutedims(x, Val((2,3,1,4))))::Tensor{Tuple{3, 2, 4, 2}} ≈ permutedims(x, [2,3,1,4])
                @test (@inferred permutedims(x, Val((2,3,4,1))))::Tensor{Tuple{3, @Symmetry{2, 2}, 4}} ≈ permutedims(x, [2,3,4,1])
                @test (@inferred permutedims(x, Val((2,4,1,3))))::Tensor{Tuple{3, 2, 4, 2}} ≈ permutedims(x, [2,4,1,3])
                @test (@inferred permutedims(x, Val((2,4,3,1))))::Tensor{Tuple{3, @Symmetry{2, 2}, 4}} ≈ permutedims(x, [2,4,3,1])

                @test (@inferred permutedims(x, Val((3,1,2,4))))::Tensor{Tuple{2, 4, 3, 2}} ≈ permutedims(x, [3,1,2,4])
                @test (@inferred permutedims(x, Val((3,1,4,2))))::Tensor{Tuple{2, 4, 2, 3}} ≈ permutedims(x, [3,1,4,2])
                @test (@inferred permutedims(x, Val((3,2,1,4))))::Tensor{Tuple{2, 3, 4, 2}} ≈ permutedims(x, [3,2,1,4])
                @test (@inferred permutedims(x, Val((3,2,4,1))))::Tensor{Tuple{2, 3, 2, 4}} ≈ permutedims(x, [3,2,4,1])
                @test (@inferred permutedims(x, Val((3,4,1,2))))::Tensor{Tuple{@Symmetry{2, 2}, 4, 3}} ≈ permutedims(x, [3,4,1,2])
                @test (@inferred permutedims(x, Val((3,4,2,1))))::Tensor{Tuple{@Symmetry{2, 2}, 3, 4}} ≈ permutedims(x, [3,4,2,1])

                @test (@inferred permutedims(x, Val((4,1,2,3))))::Tensor{Tuple{2, 4, 3, 2}} ≈ permutedims(x, [4,1,2,3])
                @test (@inferred permutedims(x, Val((4,1,3,2))))::Tensor{Tuple{2, 4, 2, 3}} ≈ permutedims(x, [4,1,3,2])
                @test (@inferred permutedims(x, Val((4,2,1,3))))::Tensor{Tuple{2, 3, 4, 2}} ≈ permutedims(x, [4,2,1,3])
                @test (@inferred permutedims(x, Val((4,2,3,1))))::Tensor{Tuple{2, 3, 2, 4}} ≈ permutedims(x, [4,2,3,1])
                @test (@inferred permutedims(x, Val((4,3,1,2))))::Tensor{Tuple{@Symmetry{2, 2}, 4, 3}} ≈ permutedims(x, [4,3,1,2])
                @test (@inferred permutedims(x, Val((4,3,2,1))))::Tensor{Tuple{@Symmetry{2, 2}, 3, 4}} ≈ permutedims(x, [4,3,2,1])
            end
        end
    end
end
