# just copied from test/Tensor.jl
@testset "MTensor Constructors" begin
    @testset "Inner constructors" begin
        @test (@inferred MTensor{Tuple{2,2}, Int, 2, 4}((1,2,3,4)))::MTensor{Tuple{2,2}, Int, 2, 4} |> Tuple == (1,2,3,4)
        @test (@inferred MTensor{Tuple{2,2}, Float64, 2, 4}((1,2,3,4)))::MTensor{Tuple{2,2}, Float64, 2, 4} |> Tuple == (1.0,2.0,3.0,4.0)
        @test (@inferred MTensor{Tuple{2,2}, Int, 2, 4}((1.0,2,3,4)))::MTensor{Tuple{2,2}, Int, 2, 4} |> Tuple == (1,2,3,4)
        @test (@inferred MTensor{Tuple{2,2}, Float64, 2, 4}((1.0,2,3,4)))::MTensor{Tuple{2,2}, Float64, 2, 4} |> Tuple == (1.0,2.0,3.0,4.0)
        @test (@inferred MTensor{Tuple{2,Symmetry{Tuple{2,2}}}, Int, 3, 6}((1,2,3,4,5,6)))::MTensor{Tuple{2,Symmetry{Tuple{2,2}}}, Int, 3, 6} |> Tuple == (1,2,3,4,5,6)
        @test (@inferred MTensor{Tuple{2,Symmetry{Tuple{2,2}}}, Float64, 3, 6}((1,2,3,4,5,6)))::MTensor{Tuple{2,Symmetry{Tuple{2,2}}}, Float64, 3, 6} |> Tuple == (1.0,2.0,3.0,4.0,5.0,6.0)

        # bad input
        @test_throws Exception MTensor{Tuple{2,2}, Int, 2, 4}((1,2,3))
        @test_throws Exception MTensor{Tuple{2,2}, Int, 2, 4}(())
        @test_throws Exception MTensor{Tuple{2}, Vector{Int}, 1, 2}(([1,2],[3,4]))

        # bad parameters
        @test_throws Exception MTensor{Tuple{Int,2}, Int, 2, 4}((1,2,3,4)) # bad size
        @test_throws Exception MTensor{Tuple{2,2}, Int, 2, 5}((1,2,3,4,5)) # bad ncomponents
        @test_throws Exception MTensor{Tuple{2,2}, Int, 3, 4}((1,2,3,4))   # bad ndims
        @test_throws Exception MTensor{Tuple{2,Symmetry{Int,2}}, Int, 3, 6}((1,2,3,4,5,6)) # bad Symmetry size
        @test_throws Exception MTensor{Tuple{2,Symmetry{2,2}}, Int, 3, 5}((1,2,3,4,5))     # bad ncomponents
        @test_throws Exception MTensor{Tuple{2,Symmetry{2,2}}, Int, 2, 6}((1,2,3,4,5,6))   # bad ndims
    end
    @testset "getindex/setindex!" begin
        for T in (Float32, Float64)
            x = rand(MTensor{Tuple{3, 3}, T})
            y = Array(x)
            z = 1:9
            for i in eachindex(x, y, z)
                @test (@inferred x[i])::T == y[i]
                @test (@inferred setindex!(x, z[i], i)) === x
            end
            for i in eachindex(x, z)
                @test x[i] == z[i]
            end
        end
    end
end
