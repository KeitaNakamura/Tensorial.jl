@testset "Broadcast" begin
    for T in (Float32, Float64)
        v = rand(Vec{3, T})
        A = rand(Mat{2, 3, T})
        vs = [rand(Vec{3, T}) for _ in 1:4]
        As = [rand(Mat{2, 3, T}) for _ in 1:4]
        # with array (tensors behave like scalar)
        @test (@inferred(v .+ vs))::Vector{Vec{3, T}} ≈ map(y -> v + y, vs)
        @test (@inferred(v .- vs))::Vector{Vec{3, T}} ≈ map(y -> v - y, vs)
        @test (@inferred(v .+ vs .- vs .+ v))::Vector{Vec{3, T}} ≈ map(y -> v + y - y + v, vs)
        @test (vs .= v)::Vector{Vec{3, T}} ≈ map(y -> v, vs)
        @test (@inferred(A .+ As))::Vector{Mat{2, 3, T, 6}} ≈ map(y -> A + y, As)
        @test (@inferred(A .- As))::Vector{Mat{2, 3, T, 6}} ≈ map(y -> A - y, As)
        @test (@inferred(A .+ As .- As .+ A))::Vector{Mat{2, 3, T, 6}} ≈ map(y -> A + y - y + A, As)
        @test (As .= A)::Vector{Mat{2, 3, T, 6}} ≈ map(y -> A, As)
        # with scalar
        @test (@inferred(v .+ 1))::Vec{3, T} ≈ map(y -> y + 1, v)
        @test (@inferred(A .+ 1))::Mat{2, 3, T} ≈ map(y -> y + 1, A)
        @test (@inferred(v .+ 1 .+ 2 .+ v))::Vec{3, T} ≈ map(y -> y + 1 + 2 + y, v)
        @test (@inferred(A .+ 1 .+ 2 .+ A))::Mat{2, 3, T} ≈ map(y -> y + 1 + 2 + y, A)
        # with tuple
        @test (@inferred(v .+ (1,2,3)))::Vec{3, T} ≈ map(+, v, (1,2,3))
        @test (@inferred(v .- (1,2,3)))::Vec{3, T} ≈ map(-, v, (1,2,3))
    end
end
