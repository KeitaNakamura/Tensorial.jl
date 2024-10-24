@testset "Broadcast" begin
    for T in (Float32, Float64)
        v = rand(Vec{3, T})
        A = rand(Mat{2, 3, T})
        v_array = Array(v)
        A_array = Array(A)
        # with arrays
        @test (@inferred(v .+ v_array))::Vector{T} ≈ Array(v) .+ v_array
        @test (@inferred(A .* A_array))::Matrix{T} ≈ Array(A) .* A_array
        @test (@inferred(v .+ v'))::Matrix{T} ≈ v_array .+ v_array'
        @test (@inferred(v' .+ v))::Matrix{T} ≈ v_array' .+ v_array
        # with scalar
        @test (@inferred(v .+ 1))::Vec{3, T} ≈ map(y -> y + 1, v)
        @test (@inferred(A .+ 1))::Mat{2, 3, T} ≈ map(y -> y + 1, A)
        @test (@inferred(v .+ 1 .+ 2 .+ v))::Vec{3, T} ≈ map(y -> y + 1 + 2 + y, v)
        @test (@inferred(A .+ 1 .+ 2 .+ A))::Mat{2, 3, T} ≈ map(y -> y + 1 + 2 + y, A)
        # with tuple
        @test (@inferred(v .+ (1,2,3)))::Vec{3, T} ≈ map(+, v, (1,2,3))
        @test (@inferred(v .- (1,2,3)))::Vec{3, T} ≈ map(-, v, (1,2,3))
        # others
        @test (@inferred(broadcast(sqrt, v)))::Vec{3, T} ≈ map(sqrt, v)
        @test (@inferred(broadcast(sqrt, A)))::Mat{2, 3, T} ≈ map(sqrt, A)
    end
end
