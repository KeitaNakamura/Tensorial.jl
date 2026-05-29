_broadcast_dotted_scalar(v, μ) = @. 2μ * v
_broadcast_dotted_nested_scalar(v, μ) = @. (2μ + one(μ)) * v
_broadcast_dotted_function_scalar(v, μ) = @. sin(μ) * v
_broadcast_dotted_scalar_tuple(v, μ) = @. 2μ * v * (true, true, false)

@testset "Broadcast" begin
    for T in (Float32, Float64)
        v = rand(Vec{3, T})
        A = rand(Mat{2, 3, T})
        v_array = Array(v)
        A_array = Array(A)
        μ = T(3)
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
        @test (@inferred(v .+ Ref(one(T))))::Vec{3, T} ≈ map(y -> y + one(T), v)
        @test (@inferred(_broadcast_dotted_scalar(v, μ)))::Vec{3, T} ≈ map(y -> 2μ * y, v)
        @test (@inferred(_broadcast_dotted_nested_scalar(v, μ)))::Vec{3, T} ≈ map(y -> (2μ + one(μ)) * y, v)
        @test (@inferred(_broadcast_dotted_function_scalar(v, μ)))::Vec{3, T} ≈ map(y -> sin(μ) * y, v)
        # with tuple
        @test (@inferred(v .+ (1,2,3)))::Vec{3, T} ≈ map(+, v, (1,2,3))
        @test (@inferred(v .- (1,2,3)))::Vec{3, T} ≈ map(-, v, (1,2,3))
        @test (@inferred(v .+ (one(T),)))::Vec{3, T} ≈ map(y -> y + one(T), v)
        @test (@inferred(v .* (true, true, false)))::Vec{3, T} ≈ map(*, v, (true, true, false))
        @test (@inferred(_broadcast_dotted_scalar_tuple(v, μ)))::Vec{3, T} ≈ map((y, m) -> 2μ * y * m, v, (true, true, false))
        @test_throws DimensionMismatch v .+ (one(T), one(T))
        # others
        @test (@inferred(broadcast(sqrt, v)))::Vec{3, T} ≈ map(sqrt, v)
        @test (@inferred(broadcast(sqrt, A)))::Mat{2, 3, T} ≈ map(sqrt, A)
    end
end
