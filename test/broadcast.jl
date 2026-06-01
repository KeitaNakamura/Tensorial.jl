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

    @testset "copyto! to AbstractArray" begin
        for T in (Float32, Float64)
            A = zeros(T, 3, 3)
            A_sarray = zeros(T, 3, 3)
            I = 1:2
            J = 1:2
            X = Tensor{Tuple{2, 2}, T}((1, 2, 3, 4))

            copyto!(view(A, I, J), X)
            copyto!(view(A_sarray, I, J), SArray(X))
            @test A == A_sarray

            fill!(A, zero(T))
            fill!(A_sarray, zero(T))
            A[I, J] .= X
            A_sarray[I, J] .= SArray(X)
            @test A == A_sarray

            fill!(A, zero(T))
            fill!(A_sarray, zero(T))
            A[I, J] .= X .+ X
            A_sarray[I, J] .= SArray(X) .+ SArray(X)
            @test A == A_sarray

            fill!(A, zero(T))
            fill!(A_sarray, zero(T))
            A[I, J] .= X .* T(3)
            A_sarray[I, J] .= SArray(X) .* T(3)
            @test A == A_sarray

            S = SymmetricSecondOrderTensor{2, T}((1, 2, 3))
            fill!(A, zero(T))
            fill!(A_sarray, zero(T))
            A[I, J] .= S
            A_sarray[I, J] .= SArray(S)
            @test A == A_sarray

            C = SymmetricFourthOrderTensor{2, T}(ntuple(i -> T(i), 9))
            B = zeros(T, 2, 2, 2, 2)
            B_sarray = similar(B)
            B .= C
            B_sarray .= SArray(C)
            @test B == B_sarray
        end
    end
end
