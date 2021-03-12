@testset "Quaternion" begin
    for T in (Float32, Float64)
        # basic constructors
        @test (@inferred Quaternion{T}((1,2,3,4)))::Quaternion{T} |> Tuple === map(T, (1,2,3,4))
        @test (@inferred Quaternion((1,2,3,T(4))))::Quaternion{T} |> Tuple === map(T, (1,2,3,4))
        @test (@inferred Quaternion{T}(1,2,3,4))::Quaternion{T} |> Tuple === map(T, (1,2,3,4))
        @test (@inferred Quaternion(1,2,3,T(4)))::Quaternion{T} |> Tuple === map(T, (1,2,3,4))

        # quaternion
        q = (@inferred quaternion(T(π/4), Vec{3, T}(1,2,3)))::Quaternion{T}
        @test norm(q) ≈ 1
        @test q/q ≈ 1
        q = (@inferred quaternion(T(π/4), Vec{3, T}(1,2,3), normalize = false))::Quaternion{T}
        @test !(norm(q) ≈ 1)
        @test q/q ≈ 1
        @test q ≈ (@inferred quaternion(T(45), Vec{3, T}(1,2,3), normalize = false, degree = true))::Quaternion{T}

        q = quaternion(rand(T), rand(Vec{3, T}))
        p = quaternion(rand(T), rand(Vec{3, T}), normalize = false)

        # math operations
        @test (@inferred +q)::Quaternion{T} === q
        @test (@inferred +p)::Quaternion{T} === p
        @test (@inferred -q)::Quaternion{T} == -1 * q
        @test (@inferred -p)::Quaternion{T} == -1 * p
        @test (@inferred q + p)::Quaternion{T} == Quaternion(Vec(q) + Vec(p))
        @test (@inferred q - p)::Quaternion{T} == Quaternion(Vec(q) - Vec(p))
        @test (@inferred 2 * q)::Quaternion{T} == Quaternion(2 * Vec(q))
        @test (@inferred 2 * p)::Quaternion{T} == Quaternion(2 * Vec(p))
        @test (@inferred q * 2)::Quaternion{T} == Quaternion(Vec(q) * 2)
        @test (@inferred p * 2)::Quaternion{T} == Quaternion(Vec(p) * 2)
        @test (@inferred q / 2)::Quaternion{T} == Quaternion(Vec(q) / 2)
        @test (@inferred p / 2)::Quaternion{T} == Quaternion(Vec(p) / 2)
        @test (@inferred norm(q))::T ≈ (@inferred abs(q))::T
        @test (@inferred norm(p))::T ≈ (@inferred abs(p))::T
        @test (@inferred exp(log(q)))::Quaternion{T} ≈ q
        @test (@inferred exp(log(p)))::Quaternion{T} ≈ p

        # rotation
        v = rand(Vec{3, T})
        Rq = rotmat(q)
        Rp = rotmat(p)
        r = p * q
        @test (q * v / q).vector ≈ Rq ⋅ v
        @test (r * v / r).vector ≈ Rp ⋅ Rq ⋅ v
    end
end
