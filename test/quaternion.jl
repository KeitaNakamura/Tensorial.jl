ToVec3(x::Vec{3}) = x
ToVec3(x::Vec{2}) = Vec(x[1], x[2], 0)

@testset "Quaternion" begin
    for T in (Float32, Float64)
        # basic constructors
        @test (@inferred Quaternion{T}((1,2,3,4)))::Quaternion{T} |> Tuple === map(T, (1,2,3,4))
        @test (@inferred Quaternion((1,2,3,T(4))))::Quaternion{T} |> Tuple === map(T, (1,2,3,4))
        @test (@inferred Quaternion{T}(1,2,3,4))::Quaternion{T} |> Tuple === map(T, (1,2,3,4))
        @test (@inferred Quaternion(1,2,3,T(4)))::Quaternion{T} |> Tuple === map(T, (1,2,3,4))
        @test (@inferred Quaternion(Vec(1,2,T(3))))::Quaternion{T} |> Tuple === map(T, (0,1,2,3))
        @test (@inferred Quaternion{T}(Vec(1,2,3)))::Quaternion{T} |> Tuple === map(T, (0,1,2,3))

        # properties
        q = Quaternion{T}(1,2,3,4)
        @test propertynames(q) == (:scalar, :vector, :data)
        get_scalar = q -> q.scalar
        get_vector = q -> q.vector
        get_data = q -> q.data
        @test (@inferred get_scalar(q))::T == T(1)
        @test (@inferred get_vector(q))::Vec{3, T} == Vec{3, T}(2,3,4)
        @test (@inferred get_data(q))::NTuple{4, T} == map(T, (1,2,3,4))

        # quaternion
        q = (@inferred quaternion(T(π/4), Vec{3, T}(1,2,3)))::Quaternion{T}
        q = (@inferred quaternion(T, π/4, Vec(1,2,3)))::Quaternion{T}
        @test length(q) == 4
        @test size(q) == (4,)
        @test norm(q) ≈ 1
        @test q/q ≈ 1
        q = (@inferred quaternion(T(π/4), Vec{3, T}(1,2,3), normalize = false))::Quaternion{T}
        q = (@inferred quaternion(T, π/4, Vec(1,2,3), normalize = false))::Quaternion{T}
        @test !(norm(q) ≈ 1)
        @test q/q ≈ 1
        @test q ≈ (@inferred quaternion(T(45), Vec{3, T}(1,2,3), normalize = false, degree = true))::Quaternion{T}
        @test q ≈ (@inferred quaternion(T, 45, Vec(1,2,3), normalize = false, degree = true))::Quaternion{T}

        q = quaternion(rand(T), rand(Vec{3, T}))
        p = quaternion(rand(T), rand(Vec{3, T}), normalize = false)

        # conversion
        @test (@inferred convert(Quaternion{T}, q))::Quaternion{T} == q
        @test (@inferred convert(Quaternion{T}, 3))::Quaternion{T} == Quaternion{T}(3,0,0,0)

        # promotion
        @test (@inferred promote_rule(Quaternion{T}, T)) == Quaternion{T}
        @test (@inferred promote_rule(Quaternion{T}, Int)) == Quaternion{T}
        @test (@inferred promote_rule(Quaternion{T}, Quaternion{T})) == Quaternion{T}
        @test (@inferred promote_rule(Quaternion{T}, Quaternion{Int})) == Quaternion{T}
        @test ((@inferred promote(q, T(3)))::NTuple{2, Quaternion{T}})[2] == Quaternion(3.0,0,0,0)
        @test ((@inferred promote(q, 3))::NTuple{2, Quaternion{T}})[2] == Quaternion(3.0,0,0,0)

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
        @test (@inferred exp(Quaternion{T}(1,0,0,0)))::Quaternion{T} ≈ exp(1)

        Rq = rotmat(q)
        Rp = rotmat(p)
        r = p * q

        for dim in (2, 3)
            dim == 2 && continue
            # check multiplications
            v = rand(Vec{dim, T})
            v3 = ToVec3(v)
            @test (q * v / q).vector ≈ Rq ⋅ v3
            @test (p * v / p).vector ≈ Rp ⋅ v3
            @test (r * v / r).vector ≈ Rp ⋅ Rq ⋅ v3
            @test (q * v * inv(q)).vector ≈ Rq ⋅ v3
            @test (p * v * inv(p)).vector ≈ Rp ⋅ v3
            @test (r * v * inv(r)).vector ≈ Rp ⋅ Rq ⋅ v3
            # inverse of rotation
            @test (inv(q) * v * q).vector ≈ inv(Rq) ⋅ v3
            # check order of multiplications
            @test ((q * v) / q).vector ≈ Rq ⋅ v3
            @test (q * (v / q)).vector ≈ Rq ⋅ v3
            # rotate
            @test rotate(v, q) ≈ rotate(v, Rq)
            @test rotate(v, p) ≈ rotate(v, Rp)
            @test rotate(v, r) ≈ rotate(v, Rp ⋅ Rq)
            @test rotate(v, inv(q)) ≈ rotate(v, inv(Rq))
            @test rotate(v, inv(p)) ≈ rotate(v, inv(Rp))
            @test rotate(v, inv(r)) ≈ rotate(v, inv(Rp ⋅ Rq))
        end
        # test with rotmat(θ, n)
        θ = rand(T)
        n = rand(Vec{3, T})
        @test rotmat(θ, n) ≈ rotmat(quaternion(θ, n))
    end
end
