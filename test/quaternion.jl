@testset "Quaternion" begin
    for T in (Float32, Float64)
        # basic constructors
        @test (@inferred Quaternion{T}((1,2,3,4)))::Quaternion{T} |> Tuple === map(T, (1,2,3,4))
        @test (@inferred Quaternion{T}(1,2,3,4))::Quaternion{T} |> Tuple === map(T, (1,2,3,4))
        @test (@inferred Quaternion{T}(Vec(1,2,3,4)))::Quaternion{T} |> Tuple === map(T, (1,2,3,4))
        @test (@inferred Quaternion{T}(Vec(1,2,3)))::Quaternion{T} |> Tuple === map(T, (0,1,2,3))
        @test (@inferred Quaternion{T}(4, Vec(1,2,3)))::Quaternion{T} |> Tuple === map(T, (4,1,2,3))
        @test (@inferred Quaternion{T}(4))::Quaternion{T} |> Tuple === map(T, (4,0,0,0))
        @test (@inferred Quaternion((T(1),2,3,4)))::Quaternion{T} |> Tuple === map(T, (1,2,3,4))
        @test (@inferred Quaternion(T(1),2,3,4))::Quaternion{T} |> Tuple === map(T, (1,2,3,4))
        @test (@inferred Quaternion(Vec(T(1),2,3,4)))::Quaternion{T} |> Tuple === map(T, (1,2,3,4))
        @test (@inferred Quaternion(Vec(T(1),2,3)))::Quaternion{T} |> Tuple === map(T, (0,1,2,3))
        @test (@inferred Quaternion(T(4), Vec(1,2,3)))::Quaternion{T} |> Tuple === map(T, (4,1,2,3))
        @test (@inferred Quaternion(T(4)))::Quaternion{T} |> Tuple === map(T, (4,0,0,0))
        @test_throws MethodError Quaternion{T}(Vec(1,2))
        @test_throws MethodError Quaternion{T}(Vec(1))
        @test_throws MethodError Quaternion{T}(4, Vec(1,2))
        @test_throws MethodError Quaternion{T}(4, Vec(1))
        @test_throws MethodError Quaternion(Vec(T(1),2))
        @test_throws MethodError Quaternion(Vec(T(1)))
        @test_throws MethodError Quaternion(T(4), Vec(1,2))
        @test_throws MethodError Quaternion(T(4), Vec(1))
        @test_throws ArgumentError Quaternion(1 + 2im)

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
        n3 = normalize(Vec{3, T}(1,2,3))
        @test_throws MethodError quaternion(T(π/4), Vec{2, T}(1,2))
        @test_throws MethodError quaternion(T, π/4, Vec{2, T}(1,2))
        q = (@inferred quaternion(T(π/4), n3))::Quaternion{T}
        q = (@inferred quaternion(T, π/4, n3))::Quaternion{T}
        @test length(q) == 4
        @test size(q) == (4,)
        @test norm(q) ≈ 1
        @test q/q ≈ 1

        q = quaternion(rand(T), normalize(rand(Vec{3, T})))
        p = Quaternion(rand(T), rand(Vec{3, T}))

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

        # number interface
        @test (@inferred zero(Quaternion{T}))::Quaternion{T} === Quaternion{T}(0,0,0,0)
        @test (@inferred zero(q))::Quaternion{T} === Quaternion{T}(0,0,0,0)
        @test (@inferred one(Quaternion{T}))::Quaternion{T} === Quaternion{T}(1,0,0,0)
        @test (@inferred one(q))::Quaternion{T} === Quaternion{T}(1,0,0,0)
        @test (@inferred real(Quaternion{T})) === T
        @test (@inferred iszero(zero(q)))::Bool
        @test (@inferred isreal(Quaternion{T}(1,0,0,0)))::Bool
        @test !(@inferred isreal(q))::Bool
        @test (@inferred isinf(Quaternion{T}(Inf,0,0,0)))::Bool
        @test (@inferred isnan(Quaternion{T}(NaN,0,0,0)))::Bool

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
        @test_throws MethodError Quaternion{T}(1, 2, 3)
        @test_throws MethodError (1 + 2im) * q
        @test_throws MethodError q * (1 + 2im)
        @test_throws MethodError q / (1 + 2im)

        # angleaxis
        @test (quaternion((@inferred angleaxis(q))::Tuple{T, Vec{3, T}}...)) ≈ q
        @test quaternion((@inferred angleaxis(-q))::Tuple{T, Vec{3, T}}...) ≈ q
        @test (@inferred angleaxis(Quaternion{T}(1)))::Tuple{T, Vec{3, T}} === (zero(T), Vec(one(T), zero(T), zero(T)))
        @test (@inferred angleaxis(Quaternion{T}(-1)))::Tuple{T, Vec{3, T}} === (zero(T), Vec(one(T), zero(T), zero(T)))
        @test_throws DomainError angleaxis(Quaternion{T}(0))

        a = rand(T)
        @test (@inferred exp(log(q)))::Quaternion{T} ≈ q
        @test (@inferred exp(log(p)))::Quaternion{T} ≈ p
        @test (@inferred exp(a + q))::Quaternion{T} ≈ exp(a) * exp(q)
        @test (@inferred exp(a + p))::Quaternion{T} ≈ exp(a) * exp(p)
        @test (@inferred log(a * q))::Quaternion{T} ≈ log(a) + log(q)
        @test (@inferred log(a * p))::Quaternion{T} ≈ log(a) + log(p)
        @test (@inferred exp(Quaternion{T}(1,0,0,0)))::Quaternion{T} ≈ exp(1)
        @test (@inferred log(Quaternion{T}(2)))::Quaternion{T} ≈ Quaternion{T}(log(T(2)), 0, 0, 0)
        @test (@inferred log(Quaternion{T}(-2)))::Quaternion{T} ≈ Quaternion{T}(log(T(2)), T(π), 0, 0)
        logzero = (@inferred log(Quaternion{T}(0)))::Quaternion{T}
        @test logzero.scalar == T(-Inf)
        @test logzero.vector === zero(Vec{3, T})
        @test (@inferred exp(log(Quaternion{T}(-2))))::Quaternion{T} ≈ Quaternion{T}(-2)
        @test (@inferred sqrt(Quaternion{T}(4)))::Quaternion{T} ≈ Quaternion{T}(2)
        @test (@inferred sqrt(Quaternion{T}(-4)))::Quaternion{T} ≈ Quaternion{T}(0,2,0,0)
        @test (@inferred sqrt(q))::Quaternion{T} * sqrt(q) ≈ q
        @test (@inferred sqrt(p))::Quaternion{T} * sqrt(p) ≈ p
        qneg = Quaternion{T}(-1, 2, -3, 4)
        @test (@inferred exp(log(qneg)))::Quaternion{T} ≈ qneg
        @test (@inferred sqrt(qneg))::Quaternion{T} * sqrt(qneg) ≈ qneg

        qlarge = Quaternion{T}(floatmax(T) / 4, floatmax(T) / 4, floatmax(T) / 4, floatmax(T) / 4)
        @test (@inferred abs(qlarge))::T ≈ floatmax(T) / 2
        invlarge = (@inferred inv(Quaternion{T}(floatmax(T) / 4)))::Quaternion{T}
        @test isfinite(invlarge.scalar)
        @test invlarge ≈ Quaternion{T}(T(4) / floatmax(T))
        @test (@inferred inv(qlarge))::Quaternion{T} * qlarge ≈ one(qlarge)
        @test (@inferred inv(Quaternion{T}(4floatmin(T))))::Quaternion{T} ≈ Quaternion{T}(inv(4floatmin(T)))

        Rq = rotmat(q)
        Rp = rotmat(p)
        r = p * q

        # check multiplications
        x = rand(Vec{3, T})
        @test (q * x / q).vector ≈ Rq * x
        @test (p * x / p).vector ≈ Rp * x
        @test (r * x / r).vector ≈ Rp * Rq * x
        @test (q * x * inv(q)).vector ≈ Rq * x
        @test (p * x * inv(p)).vector ≈ Rp * x
        @test (r * x * inv(r)).vector ≈ Rp * Rq * x
        # inverse of rotation
        @test (inv(q) * x * q).vector ≈ inv(Rq) * x
        # check order of multiplications
        @test ((q * x) / q).vector ≈ Rq * x
        @test (q * (x / q)).vector ≈ Rq * x
        # rotate
        @test rotate(x, q) ≈ rotate(x, Rq)
        @test rotate(x, p) ≈ rotate(x, Rp)
        @test rotate(x, r) ≈ rotate(x, Rp * Rq)
        @test rotate(x, inv(q)) ≈ rotate(x, inv(Rq))
        @test rotate(x, inv(p)) ≈ rotate(x, inv(Rp))
        @test rotate(x, inv(r)) ≈ rotate(x, inv(Rp * Rq))
        @test_throws MethodError rotate(rand(Vec{2, T}), q)
        @test_throws MethodError q * rand(Vec{2, T})
        @test_throws MethodError rand(Vec{2, T}) / q
        # test with rotmat(θ, n)
        θ = rand(T)
        n = normalize(rand(Vec{3, T}))
        @test rotmat(θ, n) ≈ rotmat(quaternion(θ, n))
    end
end
