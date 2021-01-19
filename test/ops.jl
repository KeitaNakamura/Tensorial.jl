@testset "Basic operations" begin
    for T in (Float32, Float64)
        x = rand(Tensor{Tuple{2,          3,3, 2,@Symmetry{2,2}}, T})
        y = rand(Tensor{Tuple{2,@Symmetry{3,3},@Symmetry{2,2,2}}, T})
        a = 2
        @test size(x) == size(y)
        @test (@inferred x + y)::Tensor{Tuple{2,3,3,2,@Symmetry{2,2}}, T} == Array(x) + Array(y)
        @test (@inferred x - y)::Tensor{Tuple{2,3,3,2,@Symmetry{2,2}}, T} == Array(x) - Array(y)
        @test (@inferred a * x)::typeof(x) == a * Array(x)
        @test (@inferred a * y)::typeof(y) == a * Array(y)
        @test (@inferred x * a)::typeof(x) == a * Array(x)
        @test (@inferred y * a)::typeof(y) == a * Array(y)
        @test (@inferred x / a)::typeof(x) == Array(x) / a
        @test (@inferred y / a)::typeof(y) == Array(y) / a
        @test (@inferred +x)::typeof(x) == x
        @test (@inferred +y)::typeof(y) == y
        @test (@inferred -x)::typeof(x) == -Array(x)
        @test (@inferred -y)::typeof(y) == -Array(y)
        # check eltype promotion
        x = Vec(1, 0)
        y = Vec(T(1), 0)
        a = T(0.5)
        @test (@inferred x + y)::Vec{2, T} == Array(x) + Array(y)
        @test (@inferred x - y)::Vec{2, T} == Array(x) - Array(y)
        @test (@inferred a * x)::Vec{2, T} == a * Array(x)
        @test (@inferred x * a)::Vec{2, T} == Array(x) * a
        @test (@inferred x / a)::Vec{2, T} == Array(x) / a
        # bad operations
        @test_throws Exception x * y
        @test_throws Exception y * x
    end
end

@testset "Tensor operations" begin
    @testset "contraction" begin
        for T in (Float32, Float64)
            x = rand(Tensor{Tuple{3,@Symmetry{3,3}}, T})
            y = rand(Tensor{Tuple{@Symmetry{3,3,3}}, T})
            # single contraction
            z = (@inferred contraction(x, y, Val(1)))::Tensor{Tuple{3,3,@Symmetry{3,3}}, T}
            X = Array(x);
            Y = Array(y);
            Z = zeros(T, 3,3,3,3)
            for i in axes(X,1), j in axes(X,2), k in axes(X,3), l in axes(Y,2), m in axes(Y,3)
                Z[i,j,l,m] += X[i,j,k] * Y[k,l,m]
            end
            @test Array(z) ≈ Z
            # double contraction
            z = (@inferred contraction(x, y, Val(2)))::Tensor{Tuple{3,3}, T}
            X = Array(x);
            Y = Array(y);
            Z = zeros(T, 3,3)
            for i in axes(X,1), j in axes(X,2), k in axes(X,3), l in axes(Y,3)
                Z[i,l] += X[i,j,k] * Y[j,k,l]
            end
            @test Array(z) ≈ Z
            # triple contraction
            z = (@inferred contraction(x, y, Val(3)))::T
            X = Array(x);
            Y = Array(y);
            Z = zero(T)
            for i in axes(X,1), j in axes(X,2), k in axes(X,3)
                Z += X[i,j,k] * Y[i,j,k]
            end
            @test z ≈ Z
            # zero contraction (otimes)
            z = (@inferred contraction(x, y, Val(0)))::Tensor{Tuple{3,@Symmetry{3,3},@Symmetry{3,3,3}}, T}
            X = Array(x);
            Y = Array(y);
            Z = zeros(T, 3,3,3,3,3,3)
            for i in axes(X,1), j in axes(X,2), k in axes(X,3), l in axes(Y,1), m in axes(Y,2), n in axes(Y,3)
                Z[i,j,k,l,m,n] = X[i,j,k] * Y[l,m,n]
            end
            @test Array(z) ≈ Z
        end
    end
    @testset "otimes/dot/norm" begin
        for T in (Float32, Float64)
            # square
            x = rand(Vec{3, T})
            y = rand(Vec{3, T})
            z = (@inferred x ⊗ y)::Tensor{Tuple{3,3}, T}
            @test Array(z) ≈ Array(x) * Array(y)'
            @test (@inferred x ⋅ y) ≈ Array(x)' * Array(y)
            @test (@inferred norm(x)) ≈ norm(Array(x))
            @test (@inferred norm(z)) ≈ norm(Array(z))
            # nonsquare
            x = rand(Vec{3, T})
            y = rand(Vec{2, T})
            z = (@inferred x ⊗ y)::Tensor{Tuple{3,2}, T}
            @test Array(z) ≈ Array(x) * Array(y)'
            @test (@inferred norm(z)) ≈ norm(Array(z))
        end
    end
    @testset "dotdot" begin
        for T in (Float32, Float64)
            x = rand(Vec{3, T})
            y = rand(Vec{3, T})
            S = rand(SymmetricFourthOrderTensor{3, T})
            A = FourthOrderTensor{3, T}((i,j,k,l) -> S[i,k,j,l])
            @test (@inferred dotdot(x, S, y))::Tensor{Tuple{3,3}, T} ≈ A ⊡ (x ⊗ y)
        end
    end
    @testset "tr/mean/vol/dev" begin
        for T in (Float32, Float64)
            x = rand(SecondOrderTensor{3, T})
            y = rand(SymmetricSecondOrderTensor{3, T})
            # tr/mean
            @test (@inferred tr(x))::T ≈ tr(Array(x))
            @test (@inferred tr(y))::T ≈ tr(Array(y))
            @test (@inferred mean(x))::T ≈ tr(Array(x)) / 3
            @test (@inferred mean(y))::T ≈ tr(Array(y)) / 3
            # vol/dev
            @test (@inferred vol(x))::typeof(x) + (@inferred dev(x))::typeof(x) ≈ x
            @test (@inferred vol(y))::typeof(y) + (@inferred dev(y))::typeof(y) ≈ y
            @test (@inferred vol(x))::typeof(x) ⊡ (@inferred dev(x))::typeof(x) ≈ zero(T)  atol = sqrt(eps(T))
            @test (@inferred vol(y))::typeof(y) ⊡ (@inferred dev(y))::typeof(y) ≈ zero(T)  atol = sqrt(eps(T))
        end
    end
    @testset "transpose/adjoint" begin
        for T in (Float32, Float64)
            x = rand(SecondOrderTensor{3, T})
            y = rand(SymmetricSecondOrderTensor{3, T})
            xᵀ = (@inferred transpose(x))::typeof(x)
            @test Array(x') == Array(xᵀ) == Array(x)'
            @test (@inferred transpose(y))::typeof(y) == y' == y
        end
    end
    @testset "symmetric" begin
        for T in (Float32, Float64), dim in 1:4
            x = rand(SecondOrderTensor{dim, T})
            y = rand(SymmetricSecondOrderTensor{dim, T})
            @test (@inferred symmetric(x))::SymmetricSecondOrderTensor{dim, T} ≈ (x + x')/2
            @test (@inferred symmetric(y))::typeof(y) == y
        end
    end
    @testset "det/inv" begin
        for T in (Float32, Float64), dim in 1:4
            Random.seed!(1234)
            # second order
            x = rand(SecondOrderTensor{dim, T})
            y = rand(SymmetricSecondOrderTensor{dim, T})
            @test (@inferred det(x))::T ≈ det(Array(x))
            @test (@inferred det(y))::T ≈ det(Array(y))
            @test (@inferred inv(x))::typeof(x) |> Array ≈ inv(Array(x))
            @test (@inferred inv(y))::typeof(y) |> Array ≈ inv(Array(y))
            # fourth order
            dim == 4 && continue
            x = rand(FourthOrderTensor{dim, T})
            y = rand(SymmetricFourthOrderTensor{dim, T})
            @test (@inferred inv(x))::typeof(x) ⊡ x ≈ one(x)
            @test (@inferred inv(y))::typeof(y) ⊡ y ≈ one(y)
        end
    end
    @testset "cross" begin
        for T in (Float32, Float64), dim in 1:3
            x = rand(Vec{dim, T})
            y = rand(Vec{dim, T})
            @test (@inferred x × x)::Vec{3, T} ≈ zero(Vec{3, T})
            @test x × y ≈ -y × x
            if dim == 2
                a = Vec{2, T}(1,0)
                b = Vec{2, T}(0,1)
                @test (@inferred a × b)::Vec{3, T} ≈ Vec{3, T}(0,0,1)
            end
            if dim == 3
                a = Vec{3, T}(1,0,0)
                b = Vec{3, T}(0,1,0)
                @test (@inferred a × b)::Vec{3, T} ≈ Vec{3, T}(0,0,1)
            end
        end
    end
    @testset "pow" begin
        for T in (Float32, Float64), dim in 1:3
            x = rand(SecondOrderTensor{dim, T})
            y = rand(SymmetricSecondOrderTensor{dim, T})
            fm3, fm2, fm1, f0, fp1, fp2, fp3 = x -> x^-3, x -> x^-2, x -> x^-1, x -> x^0, x -> x^1, x -> x^2, x -> x^3
            @test (@inferred fm3(x))::typeof(x) ≈ inv(x) ⋅ inv(x) ⋅ inv(x)
            @test (@inferred fm2(x))::typeof(x) ≈ inv(x) ⋅ inv(x)
            @test (@inferred fm1(x))::typeof(x) == inv(x)
            @test (@inferred f0(x))::typeof(x) == one(x)
            @test (@inferred fp1(x))::typeof(x) == x
            @test (@inferred fp2(x))::typeof(x) ≈ x ⋅ x
            @test (@inferred fp3(x))::typeof(x) ≈ x ⋅ x ⋅ x
        end
    end
end
