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
        for (SymTypeX, SymTypeY, SymTypeZ) in ((@Symmetry{3,3}, @Symmetry{3,3,3}, @Symmetry{3,3}),
                                               (@Skew{3,3}, @Skew{3,3,3}, @Skew{3,3}))
            for T in (Float32, Float64)
                x = rand(Tensor{Tuple{3,SymTypeX}, T})
                y = rand(Tensor{Tuple{SymTypeY}, T})
                # single contraction
                ## case 1
                z = (@inferred contraction(x, y, Val(1)))::Tensor{Tuple{3,3,SymTypeZ}, T}
                X = Array(x);
                Y = Array(y);
                Z = zeros(T, 3,3,3,3)
                for i in axes(X,1), j in axes(X,2), k in axes(X,3), l in axes(Y,2), m in axes(Y,3)
                    Z[i,j,l,m] += X[i,j,k] * Y[k,l,m]
                end
                @test Array(z) ≈ Z
                ## case 2
                v = rand(Vec{3, T})
                z = (@inferred contraction(y, v, Val(1)))::Tensor{Tuple{SymTypeX}, T}
                V = Array(v);
                Z = zeros(T, 3,3)
                for i in axes(X,1), j in axes(X,2), k in axes(X,3)
                    Z[i,j] += Y[i,j,k] * V[k]
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
                z = (@inferred contraction(x, y, Val(0)))::Tensor{Tuple{3,SymTypeX,SymTypeY}, T}
                X = Array(x);
                Y = Array(y);
                Z = zeros(T, 3,3,3,3,3,3)
                for i in axes(X,1), j in axes(X,2), k in axes(X,3), l in axes(Y,1), m in axes(Y,2), n in axes(Y,3)
                    Z[i,j,k,l,m,n] = X[i,j,k] * Y[l,m,n]
                end
                @test Array(z) ≈ Z
            end
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
    @testset "skew" begin
        for T in (Float32, Float64), dim in 1:4
            x = rand(SecondOrderTensor{dim, T})
            @test (@inferred skew(x))::SkewSymmetricSecondOrderTensor{dim, T} ≈ (x - x') / 2
            @test (@inferred symmetric(x) + skew(x))::typeof(x) ≈ x
            @test (@inferred transpose(skew(x)))::SkewSymmetricSecondOrderTensor{dim, T} == -skew(x)
            x = rand(SymmetricSecondOrderTensor{dim, T})
            @test (@inferred skew(x))::SkewSymmetricSecondOrderTensor{dim, T} ≈ zero(x)
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
            fm5, fm4, fm3, fm2, fm1, f0, fp1, fp2, fp3, fp4, fp5 = x -> x^-5, x -> x^-4, x -> x^-3, x -> x^-2, x -> x^-1, x -> x^0, x -> x^1, x -> x^2, x -> x^3, x -> x^4, x -> x^5
            for t in (x, y)
                @test (@inferred fm5(t))::typeof(t) ≈ inv(t) ⋅ inv(t) ⋅ inv(t) ⋅ inv(t) ⋅ inv(t)
                @test (@inferred fm4(t))::typeof(t) ≈ inv(t) ⋅ inv(t) ⋅ inv(t) ⋅ inv(t)
                @test (@inferred fm3(t))::typeof(t) ≈ inv(t) ⋅ inv(t) ⋅ inv(t)
                @test (@inferred fm2(t))::typeof(t) ≈ inv(t) ⋅ inv(t)
                @test (@inferred fm1(t))::typeof(t) == inv(t)
                @test (@inferred f0(t))::typeof(t) == one(t)
                @test (@inferred fp1(t))::typeof(t) == t
                @test (@inferred fp2(t))::typeof(t) ≈ t ⋅ t
                @test (@inferred fp3(t))::typeof(t) ≈ t ⋅ t ⋅ t
                @test (@inferred fp4(t))::typeof(t) ≈ t ⋅ t ⋅ t ⋅ t
                @test (@inferred fp5(t))::typeof(t) ≈ t ⋅ t ⋅ t ⋅ t ⋅ t
            end
            z = rand(SkewSymmetricSecondOrderTensor{dim, T})
            @test (@inferred fp1(z))::typeof(z) == z
            @test (@inferred fp2(z))::typeof(y) ≈ z ⋅ z
            @test (fp3(z))::typeof(z) ≈ z ⋅ z ⋅ z
            @test (fp4(z))::typeof(y) ≈ z ⋅ z ⋅ z ⋅ z
            @test (fp5(z))::typeof(z) ≈ z ⋅ z ⋅ z ⋅ z ⋅ z
        end
    end
    @testset "rotmat" begin
        for T in (Float32, Float64)
            α = deg2rad(T(10))
            β = deg2rad(T(20))
            γ = deg2rad(T(30))
            for (XYZ, zyx) in ((:XZX, :xzx), (:XYX, :xyx), (:YXY, :yxy), (:YZY, :yzy), (:ZYZ, :zyz), (:ZXZ, :zxz),
                               (:XZY, :yzx), (:XYZ, :zyx), (:YXZ, :zxy), (:YZX, :xzy), (:ZYX, :xyz), (:ZXY, :yxz))
                @test (@inferred rotmat(Vec(α,β,γ), sequence = XYZ))::Mat{3,3,T} ≈ (@inferred rotmat(Vec(γ,β,α), sequence = zyx))::Mat{3,3,T}
                @test (@inferred rotmat(Vec(α,β,γ), sequence = XYZ))::Mat{3,3,T} ≈ (@inferred rotmat(Vec(rad2deg(α),rad2deg(β),rad2deg(γ)), sequence = XYZ, degree = true))::Mat{3,3,T}
                @test (@inferred rotmat(Vec(α,β,γ), sequence = zyx))::Mat{3,3,T} ≈ (@inferred rotmat(Vec(rad2deg(α),rad2deg(β),rad2deg(γ)), sequence = zyx, degree = true))::Mat{3,3,T}
            end
            # 2D rotmat
            @test (@inferred rotmat(α))::Mat{2,2,T} ≈ (@inferred rotmat(rad2deg(α), degree = true))
            @test (@inferred rotmat(α)) |> Array ≈ (@inferred rotmatx(α))[[2,3], [2,3]]
            @test (@inferred rotmat(α)) |> Array ≈ (@inferred rotmaty(α))[[3,1], [3,1]]
            @test (@inferred rotmat(α)) |> Array ≈ (@inferred rotmatz(α))[[1,2], [1,2]]
            # 3D rotmat
            @test (@inferred rotmatx(α))::Mat{3,3,T} ≈ (@inferred rotmatx(rad2deg(α), degree = true))::Mat{3,3,T}
            @test (@inferred rotmaty(α))::Mat{3,3,T} ≈ (@inferred rotmaty(rad2deg(α), degree = true))::Mat{3,3,T}
            @test (@inferred rotmatz(α))::Mat{3,3,T} ≈ (@inferred rotmatz(rad2deg(α), degree = true))::Mat{3,3,T}
            for dim in (2, 3)
                a = rand(Vec{dim, T}); a /= norm(a)
                b = rand(Vec{dim, T}); b /= norm(b)
                @test (@inferred rotmat(a => b))::Mat{dim, dim, T} ⋅ a ≈ b
            end
        end
        @test_throws Exception rotmat(Vec(1,0) => Vec(1,1)) # length of two vectors must be the same
    end
    @testset "eigen" begin
        for T in (Float32, Float64)
            for dim in (2, 3)
                x = rand(SymmetricSecondOrderTensor{dim, T})
                @test (@inferred eigvals(x)) |> Array ≈ eigvals(Array(x))
                @test (@inferred eigen(x)).values |> Array ≈ eigen(Array(x)).values
                @test (@inferred eigen(x)).vectors ≈ (@inferred eigvecs(x))
            end
        end
    end
end

@testset "UniformScaling" begin
    for T in (Float32, Float64)
        for SquareTensorType in (Tensor{Tuple{3, 3}, T}, Tensor{Tuple{@Symmetry{3, 3}}, T})
            x = rand(SquareTensorType)
            @test (@inferred x + I)::SquareTensorType == x + one(x)
            @test (@inferred x - I)::SquareTensorType == x - one(x)
            @test (@inferred I + x)::SquareTensorType == one(x) + x
            @test (@inferred I - x)::SquareTensorType == one(x) - x
            y = rand(Mat{3, 4, T})
            v = rand(Vec{3, T})
            @test (@inferred x ⋅ I)::SquareTensorType == x ⋅ one(x)
            @test (@inferred I ⋅ x)::SquareTensorType == one(x) ⋅ x
            @test (@inferred y ⋅ I)::Mat{3, 4, T} == y ⋅ one(Mat{4, 4})
            @test (@inferred I ⋅ y)::Mat{3, 4, T} == one(Mat{3, 3}) ⋅ y
            @test (@inferred v ⋅ I)::Vec{3, T} == v ⋅ one(x)
            @test (@inferred I ⋅ v)::Vec{3, T} == one(x) ⋅ v
            @test (@inferred I ⊡ x)::T == one(x) ⊡ x
            @test (@inferred x ⊡ I)::T == x ⊡ one(x)
            # wrong input
            @test_throws Exception x * I
            @test_throws Exception y * I
            @test_throws Exception v * I
            @test_throws Exception I * x
            @test_throws Exception I * y
            @test_throws Exception I * v
        end
    end
end
