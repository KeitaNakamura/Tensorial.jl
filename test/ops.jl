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
        # with Array
        @test (@inferred x + Array(y))::Tensor{Tuple{size(x)...}, T} == Array(x) + Array(y)
        @test (@inferred Array(x) + y)::Tensor{Tuple{size(y)...}, T} == Array(x) + Array(y)
        @test (@inferred x - Array(y))::Tensor{Tuple{size(x)...}, T} == Array(x) - Array(y)
        @test (@inferred Array(x) - y)::Tensor{Tuple{size(y)...}, T} == Array(x) - Array(y)
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
            z = (@inferred contract(x, y, Val(1)))::Tensor{Tuple{3,3,@Symmetry{3,3}}, T}
            X = Array(x);
            Y = Array(y);
            Z = zeros(T, 3,3,3,3)
            for i in axes(X,1), j in axes(X,2), k in axes(X,3), l in axes(Y,2), m in axes(Y,3)
                Z[i,j,l,m] += X[i,j,k] * Y[k,l,m]
            end
            @test z ≈ Z
            @test_deprecated contraction(x, y, Val(1))
            # double contraction
            z = (@inferred contract(x, y, Val(2)))::Tensor{Tuple{3,3}, T}
            X = Array(x);
            Y = Array(y);
            Z = zeros(T, 3,3)
            for i in axes(X,1), j in axes(X,2), k in axes(X,3), l in axes(Y,3)
                Z[i,l] += X[i,j,k] * Y[j,k,l]
            end
            @test (@inferred Tensorial.contract2(x, y))::Tensor{Tuple{3,3}, T} ≈ z
            @test_deprecated double_contraction(x, y)
            @test z ≈ Z
            # triple contraction
            z = (@inferred contract(x, y, Val(3)))::T
            X = Array(x);
            Y = Array(y);
            Z = zero(T)
            for i in axes(X,1), j in axes(X,2), k in axes(X,3)
                Z += X[i,j,k] * Y[i,j,k]
            end
            @test (@inferred Tensorial.contract3(x, y))::T ≈ z
            @test z ≈ Z
            # zero contraction (otimes)
            z = (@inferred contract(x, y, Val(0)))::Tensor{Tuple{3,@Symmetry{3,3},@Symmetry{3,3,3}}, T}
            X = Array(x);
            Y = Array(y);
            Z = zeros(T, 3,3,3,3,3,3)
            for i in axes(X,1), j in axes(X,2), k in axes(X,3), l in axes(Y,1), m in axes(Y,2), n in axes(Y,3)
                Z[i,j,k,l,m,n] = X[i,j,k] * Y[l,m,n]
            end
            @test z ≈ Z
            @test (@inferred Tensorial.contract(x, 2, Val(0)))::typeof(x) ≈ 2x
            @test (@inferred Tensorial.contract(3, y, Val(0)))::typeof(y) ≈ 3y
            @test (@inferred Tensorial.contract(T(2), T(3), Val(0)))::T ≈ 6
            # dimension error
            A = rand(Mat{3,3,T})
            x = rand(Vec{3,T})
            @test_throws Exception A ⊡ x
        end
    end
    @testset "otimes/dot/norm/normalize" begin
        for T in (Float32, Float64)
            # square
            x = rand(Vec{3, T})
            y = rand(Vec{3, T})
            z = (@inferred x ⊗ y)::Tensor{Tuple{3,3}, T}
            @test z ≈ Array(x) * Array(y)'
            @test (@inferred x ⋅ y) ≈ Array(x)' * Array(y)
            @test (@inferred norm(x)) ≈ norm(Array(x))
            @test (@inferred norm(z)) ≈ norm(Array(z))
            @test (@inferred normalize(x))::typeof(x) ≈ normalize(Array(x))
            @test (@inferred normalize(z))::typeof(z) ≈ Array(z) / norm(Array(z))
            @test (@inferred ⊗(x, y, x, y))::Tensor{Tuple{3,3,3,3}, T} ≈ x ⊗ y ⊗ x ⊗ y
            @test (@inferred ⊗(x))::typeof(x) ≈ x
            @test (@inferred 2 ⊗ x)::typeof(x) ≈ 2x
            @test (@inferred y ⊗ 3)::typeof(y) ≈ 3y
            @test (@inferred T(2) ⊗ T(3))::T ≈ 6
            @test (@inferred x^⊗(0))::T == 1
            @test (@inferred x^⊗(1))::Tensor{Tuple{3}, T} ≈ x
            @test (@inferred x^⊗(2))::Tensor{Tuple{@Symmetry{3,3}}, T} ≈ x ⊗ x
            @test (@inferred x^⊗(3))::Tensor{Tuple{@Symmetry{3,3,3}}, T} ≈ x ⊗ x ⊗ x
            @test (@inferred x^⊗(4))::Tensor{Tuple{@Symmetry{3,3,3,3}}, T} ≈ x ⊗ x ⊗ x ⊗ x
            # nonsquare
            x = rand(Vec{3, T})
            y = rand(Vec{2, T})
            z = (@inferred x ⊗ y)::Tensor{Tuple{3,2}, T}
            @test z ≈ Array(x) * Array(y)'
            @test (@inferred norm(z)) ≈ norm(Array(z))
            @test (@inferred normalize(z))::typeof(z) ≈ Array(z) / norm(Array(z))
            @test (@inferred ⊗(x, y, x, y))::Tensor{Tuple{3,2,3,2}, T} ≈ x ⊗ y ⊗ x ⊗ y
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
    @testset "tr" begin
        for T in (Float32, Float64)
            x = rand(SecondOrderTensor{3, T})
            y = rand(SymmetricSecondOrderTensor{3, T})
            # tr
            @test (@inferred tr(x))::T ≈ tr(Array(x))
            @test (@inferred tr(y))::T ≈ tr(Array(y))
        end
    end
    @testset "transpose/adjoint" begin
        for T in (Float32, Float64)
            x = rand(SecondOrderTensor{3, T})
            y = rand(SymmetricSecondOrderTensor{3, T})
            xᵀ = (@inferred transpose(x))::typeof(x)
            @test x' == xᵀ == Array(x)'
            @test (@inferred transpose(y))::typeof(y) == y' == y
        end
    end
    @testset "symmetric" begin
        for T in (Float32, Float64), dim in 1:4
            x = rand(SecondOrderTensor{dim, T})
            y = rand(SymmetricSecondOrderTensor{dim, T})
            @test (@inferred symmetric(x))::SymmetricSecondOrderTensor{dim, T} ≈ (x + x')/2
            @test (@inferred symmetric(x, :U))::SymmetricSecondOrderTensor{dim, T} ≈ Symmetric(Array(x), :U)
            @test (@inferred symmetric(x, :L))::SymmetricSecondOrderTensor{dim, T} ≈ Symmetric(Array(x), :L)
            @test (@inferred symmetric(y))::typeof(y) == y
            @test (@inferred symmetric(y, :U))::typeof(y) == y
            @test (@inferred symmetric(y, :L))::typeof(y) == y
            @test_throws Exception symmetric(x, :X)
            @test_throws Exception symmetric(y, :X)
        end
    end
    @testset "minorsymmetric" begin
        for T in (Float32, Float64), dim in 1:4
            x = rand(FourthOrderTensor{dim, T})
            y = rand(SymmetricFourthOrderTensor{dim, T})
            @test (@inferred minorsymmetric(x))::SymmetricFourthOrderTensor{dim, T} ≈ @einsum (i,j,k,l) -> (x[i,j,k,l]+x[j,i,k,l]+x[i,j,l,k]+x[j,i,l,k])/4
            @test (@inferred minorsymmetric(y))::typeof(y) == y
        end
    end
    @testset "skew" begin
        for T in (Float32, Float64), dim in 1:4
            x = rand(SecondOrderTensor{dim, T})
            @test (@inferred skew(x))::SecondOrderTensor{dim, T} ≈ (x - x') / 2
            @test symmetric(x) + skew(x) ≈ x
            x = rand(SymmetricSecondOrderTensor{dim, T})
            @test (@inferred skew(x))::SecondOrderTensor{dim, T} ≈ zero(SecondOrderTensor{dim})
        end
        for T in (Float32, Float64)
            ω = rand(Vec{3, T})
            @test (@inferred skew(ω))::SecondOrderTensor{3, T} ≈ [ 0    -ω[3]  ω[2]
                                                                   ω[3]  0    -ω[1]
                                                                  -ω[2]  ω[1]  0]
            x = rand(Vec{3, T})
            @test skew(ω) ⋅ x ≈ ω × x
        end
    end
    @testset "det" begin
        for T in (Float32, Float64), dim in 1:4
            Random.seed!(1234)
            x = rand(SecondOrderTensor{dim, T})
            y = rand(SymmetricSecondOrderTensor{dim, T})
            @test (@inferred det(x))::T ≈ det(Array(x))
            @test (@inferred det(y))::T ≈ det(Array(y))
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
    @testset "rotmat" begin
        for T in (Float32, Float64)
            α = deg2rad(T(10))
            β = deg2rad(T(20))
            γ = deg2rad(T(30))
            for (XYZ, zyx) in ((:XZX, :xzx), (:XYX, :xyx), (:YXY, :yxy), (:YZY, :yzy), (:ZYZ, :zyz), (:ZXZ, :zxz),
                               (:XZY, :yzx), (:XYZ, :zyx), (:YXZ, :zxy), (:YZX, :xzy), (:ZYX, :xyz), (:ZXY, :yxz))
                @test (@inferred rotmat(Vec(α,β,γ), sequence = XYZ))::Mat{3,3,T} ≈ (@inferred rotmat(Vec(γ,β,α), sequence = zyx))::Mat{3,3,T}
            end
            # 2D/3D rotmat
            @test (@inferred rotmat(α)) ≈ (@inferred rotmatx(α))[[2,3], [2,3]]
            @test (@inferred rotmat(α)) ≈ (@inferred rotmaty(α))[[3,1], [3,1]]
            @test (@inferred rotmat(α)) ≈ (@inferred rotmatz(α))[[1,2], [1,2]]
            for dim in (2, 3)
                a = normalize(rand(Vec{dim, T}))
                b = normalize(rand(Vec{dim, T}))
                @test (@inferred rotmat(a => b))::Mat{dim, dim, T} ⋅ a ≈ b
            end
            @test (@inferred rotmat(T(π/3), Vec{3, T}(0,0,1)))::Mat{3,3,T} ⋅ Vec(1,0,0) ≈ [cos(π/3), sin(π/3), 0]
        end
        @test_throws Exception rotmat(Vec(1,0) => Vec(1,1)) # length of two vectors must be the same
    end
    @testset "rotate" begin
        for T in (Float32, Float64)
            for dim in 2:3
                a = normalize(rand(Vec{dim, T}))
                b = normalize(rand(Vec{dim, T}))
                R = rotmat(a => b)
                v = rand(Vec{dim, T})
                A = rand(SecondOrderTensor{dim, T})
                S = rand(SymmetricSecondOrderTensor{dim, T})
                @test (@inferred rotate(v, R))::Vec{dim, T} ≈ v ⋅ R
                @test (@inferred rotate(A, R))::SecondOrderTensor{dim, T} ≈ R ⋅ A ⋅ R'
                @test (@inferred rotate(S, R))::SymmetricSecondOrderTensor{dim, T} ≈ R ⋅ S ⋅ R'
            end
            # v in 2D, R in 3D
            for R in (rotmatx(T(π/4)), rotmaty(T(π/4)), rotmatz(T(π/4)))
                v = rand(Vec{2, T})
                @test (@inferred rotate(v, R))::Vec{3, T} ≈ rotate(v, quaternion(angleaxis(R)...))
            end
        end
    end
    @testset "angleaxis" begin
        Random.seed!(1234)
        for T in (Float32, Float64)
            a = normalize(rand(Vec{3, T}))
            b = normalize(rand(Vec{3, T}))
            R = rotmat(a => b)
            @test rotmat((@inferred angleaxis(R))::Tuple{T, Vec{3, T}}...) ≈ R
        end
    end
    @testset "exp/log" begin
        for T in (Float32, Float64)
            for dim in (2, 3)
                x = rand(SymmetricSecondOrderTensor{dim, T})
                y = exp(x)
                @test (@inferred exp(x))::SymmetricSecondOrderTensor{dim, T} ≈ exp(Array(x))
                @test (@inferred log(y))::SymmetricSecondOrderTensor{dim, T} ≈ x
            end
        end
    end
end

@testset "Call methods in StaticArrays" begin
    @testset "exp" begin
        # error for `Float32`
        # see https://github.com/JuliaArrays/StaticArrays.jl/issues/785
        for T in (Float64,)
            for dim in (2, 3)
                x = rand(SecondOrderTensor{dim, T})
                y = rand(SymmetricSecondOrderTensor{dim, T})
                @test (@inferred exp(x))::SecondOrderTensor{dim, T} ≈ exp(Array(x))
                @test (@inferred exp(y))::SymmetricSecondOrderTensor{dim, T} ≈ exp(Array(y))
            end
        end
    end
    @testset "diag/diagm" begin
        x = @Mat [1 2 3
                  4 5 6
                  7 8 9]
        @test @inferred(diag(x)) === Vec(1,5,9)
        @test @inferred(diag(x, Val(1))) === Vec(2,6)
        @test @inferred(diag(x, Val(2))) === Vec(3)
        @test @inferred(diag(x, Val(3))) === Vec{0,Int}()
        @test @inferred(diag(x, Val(-1))) === Vec(4,8)
        @test @inferred(diag(x, Val(-2))) === Vec(7)
        @test @inferred(diag(x, Val(-3))) === Vec{0,Int}()
        d = Vec(1,2,3)
        @test @inferred(diagm(d)) === @Mat [1 0 0
                                            0 2 0
                                            0 0 3]
        @test @inferred(diagm(Val(1)=>d, Val(-1)=>d)) === @Mat [0 1 0 0
                                                                1 0 2 0
                                                                0 2 0 3
                                                                0 0 3 0]
    end
    @testset "qr" begin
        Random.seed!(1234)
        At = rand(Mat{3,3})
        Q, R, p = @inferred qr(At)
        As = SArray(At)
        Fs = qr(As)
        @test Q == Fs.Q
        @test R == Fs.R
        @test p == Fs.p
    end
    @testset "lu" begin
        Random.seed!(1234)
        At = rand(Mat{3,3})
        L, U, p = @inferred lu(At)
        As = SArray(At)
        Fs = lu(As)
        @test L == Fs.L
        @test U == Fs.U
        @test p == Fs.p
    end
    @testset "eigen" begin
        Random.seed!(1234)
        x = rand(SymmetricSecondOrderTensor{3})
        @test (@inferred eigvals(x)) ≈ eigvals(Array(x))
        @test (@inferred eigen(x)).values ≈ eigen(Array(x)).values
        @test (@inferred eigen(x)).vectors ≈ (@inferred eigvecs(x))
    end
    @testset "svd" begin
        Random.seed!(1234)
        At = rand(Mat{3,3})
        Ft = @inferred svd(At)
        U, S, V = Ft
        As = SArray(At)
        Fs = svd(As)
        @test U == Fs.U
        @test S == Fs.S
        @test V == Fs.V
        @test Ft.Vt == Fs.Vt
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
