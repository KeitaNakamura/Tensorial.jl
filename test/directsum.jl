@testset "DirectSumArray" begin
    @testset "pack/unpack round-trip: nonsymmetric" begin
        A = SecondOrderTensor{2}((1.0, 2.0, 3.0, 4.0))
        s = 3.0
        x = A ⊕ s

        @test (@inferred unpack(x))::Tuple{typeof(A), Float64} == (A, s)
        @test unpack(x, 1)::typeof(A) == A
        @test unpack(x, 2)::Float64 == s

        @test Tensorial._pack(typeof(x).parameters[1], unpack(x)) == x
        @test flatview(x) == [1.0, 2.0, 3.0, 4.0, 3.0]
    end

    @testset "pack/unpack round-trip: symmetric with nonzero offdiag" begin
        A = SymmetricSecondOrderTensor{2}((1.0, 2.0, 4.0))
        s = 3.0
        x = A ⊕ s

        @test (@inferred unpack(x))::Tuple{typeof(A), Float64} == (A, s)
        @test unpack(x, 1)::typeof(A) == A
        @test unpack(x, 2)::Float64 == s

        @test Tensorial._pack(typeof(x).parameters[1], unpack(x)) == x
        @test flatview(x) == [1.0, sqrt(2) * 2.0, 4.0, 3.0]
    end

    @testset "pack overloads for DirectSumVector" begin
        A = SecondOrderTensor{2}((1.0, 2.0, 3.0, 4.0))
        B = SecondOrderTensor{2}((5.0, 6.0, 7.0, 8.0))
        s = 3.0
        t = 4.0

        x = A ⊕ s
        y = B ⊕ t

        z1 = pack(x, t)
        @test unpack(z1, 1)::typeof(A) == A
        @test unpack(z1, 2)::Float64 == s
        @test unpack(z1, 3)::Float64 == t

        z2 = pack(t, x)
        @test unpack(z2, 1)::Float64 == t
        @test unpack(z2, 2)::typeof(A) == A
        @test unpack(z2, 3)::Float64 == s

        z3 = pack(x, y)
        @test unpack(z3, 1)::typeof(A) == A
        @test unpack(z3, 2)::Float64 == s
        @test unpack(z3, 3)::typeof(B) == B
        @test unpack(z3, 4)::Float64 == t
    end

    @testset "pack of DirectSumArray: vector case" begin
        A = SecondOrderTensor{2}((1.0, 2.0, 3.0, 4.0))
        B = SecondOrderTensor{2}((5.0, 6.0, 7.0, 8.0))
        s = 3.0
        t = 4.0

        x = A ⊕ s
        y = B ⊕ t
        z = pack(x, y)

        @test size(z) == (4,)
        @test unpack(z, 1)::typeof(A) == A
        @test unpack(z, 2)::Float64 == s
        @test unpack(z, 3)::typeof(B) == B
        @test unpack(z, 4)::Float64 == t
    end

    @testset "pack of DirectSumArray: matrix case" begin
        A = SecondOrderTensor{2}((1.0, 2.0, 3.0, 4.0))
        s = 3.0
        x = A ⊕ s
        G1 = gradient(identity, x)

        B = SecondOrderTensor{2}((5.0, 6.0, 7.0, 8.0))
        t = 4.0
        y = B ⊕ t
        G2 = gradient(identity, y)

        Z = pack(G1, G2)

        @test size(Z) == (4, 4)

        @test unpack(Z, 1, 1)::FourthOrderTensor{2, Float64, 16} == one(unpack(G1, 1, 1))
        @test unpack(Z, 1, 2)::typeof(A) == zero(unpack(G1, 1, 2))
        @test unpack(Z, 2, 1)::typeof(A) == zero(unpack(G1, 2, 1))
        @test unpack(Z, 2, 2)::Float64 == 1.0

        @test unpack(Z, 3, 3)::FourthOrderTensor{2, Float64, 16} == one(unpack(G2, 1, 1))
        @test unpack(Z, 3, 4)::typeof(B) == zero(unpack(G2, 1, 2))
        @test unpack(Z, 4, 3)::typeof(B) == zero(unpack(G2, 2, 1))
        @test unpack(Z, 4, 4)::Float64 == 1.0

        @test unpack(Z, 1, 3)::FourthOrderTensor{2, Float64, 16} == zero(unpack(G1, 1, 1))
        @test unpack(Z, 1, 4)::typeof(A) == zero(unpack(G1, 1, 2))
        @test unpack(Z, 2, 3)::typeof(A) == zero(unpack(G1, 2, 1))
        @test unpack(Z, 2, 4)::Float64 == 0.0

        @test unpack(Z, 3, 1)::FourthOrderTensor{2, Float64, 16} == zero(unpack(G2, 1, 1))
        @test unpack(Z, 3, 2)::typeof(B) == zero(unpack(G2, 1, 2))
        @test unpack(Z, 4, 1)::typeof(B) == zero(unpack(G2, 2, 1))
        @test unpack(Z, 4, 2)::Float64 == 0.0
    end

    @testset "zero" begin
        x = SecondOrderTensor{2, Float32}((1.0, 2.0, 3.0, 4.0)) ⊕ 3.0f0

        z1 = zero(x)
        z2 = zero(x)

        @test z1 == z2
        @test unpack(z1, 1)::SecondOrderTensor{2, Float32, 4} == zero(SecondOrderTensor{2})
        @test unpack(z1, 2)::Float32 == 0.0
        @test flatview(z1) == zeros(5)
    end

    @testset "basic arithmetic" begin
        A = SecondOrderTensor{2}((1.0, 2.0, 3.0, 4.0))
        B = SecondOrderTensor{2}((5.0, 6.0, 7.0, 8.0))
        x = A ⊕ 3.0
        y = B ⊕ 4.0

        @test unpack(x + y)::Tuple{typeof(A), Float64} == (A + B, 7.0)
        @test unpack(x - y)::Tuple{typeof(A), Float64} == (A - B, -1.0)
        @test unpack(2x)::Tuple{typeof(A), Float64} == (2A, 6.0)
        @test unpack(x / 2)::Tuple{typeof(A), Float64} == (A / 2, 1.5)
    end

    @testset "unary plus and minus" begin
        A = SecondOrderTensor{2, Float32}((1.0, 2.0, 3.0, 4.0))
        x = A ⊕ 3.0f0

        @test +x == x
        @test unpack(-x, 1)::typeof(A) == -A
        @test unpack(-x, 2)::Float32 == -3.0
        @test x + (-x) == zero(x)
        @test -(-x) == x
    end

    @testset "contract: vector-vector" begin
        x = SecondOrderTensor{2}((1.0, 2.0, 3.0, 4.0)) ⊕ 5.0
        y = SecondOrderTensor{2}((6.0, 7.0, 8.0, 9.0)) ⊕ 10.0

        @test (@inferred x ⊡ y)::Float64 == dot(x, y)
    end

    @testset "matrix-vector multiplication" begin
        x = SecondOrderTensor{2}((1.0, 2.0, 3.0, 4.0)) ⊕ 5.0
        Axis = only(Tensorial.ofaxes(x))

        J11 = 2.0 * one(FourthOrderTensor{2})
        J12 = SecondOrderTensor{2}((1.0, 2.0, 3.0, 4.0))
        J21 = SecondOrderTensor{2}((5.0, 6.0, 7.0, 8.0))
        J22 = 9.0

        J = Tensorial._pack(Tuple{Axis, Axis}, (J11, J21, J12, J22))
        y = J * x

        x1, x2 = unpack(x)
        y_expected = (J11 ⊡₂ x1 + J12 * x2) ⊕ (J21 ⊡₂ x1 + J22 * x2)

        @test y == y_expected
    end

    @testset "matrix-matrix multiplication" begin
        x = SecondOrderTensor{2}((1.0, 2.0, 3.0, 4.0)) ⊕ 5.0
        Axis = only(Tensorial.ofaxes(x))

        J11 = 2.0 * one(FourthOrderTensor{2})
        J12 = SecondOrderTensor{2}((1.0, 2.0, 3.0, 4.0))
        J21 = SecondOrderTensor{2}((5.0, 6.0, 7.0, 8.0))
        J22 = 9.0

        A = Tensorial._pack(Tuple{Axis, Axis}, (J11, J21, J12, J22))
        B = Tensorial._pack(Tuple{Axis, Axis}, (J11, J21, J12, J22))

        C = A * B

        @test flatview(C) ≈ flatview(A) * flatview(B)
    end

    @testset "contract: vector-vector" begin
        x = SecondOrderTensor{2}((1.0, 2.0, 3.0, 4.0)) ⊕ 5.0
        y = SecondOrderTensor{2}((6.0, 7.0, 8.0, 9.0)) ⊕ 10.0

        c = @inferred x ⊡ y

        @test c::Float64 == dot(x, y)
    end

    @testset "contract: vector-vector symmetric" begin
        x = SymmetricSecondOrderTensor{2}((1.0, 2.0, 4.0)) ⊕ 5.0
        y = SymmetricSecondOrderTensor{2}((6.0, 7.0, 9.0)) ⊕ 10.0

        c = @inferred x ⊡ y

        @test c::Float64 ≈ dot(x, y)
    end

    @testset "contract: matrix-vector" begin
        x = SecondOrderTensor{2}((1.0, 2.0, 3.0, 4.0)) ⊕ 5.0
        Axis = only(Tensorial.ofaxes(x))

        J11 = 2.0 * one(FourthOrderTensor{2})
        J12 = SecondOrderTensor{2}((1.0, 2.0, 3.0, 4.0))
        J21 = SecondOrderTensor{2}((5.0, 6.0, 7.0, 8.0))
        J22 = 9.0

        J = Tensorial._pack(Tuple{Axis, Axis}, (J11, J21, J12, J22))

        y = @inferred J ⊡ x
        y1, y2 = unpack(y)
        x1, x2 = unpack(x)

        @test y1::typeof(x1) == J11 ⊡₂ x1 + J12 * x2
        @test y2::Float64 == J21 ⊡₂ x1 + J22 * x2
    end

    @testset "contract: matrix-vector symmetric" begin
        x = SymmetricSecondOrderTensor{2}((1.0, 2.0, 4.0)) ⊕ 5.0
        Axis = only(Tensorial.ofaxes(x))

        J11 = 2.0 * one(SymmetricFourthOrderTensor{2})
        J12 = SymmetricSecondOrderTensor{2}((1.0, 2.0, 5.0))
        J21 = SymmetricSecondOrderTensor{2}((6.0, 7.0, 8.0))
        J22 = 9.0

        J = Tensorial._pack(Tuple{Axis, Axis}, (J11, J21, J12, J22))

        y = @inferred J ⊡ x
        y1, y2 = unpack(y)
        x1, x2 = unpack(x)

        @test y1::typeof(x1) ≈ J11 ⊡₂ x1 + J12 * x2
        @test y2::Float64 ≈ J21 ⊡₂ x1 + J22 * x2
    end

    @testset "contract: matrix-matrix" begin
        x = SecondOrderTensor{2}((1.0, 2.0, 3.0, 4.0)) ⊕ 5.0
        Axis = only(Tensorial.ofaxes(x))

        A11 = 2.0 * one(FourthOrderTensor{2})
        A12 = SecondOrderTensor{2}((1.0, 2.0, 3.0, 4.0))
        A21 = SecondOrderTensor{2}((5.0, 6.0, 7.0, 8.0))
        A22 = 9.0

        B11 = 3.0 * one(FourthOrderTensor{2})
        B12 = SecondOrderTensor{2}((2.0, 1.0, 0.0, -1.0))
        B21 = SecondOrderTensor{2}((4.0, 3.0, 2.0, 1.0))
        B22 = 10.0

        A = Tensorial._pack(Tuple{Axis, Axis}, (A11, A21, A12, A22))
        B = Tensorial._pack(Tuple{Axis, Axis}, (B11, B21, B12, B22))

        C = @inferred A ⊡ B
        d = @inferred A ⊡₂ B

        @test C::typeof(A) == DirectSumMatrix{Axis, Axis}(Tuple(flatview(A) * flatview(B)))
        @test d::Float64 == dot(flatview(A), flatview(B))
    end

    @testset "contract: invalid contracted axes" begin
        x = (SecondOrderTensor{2}((1.0, 2.0, 3.0, 4.0)) ⊕ 5.0)
        y = (5.0 ⊕ SecondOrderTensor{2}((6.0, 7.0, 8.0, 9.0)))

        @test_throws AssertionError contract(x, y, Val(1))
    end

    @testset "tensor product of DirectSumArray" begin
        A = SecondOrderTensor{2}((1.0, 2.0, 3.0, 4.0))
        s = 5.0
        x = A ⊕ s

        B = Vec{2}((6.0, 7.0))
        t = 8.0
        y = B ⊕ t

        z = x ⊗ y

        @test size(z) == (2, 2)
        @test Tensorial.flatsize(z) == (5, 3)
        @test flatview(z) == flatview(x) ⊗ flatview(y)

        @test unpack(z, 1, 1)::typeof(A ⊗ B) == A ⊗ B
        @test unpack(z, 1, 2)::typeof(A ⊗ t) == A ⊗ t
        @test unpack(z, 2, 1)::typeof(s ⊗ B) == s ⊗ B
        @test unpack(z, 2, 2)::Float64 == s * t
    end

    @testset "dot and norm: nonsymmetric" begin
        A = SecondOrderTensor{2}((1.0, 2.0, 3.0, 4.0))
        B = SecondOrderTensor{2}((5.0, 6.0, 7.0, 8.0))
        x = A ⊕ 3.0
        y = B ⊕ 4.0

        @test dot(x, y) == dot(flatview(x), flatview(y))
        @test dot(x, x) == dot(flatview(x), flatview(x))
        @test norm(x) == norm(flatview(x))
        @test norm(y) == norm(flatview(y))
    end

    @testset "dot and norm: symmetric" begin
        A = SymmetricSecondOrderTensor{2}((1.0, 2.0, 4.0))
        B = SymmetricSecondOrderTensor{2}((5.0, 6.0, 8.0))
        x = A ⊕ 3.0
        y = B ⊕ 4.0

        @test dot(x, y) ≈ dot(flatview(x), flatview(y))
        @test dot(x, x) ≈ dot(flatview(x), flatview(x))
        @test norm(x) ≈ norm(flatview(x))
        @test norm(y) ≈ norm(flatview(y))
    end

    @testset "gradient(identity): nonsymmetric" begin
        A = SecondOrderTensor{2}((1.0, 2.0, 3.0, 4.0))
        x = A ⊕ 3.0
        G = gradient(identity, x)

        @test size(G) == (2, 2)
        @test Tensorial.flatsize(G) == (5, 5)

        @test unpack(G, 1, 1)::FourthOrderTensor{2, Float64, 16} == one(unpack(G, 1, 1))
        @test unpack(G, 1, 2)::typeof(A) == zero(unpack(G, 1, 2))
        @test unpack(G, 2, 1)::typeof(A) == zero(unpack(G, 2, 1))
        @test unpack(G, 2, 2)::Float64 == 1.0

        @test flatview(G) == Matrix(I, 5, 5)

        Gall = gradient(identity, x, :all)
        @test Gall[1] == G
        @test Gall[2] == x
    end

    @testset "gradient(identity): symmetric" begin
        A = SymmetricSecondOrderTensor{2}((1.0, 2.0, 4.0))
        x = A ⊕ 3.0
        G = gradient(identity, x)

        @test size(G) == (2, 2)
        @test Tensorial.flatsize(G) == (4, 4)

        @test unpack(G, 1, 1)::SymmetricFourthOrderTensor{2, Float64, 9} == one(unpack(G, 1, 1))
        @test unpack(G, 1, 2)::typeof(A) == zero(unpack(G, 1, 2))
        @test unpack(G, 2, 1)::typeof(A) == zero(unpack(G, 2, 1))
        @test unpack(G, 2, 2)::Float64 == 1.0

        @test flatview(G) ≈ Matrix(I, 4, 4)

        Gall = gradient(identity, x, :all)
        @test Gall[1] == G
        @test Gall[2] == x
    end

    @testset "jacobian of coupled map: nonsymmetric" begin
        A = SecondOrderTensor{2}((1.0, 2.0, 3.0, 4.0))
        s = 5.0
        x = A ⊕ s

        C = SecondOrderTensor{2}((10.0, 20.0, 30.0, 40.0))
        M = SecondOrderTensor{2}((1.0, 2.0, 3.0, 4.0))
        β = 7.0

        function f(z)
            A_, s_ = unpack(z)
            B = A_ + s_ * C
            t = dot(M, A_) + β * s_
            B ⊕ t
        end

        J = gradient(f, x)

        @test unpack(J, 1, 1)::FourthOrderTensor{2, Float64, 16} == one(unpack(J, 1, 1))
        @test unpack(J, 1, 2)::typeof(C) == C
        @test unpack(J, 2, 1)::typeof(M) == M
        @test unpack(J, 2, 2)::Float64 == β

        @test flatview(J)[1, 5] == 10.0
        @test flatview(J)[2, 5] == 20.0
        @test flatview(J)[3, 5] == 30.0
        @test flatview(J)[4, 5] == 40.0

        @test flatview(J)[5, 1] == 1.0
        @test flatview(J)[5, 2] == 2.0
        @test flatview(J)[5, 3] == 3.0
        @test flatview(J)[5, 4] == 4.0
        @test flatview(J)[5, 5] == 7.0
    end

    @testset "jacobian of coupled map: symmetric" begin
        A = SymmetricSecondOrderTensor{2}((1.0, 2.0, 4.0))
        s = 5.0
        x = A ⊕ s

        C = SymmetricSecondOrderTensor{2}((10.0, 20.0, 40.0))
        β = 7.0

        function f(z)
            A_, s_ = unpack(z)
            B = A_ + s_ * C
            t = sum(A_) + β * s_
            B ⊕ t
        end

        J = gradient(f, x)

        @test unpack(J, 1, 1)::SymmetricFourthOrderTensor{2, Float64, 9} == one(unpack(J, 1, 1))
        @test unpack(J, 1, 2)::typeof(C) == C
        @test unpack(J, 2, 1)::typeof(A) == SymmetricSecondOrderTensor{2}((1.0, 1.0, 1.0))
        @test unpack(J, 2, 2)::Float64 == β

        @test flatview(J)[1, 4] ≈ 10.0
        @test flatview(J)[2, 4] ≈ sqrt(2) * 20.0
        @test flatview(J)[3, 4] ≈ 40.0

        @test flatview(J)[4, 1] ≈ 1.0
        @test flatview(J)[4, 2] ≈ sqrt(2)
        @test flatview(J)[4, 3] ≈ 1.0
        @test flatview(J)[4, 4] ≈ 7.0
    end

    @testset "linear solve and inverse: nonsymmetric" begin
        A = SecondOrderTensor{2}((1.0, 2.0, 3.0, 4.0))
        x = A ⊕ 5.0
        Axis = only(Tensorial.ofaxes(x))

        J11 = FourthOrderTensor{2}((2.0, 0.0, 0.0, 0.0,
                                    0.0, 3.0, 0.0, 0.0,
                                    0.0, 0.0, 4.0, 0.0,
                                    0.0, 0.0, 0.0, 5.0))
        J12 = SecondOrderTensor{2}((1.0, 2.0, 3.0, 4.0))
        J21 = SecondOrderTensor{2}((5.0, 6.0, 7.0, 8.0))
        J22 = 9.0

        J = Tensorial._pack(Tuple{Axis, Axis}, (J11, J21, J12, J22))

        u_true = SecondOrderTensor{2}((1.0, -1.0, 2.0, -2.0)) ⊕ 3.0
        u1, u2 = unpack(u_true)

        b = (J11 ⊡₂ u1 + J12 * u2) ⊕ (J21 ⊡₂ u1 + J22 * u2)

        u = @inferred J \ b
        Jinv = @inferred inv(J)

        @test u::typeof(x) ≈ u_true

        @test unpack(Jinv \ u_true, 1)::typeof(u1) ≈ unpack(b, 1)
        @test unpack(Jinv \ u_true, 2)::Float64 ≈ unpack(b, 2)
    end

    @testset "linear solve and inverse: symmetric" begin
        A = SymmetricSecondOrderTensor{2}((1.0, 2.0, 4.0))
        x = A ⊕ 5.0
        Axis = only(Tensorial.ofaxes(x))

        J11 = 2.0 * one(SymmetricFourthOrderTensor{2})
        J12 = SymmetricSecondOrderTensor{2}((1.0, 2.0, 5.0))
        J21 = SymmetricSecondOrderTensor{2}((6.0, 7.0, 8.0))
        J22 = 9.0

        J = Tensorial._pack(Tuple{Axis, Axis}, (J11, J21, J12, J22))

        u_true = SymmetricSecondOrderTensor{2}((1.0, -2.0, 3.0)) ⊕ 4.0
        u1, u2 = unpack(u_true)

        b = (J11 ⊡₂ u1 + J12 * u2) ⊕ (J21 ⊡₂ u1 + J22 * u2)

        u = @inferred J \ b
        Jinv = @inferred inv(J)

        @test u::typeof(x) ≈ u_true

        @test unpack(Jinv \ u_true, 1)::typeof(u1) ≈ unpack(b, 1)
        @test unpack(Jinv \ u_true, 2)::Float64 ≈ unpack(b, 2)
    end

    @testset "hessian of coupled quadratic: nonsymmetric" begin
        A = SecondOrderTensor{2}((1.0, 2.0, 3.0, 4.0))
        s = 5.0
        x = A ⊕ s

        C = SecondOrderTensor{2}((10.0, 20.0, 30.0, 40.0))
        α = 2.0
        γ = 7.0

        function f(z)
            A_, s_ = unpack(z)
            α * dot(A_, A_) + s_ * dot(C, A_) + γ * s_^2
        end

        g = gradient(f, x)
        H = hessian(f, x)

        @test unpack(g, 1)::typeof(A) == 2α * A + s * C
        @test unpack(g, 2)::Float64 == dot(C, A) + 2γ * s

        @test unpack(H, 1, 1)::FourthOrderTensor{2, Float64, 16} == 2α * one(unpack(H, 1, 1))
        @test unpack(H, 1, 2)::typeof(C) == C
        @test unpack(H, 2, 1)::typeof(C) == C
        @test unpack(H, 2, 2)::Float64 == 2γ
    end

    @testset "hessian of coupled quadratic: symmetric" begin
        A = SymmetricSecondOrderTensor{2}((1.0, 2.0, 4.0))
        s = 5.0
        x = A ⊕ s

        C = SymmetricSecondOrderTensor{2}((10.0, 20.0, 40.0))
        α = 2.0
        γ = 7.0

        function f(z)
            A_, s_ = unpack(z)
            α * dot(A_, A_) + s_ * dot(C, A_) + γ * s_^2
        end

        g = gradient(f, x)
        H = hessian(f, x)

        @test unpack(g, 1)::typeof(A) ≈ 2α * A + s * C
        @test unpack(g, 2)::Float64 ≈ dot(C, A) + 2γ * s

        @test unpack(H, 1, 1)::SymmetricFourthOrderTensor{2, Float64, 9} ≈ 2α * one(unpack(H, 1, 1))
        @test unpack(H, 1, 2)::typeof(C) == C
        @test unpack(H, 2, 1)::typeof(C) == C
        @test unpack(H, 2, 2)::Float64 == 2γ

        @test flatview(H)[1, 4] ≈ 10.0
        @test flatview(H)[2, 4] ≈ sqrt(2) * 20.0
        @test flatview(H)[3, 4] ≈ 40.0
        @test flatview(H)[4, 4] == 14.0
    end

    @testset "linear scalar functional: nonsymmetric" begin
        A = SecondOrderTensor{2}((1.0, 2.0, 3.0, 4.0))
        x = A ⊕ 3.0
        f(z) = sum(flatview(z))

        g = gradient(f, x)
        H = hessian(f, x)

        @test unpack(g, 1)::typeof(A) == SecondOrderTensor{2}((1.0, 1.0, 1.0, 1.0))
        @test unpack(g, 2)::Float64 == 1.0

        @test unpack(H, 1, 1)::FourthOrderTensor{2, Float64, 16} == zero(unpack(H, 1, 1))
        @test unpack(H, 1, 2)::typeof(A) == zero(unpack(H, 1, 2))
        @test unpack(H, 2, 1)::typeof(A) == zero(unpack(H, 2, 1))
        @test unpack(H, 2, 2)::Float64 == 0.0
    end

    @testset "linear scalar functional: symmetric (internal coordinates)" begin
        A = SymmetricSecondOrderTensor{2}((1.0, 2.0, 4.0))
        x = A ⊕ 3.0
        f(z) = sum(flatview(z))

        g = gradient(f, x)

        G1_expected = SymmetricSecondOrderTensor{2}((1.0, 1 / sqrt(2), 1.0))
        @test unpack(g, 1)::typeof(A) ≈ G1_expected
        @test unpack(g, 2)::Float64 == 1.0

        @test flatview(g)[1] ≈ 1.0
        @test flatview(g)[2] ≈ 1.0
        @test flatview(g)[3] ≈ 1.0
    end

    @testset "linear scalar functional: symmetric (full tensor entries)" begin
        A = SymmetricSecondOrderTensor{2}((1.0, 2.0, 4.0))
        x = A ⊕ 3.0
        f(z) = sum(unpack(z, 1)) + unpack(z, 2)

        g = gradient(f, x)

        G1_expected = SymmetricSecondOrderTensor{2}((1.0, 1.0, 1.0))
        @test unpack(g, 1)::typeof(A) ≈ G1_expected
        @test unpack(g, 2)::Float64 == 1.0

        @test flatview(g)[2] ≈ sqrt(2)
    end

    @testset "flatview sizes" begin
        x = SecondOrderTensor{2}((1.0, 2.0, 3.0, 4.0)) ⊕ 3.0
        y = SymmetricSecondOrderTensor{2}((1.0, 2.0, 4.0)) ⊕ 3.0

        @test size(flatview(x)) == Tensorial.flatsize(x)
        @test size(flatview(y)) == Tensorial.flatsize(y)

        @test typeof(flatview(x)) == SVector{5, Float64}
        @test typeof(flatview(y)) == SVector{4, Float64}
    end

    @testset "summary and show" begin
        x = gradient(identity, SecondOrderTensor{2}((1.0, 2.0, 3.0, 4.0)) ⊕ 3.0)

        s = sprint(summary, x)
        sh = sprint(show, x)

        @test occursin("2×2 DirectSumMatrix", s)
        @test occursin("with storage Float64", s)

        @test occursin("2×2 DirectSumMatrix", sh)
        @test occursin("Space(2, 2, 2, 2)", sh)
        @test occursin("Space(2, 2)", sh)
        @test occursin("Space()", sh)
    end
end
