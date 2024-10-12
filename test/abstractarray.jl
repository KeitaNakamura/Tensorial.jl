@testset "AbstractArray interface" begin
    @testset "static getindex" begin
        v = Vec(5,4,3,2,1)
        @test @inferred(v[:]) === Vec(5,4,3,2,1)
        @test @inferred(v[SUnitRange(2,4)]) === Vec(4,3,2)
        @test @inferred(v[SVector(3,4,1)]) === Vec(3,2,5)
        A = Mat{2,3}(1,2,3,4,5,6)
        @test @inferred(A[:]) === Vec(1,2,3,4,5,6)
        @test @inferred(A[1,:]) === Vec(1,3,5)
        @test A[end,:] === Vec(2,4,6)
        @test @inferred(A[:,1]) === Vec(1,2)
        @test A[:,end] === Vec(5,6)
        @test @inferred(A[1,SVector(1,3)]) === Vec(1,5)
        # by @Tensor macro
        @test @Tensor(A[:,[1,3]]) === @Mat [1 5; 2 6]
        @test @Tensor(A[1,:]) === Vec(1,3,5)
        @test @Tensor(A[end,:]) === Vec(2,4,6)
        @test @Tensor(A[1,2:3]) === Vec(3,5)
        @test_throws Exception @Tensor(Array(A)[1,2:3])
        # check symmetry
        A = SymmetricSecondOrderTensor{3}(1,2,3,4,5,6)
        @test @Tensor(A[:,[1,3]]) === @Mat [1 3; 2 5; 3 6]
        @test @Tensor(A[1,:]) === Vec(1,2,3)
        @test @Tensor(A[end,:]) === Vec(3,5,6)
        @test @Tensor(A[1,2:3]) === Vec(2,3)
        @test @Tensor(A[1:2,[1,2]]) === SymmetricSecondOrderTensor{2}(1,2,4)
        n = 1 # dynamic
        @test @Tensor(A[:,[n,3]]) === @Mat [1 3; 2 5; 3 6]
        @test @Tensor(A[n,:]) === Vec(1,2,3)
        @test @Tensor(A[n,2:3]) === Vec(2,3)
        @test @Tensor(A[n:2,[1,2]]) === SymmetricSecondOrderTensor{2}(1,2,4)
        @test @Tensor(A[n:2,[n,2]]) === @Mat [1 2; 2 4]
        # complex version
        A = rand(Tensor{Tuple{3,3,@Symmetry({3,3,3})}})
        @test (@Tensor(A[1,2,1:2,2:3,3]))::Tensor{Tuple{2,2}} == Array(A)[1,2,1:2,2:3,3]
        @test (@Tensor(A[1,2:3,2:3,2:3,3]))::Tensor{Tuple{2,@Symmetry({2,2})}} == Array(A)[1,2:3,2:3,2:3,3]
        @test (@Tensor(A[1,2:3,2:3,2,2:3]))::Tensor{Tuple{2,@Symmetry({2,2})}} == Array(A)[1,2:3,2:3,2,2:3]
        @test (@Tensor(A[1,2:3,2,2:3,2:3]))::Tensor{Tuple{2,@Symmetry({2,2})}} == Array(A)[1,2:3,2,2:3,2:3]
    end
    @testset "vcat/hcat" begin
        @test @inferred(vcat(Vec(1,2,3))) === Vec(1,2,3)
        @test @inferred(vcat(Mat{1,3}(1,2,3))) === Mat{1,3}(1,2,3)
        @test @inferred(hcat(Vec(1,2,3))) === Mat{3,1}(1,2,3)
        @test @inferred(hcat(Mat{3,1}(1,2,3))) === Mat{3,1}(1,2,3)

        @test @inferred(vcat(Vec(1,2,3), Vec(4,5,6))) === Vec(1,2,3,4,5,6)
        @test @inferred(vcat(Vec(1,2,3), Mat{3,1}(4,5,6))) === Mat{6,1}(1,2,3,4,5,6)
        @test @inferred(vcat(Mat{1,3}(1,2,3), Mat{1,3}(4,5,6))) === @Mat [1 2 3; 4 5 6]
        @test @inferred(vcat(symmetric(Mat{2,2}(1,2,2,3), :U), Mat{1,2}(4,5))) === @Mat [1 2; 2 3; 4 5]
        @test @inferred(hcat(Vec(1,2,3), Vec(4,5,6))) === Mat{3,2}(1,2,3,4,5,6)
        @test @inferred(hcat(Vec(1,2,3), Mat{3,2}(4,5,6,7,8,9))) === Mat{3,3}(1,2,3,4,5,6,7,8,9)
        @test @inferred(hcat(Mat{3,1}(1,2,3), Mat{3,2}(4,5,6,7,8,9))) === Mat{3,3}(1,2,3,4,5,6,7,8,9)
        @test @inferred(hcat(symmetric(Mat{2,2}(1,2,2,3), :U), Vec(4,5))) === @Mat [1 2 4; 2 3 5]

        @test @inferred(vcat(Vec(1), Vec(2), Vec(3), Mat{2,1}(4, 5))) === Mat{5,1}(1,2,3,4,5)
        @test @inferred(vcat(Vec(1), Vec(2), Vec(3), Vec(4), Vec(5))) === Vec(1,2,3,4,5)
        @test @inferred(hcat(Vec(1), Vec(2), Vec(3), Vec(4), Vec(5))) === Mat{1,5}(1,2,3,4,5)

        # special cases for vcat with vector and real
        ## start from Vec
        @test @inferred(vcat(Vec(1), 2)) === Vec(1,2)
        @test @inferred(vcat(Vec(1), 2, Vec(3))) === Vec(1,2,3)
        ## start from Real
        n = 10
        for I in 1:n
            @test @inferred(vcat(1:I..., Vec(I+1))) === Vec((1:I+1)...)
            @test @inferred(vcat(1:I..., Vec(I+1), I+2)) === Vec((1:I+2)...)
            @test @inferred(vcat(1:I..., Vec(I+1), Vec(I+2))) === Vec((1:I+2)...)
        end
        n += 1
        @test @inferred(vcat(1:n..., Vec(n+1)))::Vector == 1:n+1
        @test @inferred(vcat(1:n..., Vec(n+1), n+2))::Vector == 1:n+2
        @test @inferred(vcat(1:n..., Vec(n+1), Vec(n+2)))::Vector == 1:n+2

        @test vcat(Vec(1.0f0), Vec(1.0)) === Vec(1.0, 1.0)
        @test hcat(Vec(1.0f0), Vec(1.0)) === Mat{1,2}(1.0, 1.0)
    end
    @testset "hvcat" begin
        f_2_2(x11,x12,x21,x22) = [x11 x12
                                  x21 x22]
        f_2_3(x11,x12,x21,x22,x23) = [x11 x12
                                      x21 x22 x23]
        f_2_2_recurse(x11,x12,x21,x22,x22_11,x22_12,x22_21,x22_22) = [x11 x12
                                                                      x21 [x22_11 x22_12; x22_21 x22_22]]
        for T in (Float32, Float64)
            x11 = rand(Mat{2,2,T})
            x12 = rand(Vec{2,T})
            x21 = rand(Mat{1,2,T})
            x22 = rand(T)
            @test (@inferred f_2_2(x11, x12, x21, x22))::Mat{3,3,T} == f_2_2(Array.((x11, x12, x21))..., x22)
            x11 = rand(SymmetricSecondOrderTensor{3,T})
            x12 = rand(SecondOrderTensor{3,T})
            x21 = rand(Vec{2,T})
            x22 = rand(SymmetricSecondOrderTensor{2,T})
            x23 = rand(Mat{2,3,T})
            @test (@inferred f_2_3(x11, x12, x21, x22, x23))::Mat{5,6,T} == f_2_3(Array.((x11, x12, x21, x22, x23))...)
            x11 = rand(T)
            x12 = rand(Mat{1,3,T})
            x21 = rand(Vec{3,T})
            x22_11 = rand(SymmetricSecondOrderTensor{2,T})
            x22_12 = rand(Vec{2,T})
            x22_21 = rand(Mat{1,2,T})
            x22_22 = rand(T)
            @test (@inferred f_2_2_recurse(x11,x12,x21,x22,x22_11,x22_12,x22_21,x22_22))::Mat{4,4,T} == f_2_2_recurse(x11,Array.((x12,x21,x22,x22_11,x22_12,x22_21))...,x22_22)
        end
    end
    @testset "reverse" begin
        @test @inferred(reverse(Vec(1, 2, 3))) === Vec(3, 2, 1)
        A = rand(Mat{3,4})
        @test @inferred(reverse(A))::typeof(A) == reverse(reverse(collect(A), dims = 1), dims = 2)
    end
    @testset "vec" begin
        for T in (Float64, Float32)
            x = rand(Vec{3,T})
            @test (@inferred vec(x))::Vec{3,T} == x
            x = rand(Mat{3,3,T})
            @test (@inferred vec(x))::Vec{9,T} == vec(Array(x))
            @test (@inferred vec(symmetric(x)))::Vec{9,T} == vec(Array(symmetric(x)))
            x = rand(SymmetricFourthOrderTensor{3,T})
            @test (@inferred vec(x))::Vec{81,T} == vec(Array(x))
        end
    end
    @testset "resize/resizedim" begin
        for T in (Float64, Float32)
            v = rand(Vec{4,T})
            x = rand(Mat{3,4,T})
            # resize
            @test (@inferred resize(v, Val((2,))))::Vec{2,T} == Array(v)[1:2]
            @test (@inferred resize(v, Val((6,))))::Vec{6,T} == [Array(v); zeros(2)]
            @test (@inferred resize(x, Val((2,3))))::Mat{2,3,T} == Array(x)[1:2,1:3]
            @test (@inferred resize(x, Val((4,6))))::Mat{4,6,T} == [Array(x) zeros(3,2); zeros(4)' zeros(2)']
            @test (@inferred (v->resize(v, 2))(v))::Vec{2,T} == Array(v)[1:2]
            @test (@inferred (v->resize(v, 6))(v))::Vec{6,T} == [Array(v); zeros(2)]
            @test (@inferred (x->resize(x, (2,3)))(x))::Mat{2,3,T} == Array(x)[1:2,1:3]
            @test (@inferred (x->resize(x, (4,6)))(x))::Mat{4,6,T} == [Array(x) zeros(3,2); zeros(4)' zeros(2)']
            # resizedim
            @test (@inferred Tensorial.resizedim(v, Val(2)))::Vec{2,T} == Array(v)[1:2]
            @test (@inferred Tensorial.resizedim(v, Val(6)))::Vec{6,T} == [Array(v); zeros(2)]
            @test (@inferred Tensorial.resizedim(x, Val(2)))::Mat{2,2,T} == Array(x)[1:2,1:2]
            @test (@inferred Tensorial.resizedim(x, Val(6)))::Mat{6,6,T} == [Array(x) zeros(3,2); zeros(3,4) zeros(3,2)]
            @test (@inferred (v->Tensorial.resizedim(v, 2))(v))::Vec{2,T} == Array(v)[1:2]
            @test (@inferred (v->Tensorial.resizedim(v, 6))(v))::Vec{6,T} == [Array(v); zeros(2)]
            @test (@inferred (x->Tensorial.resizedim(x, 2))(x))::Mat{2,2,T} == Array(x)[1:2,1:2]
            @test (@inferred (x->Tensorial.resizedim(x, 6))(x))::Mat{6,6,T} == [Array(x) zeros(3,2); zeros(3,4) zeros(3,2)]
        end
    end
end
