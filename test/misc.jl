@testset "Misc" begin
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
            @test (Tensorial.resize(v, Val(2)))::Vec{2,T} == Array(v)[1:2]
            @test (Tensorial.resize(v, Val(6)))::Vec{6,T} == [Array(v); zeros(2)]
            @test (Tensorial.resize(x, Val(2), Val(3)))::Mat{2,3,T} == Array(x)[1:2,1:3]
            @test (Tensorial.resize(x, Val(4), Val(6)))::Mat{4,6,T} == [Array(x) zeros(3,2); zeros(4)' zeros(2)']
            # resizedim
            @test (Tensorial.resizedim(v, Val(2)))::Vec{2,T} == Array(v)[1:2]
            @test (Tensorial.resizedim(v, Val(6)))::Vec{6,T} == [Array(v); zeros(2)]
            @test (Tensorial.resizedim(x, Val(2)))::Mat{2,2,T} == Array(x)[1:2,1:2]
            @test (Tensorial.resizedim(x, Val(6)))::Mat{6,6,T} == [Array(x) zeros(3,2); zeros(3,4) zeros(3,2)]
        end
    end
end
