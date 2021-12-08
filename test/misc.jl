@testset "Misc" begin
    # vec
    x = rand(Vec{3})
    @test vec(x) == x
    x = rand(Mat{3,3})
    @test vec(x) == vec(Array(x))
    @test vec(symmetric(x)) == vec(Array(symmetric(x)))
    x = rand(SymmetricFourthOrderTensor{3})
    @test vec(x) == vec(Array(x))
end
