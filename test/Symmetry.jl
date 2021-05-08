@testset "Symmetry" begin
    # chwck type instability
    f() = Symmetry(3,3)
    (@inferred f())::Symmetry{Tuple{3,3}}

    # constructors
    @test Symmetry(3,3) == Symmetry{Tuple{3,3}}()
    @test Symmetry((3,3)) == Symmetry{Tuple{3,3}}()
    @test Symmetry(Tuple{3,3}) == Symmetry{Tuple{3,3}}()

    # ncomponents
    @test Tensorial.ncomponents(Symmetry(3,3)) == 6
    @test Tensorial.ncomponents(Symmetry(2,2)) == 3

    # Dims
    @test Dims(Symmetry(3,3)) == (3,3)
    @test Dims(Symmetry(2,2)) == (2,2)

    # length
    @test length(Symmetry(3,3,3)) == 3
    @test length(Symmetry(2,2)) == 2

    # getindex
    S = Symmetry(3,3)
    @test (@inferred S[1])::Int == 3
    @test (@inferred S[2])::Int == 3
    @test_throws Exception S[3]

    # macro
    @test @Symmetry{3,3} == Symmetry{Tuple{3,3}}

    # errors
    @test_throws Exception Symmetry(Tuple{3,Int})
    @test_throws Exception Symmetry(3,2)
end
