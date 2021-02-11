@testset "Space" begin
    # basic
    @test length(Space(3,3)) == 2
    @test length(Space(Symmetry(3,3),3)) == 2
    @test ndims(Space(Symmetry(3,3),3)) == 3
    @test Tuple(Space(Symmetry(3,3),3)) == (Symmetry(3,3),3)
    @test Space(3,2)[1] == 3
    @test Space(3,2)[2] == 2
end
