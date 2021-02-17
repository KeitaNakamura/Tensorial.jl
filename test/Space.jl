@testset "Space" begin
    # basic
    @test (@inferred length(Space(3,3))) == 2
    @test (@inferred length(Space(Symmetry(3,3),3))) == 2
    @test (@inferred Tensorial.tensorsize(Space(Symmetry(3,3),3))) == (3,3,3)
    @test (@inferred Tensorial.tensororder(Space(Symmetry(3,3),3))) == 3
    @test (@inferred Tuple(Space(Symmetry(3,3),3))) == (Symmetry(3,3),3)
    @test Space(3,2)[1] == 3
    @test Space(3,2)[2] == 2
    # prohibited
    @test_throws Exception size(Space(Symmetry(3,3),3))
    @test_throws Exception ndims(Space(Symmetry(3,3),3))
    # promotion
    @test (@inferred Tensorial.promote_space(Space(3,2), Space(3,2))) == Space(3,2)
    @test (@inferred Tensorial.promote_space(Space(3,3,3), Space(Symmetry(3,3),3))) == Space(3,3,3)
    @test (@inferred Tensorial.promote_space(Space(Symmetry(3,3,3)), Space(Symmetry(3,3),3))) == Space(Symmetry(3,3),3)
    @test (@inferred Tensorial.promote_space(Space(Symmetry(3,3,3)), Space(3,Symmetry(3,3)))) == Space(3,Symmetry(3,3))
    @test (@inferred Tensorial.promote_space(Space(Symmetry(3,3,3,3)), Space(3,Symmetry(3,3),3))) == Space(3,Symmetry(3,3),3)
    # wrong promotion
    @test_throws Exception Tensorial.promote_space(Space(3,2), Space(3,1))
    @test_throws Exception Tensorial.promote_space(Space(3,2), Symmetry(Space(3,3)))
end
