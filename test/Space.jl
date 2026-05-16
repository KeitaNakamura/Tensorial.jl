using Tensorial: Space

@testset "Space" begin
    # basic
    @test (@inferred Tensorial.tensorsize(Space(Symmetry(3,3),3))) == (3,3,3)
    @test (@inferred Tensorial.tensororder(Space(Symmetry(3,3),3))) == 3
    @test (@inferred Tuple(Space(Symmetry(3,3),3))) == (Symmetry(3,3),3)
    @test Tuple(Space(3,2))[1] == 3
    @test Tuple(Space(3,2))[2] == 2
    @test Tensorial.Space(Tuple{3, Symmetry{Tuple{3,3}}}) == Space(3, Symmetry(3,3))
    @test_throws ArgumentError Space((3, :invalid))
    @test_throws ArgumentError Tensorial.Space(Tuple{Symmetry{Tuple{3}}})
    @test Tensorial.tensororder(3) == 0
    @test Tensorial.tensoraxes(Space(2,3)) == (Base.OneTo(2), Base.OneTo(3))
    # prohibited
    @test_throws Exception size(Space(Symmetry(3,3),3))
    @test_throws Exception ndims(Space(Symmetry(3,3),3))
    @test_throws ArgumentError Tensorial.StaticIndex(:invalid)
    # bounds
    @test checkbounds(Bool, Space(2,3), 6)
    @test !checkbounds(Bool, Space(2,3), 7)
    @test checkbounds(Bool, Space(2,3), 2, 3)
    @test !checkbounds(Bool, Space(2,3), 2, 3, 1)
    # drop dimensions
    @test (@inferred Tensorial.dropfirst(Space(Symmetry(3,3,3)), Val(1))) == Space(Symmetry(3,3))
    @test (@inferred Tensorial.dropfirst(Space(Symmetry(3,3,3)), Val(2))) == Space(3)
    @test (@inferred Tensorial.droplast(Space(2, Symmetry(3,3), 4), Val(2))) == Space(2,3)
    @test (@inferred Tensorial.dropfirst(Space(2,3), Val(0))) == Space(2,3)
    @test Tensorial.dropfirst(Space(2,3)) == Space(3)
    @test Tensorial.droplast(Space(2,3)) == Space(2)
    @test_throws ArgumentError Tensorial.dropfirst(Space(2), Val(:bad))
    @test_throws ArgumentError Tensorial.dropfirst(Space(2), Val(2))
    @test_throws ArgumentError Tensorial.droplast(Space(2), Val(-1))
    # contractions
    @test (@inferred Tensorial.contract(Space(2,3), Space(3,4), Val(1))) == Space(2,4)
    @test (@inferred Tensorial.contract(Space(2,3), Space(2,4), Val(0))) == Space(2,3,2,4)
    @test Tensorial.contract2(Space(2,3,4), Space(3,4,5)) == Space(2,5)
    @test_throws DimensionMismatch Tensorial.contract(Space(2,3), Space(2,4), Val(1))
    @test_throws DimensionMismatch Tensorial.contract(Space(2), Space(2), Val(2))
    # promotion
    @test (@inferred Tensorial.promote_space()) == Space()
    @test (@inferred Tensorial.promote_space(Space(3,2), Space(3,2))) == Space(3,2)
    @test (@inferred Tensorial.promote_space(Space(3,3,3), Space(Symmetry(3,3),3))) == Space(3,3,3)
    @test (@inferred Tensorial.promote_space(Space(Symmetry(3,3,3)), Space(Symmetry(3,3),3))) == Space(Symmetry(3,3),3)
    @test (@inferred Tensorial.promote_space(Space(Symmetry(3,3,3)), Space(3,Symmetry(3,3)))) == Space(3,Symmetry(3,3))
    @test (@inferred Tensorial.promote_space(Space(Symmetry(3,3,3,3)), Space(3,Symmetry(3,3),3))) == Space(3,Symmetry(3,3),3)
    @test Tensorial.promote_space(Space(2,2), Space(Symmetry(2,2)), Space(2,2)) == Space(2,2)
    # wrong promotion
    @test_throws Exception Tensorial.promote_space(Space(3,2), Space(3,1))
    @test_throws Exception Tensorial.promote_space(Space(3,2), Symmetry(Space(3,3)))
    # static indexing
    @test length(Tensorial.StaticIndex(2)) == 1
    @test (@inferred Space(2,3)[1]) == Space()
    @test Space(2,3)[Tensorial.StaticIndex(1)] == Space()
    @test (@inferred Space(2,3)[:]) == Space(6)
    @test (@inferred Space(2,3)[SVector(1,3)]) == Space(2)
    @test_throws BoundsError Space(2,3)[1,1,1]
    @test (@inferred Space(Symmetry(3,3))[:, :]) == Space(Symmetry(3,3))
    @test (@inferred Space(Symmetry(3,3))[SVector(1,2), SVector(1,2)]) == Space(2,2)
    @test (@inferred Space(Symmetry(3,3))[Tensorial.StaticIndex(SVector(1,2)), Tensorial.StaticIndex(SVector(1,2))]) == Space(Symmetry(2,2))
    @test (@inferred Space(Symmetry(2,2,2))[Tensorial.StaticIndex(SVector(1,2)), Tensorial.StaticIndex(SVector(1,2)), 1]) == Space(Symmetry(2,2))
    @test (@inferred Space(Symmetry(3,3), Symmetry(3,3))[Tensorial.StaticIndex(SVector(1,2)), Tensorial.StaticIndex(SVector(1,2)), Tensorial.StaticIndex(SVector(1,2)), Tensorial.StaticIndex(SVector(1,2))]) == Space(Symmetry(2,2), Symmetry(2,2))
end
