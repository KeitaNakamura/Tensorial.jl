struct Point{dim, T} <: AbstractVec{dim, T}
    x::NTuple{dim, T}
end
Base.Tuple(p::Point) = p.x
Base.getindex(p::Point, i::Int) = p.x[i]
@testset "AbstractTensor" begin
    for T in (Float32, Float64)
        x = Vec{2, T}(1, 2)
        p = Point(Tuple(x))
        @test (@inferred p + p)::Vec{2, T} |> Tuple == (x + x).data
        @test (@inferred p - p)::Vec{2, T} |> Tuple == (x - x).data
        @test (@inferred p ⋅ p)::T == x ⋅ x
        @test (@inferred p ⊗ p)::Mat{2, 2, T} == x ⊗ x
    end

    # size/axes
    TT = typeof(rand(Mat{3, 2}))
    @test (@inferred size(TT)) == (3, 2)
    @test (@inferred size(TT, 1)) == 3
    @test (@inferred size(TT, 2)) == 2
    @test (@inferred size(TT, 3)) == 1
    @test (@inferred size(TT, 4)) == 1
    @test (@inferred axes(TT)) == (Base.OneTo(3), Base.OneTo(2))
    @test (@inferred axes(TT, 1)) == Base.OneTo(3)
    @test (@inferred axes(TT, 2)) == Base.OneTo(2)
    @test (@inferred axes(TT, 3)) == Base.OneTo(1)
    @test (@inferred axes(TT, 4)) == Base.OneTo(1)
end
