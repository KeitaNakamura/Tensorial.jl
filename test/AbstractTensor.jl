@testset "AbstractTensor" begin
    struct Point{dim, T} <: AbstractVec{dim, T}
        x::NTuple{dim, T}
    end
    Base.Tuple(p::Point) = p.x
    Base.getindex(p::Point, i::Int) = p.x[i]
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

@testset "Operations for array interface" begin
    @testset "getindex for slicing" begin
        v = Vec(5,4,3,2,1)
        @test @inferred(v[:]) === Vec(5,4,3,2,1)
        @test @inferred(v[SOneTo(3)]) === Vec(5,4,3)
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
    end
    @testset "vcat/hcat" begin
        @test @inferred(vcat(Vec(1,2,3))) === Vec(1,2,3)
        @test @inferred(vcat(Vec(1,2,3)')) === Vec(1,2,3)'
        @test @inferred(vcat(Mat{1,3}(1,2,3))) === Mat{1,3}(1,2,3)
        @test @inferred(hcat(Vec(1,2,3))) === Mat{3,1}(1,2,3)
        @test @inferred(hcat(Vec(1,2,3)')) === Vec(1,2,3)'
        @test @inferred(hcat(Mat{3,1}(1,2,3))) === Mat{3,1}(1,2,3)

        @test @inferred(vcat(Vec(1,2,3), Vec(4,5,6))) === Vec(1,2,3,4,5,6)
        @test @inferred(vcat(Vec(1,2,3), Mat{3,1}(4,5,6))) === Mat{6,1}(1,2,3,4,5,6)
        @test @inferred(vcat(Mat{1,3}(1,2,3), Mat{1,3}(4,5,6))) === @Mat [1 2 3; 4 5 6]
        @test @inferred(vcat(Mat{1,3}(1,2,3), Vec(4,5,6)')) === @Mat [1 2 3; 4 5 6]
        @test @inferred(hcat(Vec(1,2,3), Vec(4,5,6))) === Mat{3,2}(1,2,3,4,5,6)
        @test @inferred(hcat(Vec(1,2,3)', Vec(4,5,6)')) === Mat{1,6}(1,2,3,4,5,6)
        @test @inferred(hcat(Vec(1,2,3), Mat{3,2}(4,5,6,7,8,9))) === Mat{3,3}(1,2,3,4,5,6,7,8,9)
        @test @inferred(hcat(Mat{3,1}(1,2,3), Mat{3,2}(4,5,6,7,8,9))) === Mat{3,3}(1,2,3,4,5,6,7,8,9)

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
    @testset "reverse" begin
        @test @inferred(reverse(Vec(1, 2, 3))) === Vec(3, 2, 1)
        A = rand(Mat{3,4})
        @test @inferred(reverse(A))::typeof(A) == reverse(reverse(collect(A), dims = 1), dims = 2)
    end
end
