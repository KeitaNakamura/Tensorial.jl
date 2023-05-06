@testset "Eigen" begin
    for T in (Float32, Float64)
        for dim in (2, 3)
            x = rand(SymmetricSecondOrderTensor{dim, T})
            @test (@inferred eigvals(x)) ≈ eigvals(Array(x))
            @test (@inferred eigen(x)).values ≈ eigen(Array(x)).values
            @test (@inferred eigen(x)).vectors ≈ (@inferred eigvecs(x))
        end
    end
    # copyied from StaticArrays.jl
    @testset "3x3 degenerate cases" begin
        # Rank 1
        v = randn(Vec{3,Float64})
        m = symmetric(v ⊗ v, :U)
        vv = sum(abs2, v)
        vals, vecs = eigen(m)::Eigen{<:Any,<:Any,<:Mat,<:Vec}

        @test vecs' ⋅ vecs ≈ one(Mat{3,3,Float64})
        @test vals ≈ Vec(0.0, 0.0, vv)
        @test eigvals(m) ≈ vals

        # Rank 2
        v2 = randn(Vec{3,Float64})
        v2 -= dot(v,v2)*v/(vv)
        v2v2 = sum(abs2, v2)
        m += symmetric(v2 ⊗ v2, :U)
        vals, vecs = eigen(m)::Eigen{<:Any,<:Any,<:Mat,<:Vec}

        @test vecs' ⋅ vecs ≈ one(Mat{3,3,Float64})
        if vv < v2v2
            @test vals ≈ Vec(0.0, vv, v2v2)
        else
            @test vals ≈ Vec(0.0, v2v2, vv)
        end
        @test eigvals(m) ≈ vals

        # Degeneracy (2 large)
        m = symmetric(-99*(v ⊗ v)/vv + 100*one(Mat{3,3,Float64}), :U)
        vals, vecs = eigen(m)::Eigen{<:Any,<:Any,<:Mat,<:Vec}

        @test vecs' ⋅ vecs ≈ one(Mat{3,3,Float64})
        @test vals ≈ Vec(1.0, 100.0, 100.0)
        @test eigvals(m) ≈ vals

        # Degeneracy (2 small)
        m = symmetric((v ⊗ v)/vv + 1e-2*one(Mat{3,3,Float64}), :U)
        vals, vecs = eigen(m)::Eigen{<:Any,<:Any,<:Mat,<:Vec}

        @test vecs' ⋅ vecs ≈ one(Mat{3,3,Float64})
        @test vals ≈ Vec(1e-2, 1e-2, 1.01)
        @test eigvals(m) ≈ vals

        # Block diagonal
        m = symmetric(@Mat([1.0 0.0 0.0
                            0.0 1.0 1.0
                            0.0 1.0 1.0]), :U)
        vals, vecs = eigen(m)::Eigen{<:Any,<:Any,<:Mat,<:Vec}

        @test vals ≈ [0.0, 1.0, 2.0]
        @test vecs⋅diagm(Val(0) => vals)⋅vecs' ≈ m
        @test eigvals(m) ≈ vals

        m = symmetric(@Mat([1.0 0.0 1.0
                            0.0 1.0 0.0
                            1.0 0.0 1.0]), :U)
        vals, vecs = eigen(m)::Eigen{<:Any,<:Any,<:Mat,<:Vec}

        @test vals ≈ [0.0, 1.0, 2.0]
        @test vecs⋅diagm(Val(0) => vals)⋅vecs' ≈ m
        @test eigvals(m) ≈ vals

        m = symmetric(@Mat([1.0 1.0 0.0
                            1.0 1.0 0.0
                            0.0 0.0 1.0]), :U)
        vals, vecs = eigen(m)::Eigen{<:Any,<:Any,<:Mat,<:Vec}

        @test vals ≈ [0.0, 1.0, 2.0]
        @test vecs⋅diagm(Val(0) => vals)⋅vecs' ≈ m
        @test eigvals(m) ≈ vals
    end
end
