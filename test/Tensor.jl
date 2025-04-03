@testset "Constructors" begin
    @testset "Inner constructors" begin
        @test (@inferred Tensor{Tuple{2,2}, Int, 2, 4}((1,2,3,4)))::Tensor{Tuple{2,2}, Int, 2, 4} |> Tuple == (1,2,3,4)
        @test (@inferred Tensor{Tuple{2,2}, Float64, 2, 4}((1,2,3,4)))::Tensor{Tuple{2,2}, Float64, 2, 4} |> Tuple == (1.0,2.0,3.0,4.0)
        @test (@inferred Tensor{Tuple{2,2}, Int, 2, 4}((1.0,2,3,4)))::Tensor{Tuple{2,2}, Int, 2, 4} |> Tuple == (1,2,3,4)
        @test (@inferred Tensor{Tuple{2,2}, Float64, 2, 4}((1.0,2,3,4)))::Tensor{Tuple{2,2}, Float64, 2, 4} |> Tuple == (1.0,2.0,3.0,4.0)
        @test (@inferred Tensor{Tuple{2,Symmetry{Tuple{2,2}}}, Int, 3, 6}((1,2,3,4,5,6)))::Tensor{Tuple{2,Symmetry{Tuple{2,2}}}, Int, 3, 6} |> Tuple == (1,2,3,4,5,6)
        @test (@inferred Tensor{Tuple{2,Symmetry{Tuple{2,2}}}, Float64, 3, 6}((1,2,3,4,5,6)))::Tensor{Tuple{2,Symmetry{Tuple{2,2}}}, Float64, 3, 6} |> Tuple == (1.0,2.0,3.0,4.0,5.0,6.0)

        # bad input
        @test_throws Exception Tensor{Tuple{2,2}, Int, 2, 4}((1,2,3))
        @test_throws Exception Tensor{Tuple{2,2}, Int, 2, 4}(())
        @test_throws Exception Tensor{Tuple{2}, Vector{Int}, 1, 2}(([1,2],[3,4]))

        # bad parameters
        @test_throws Exception Tensor{Tuple{Int,2}, Int, 2, 4}((1,2,3,4)) # bad size
        @test_throws Exception Tensor{Tuple{2,2}, Int, 2, 5}((1,2,3,4,5)) # bad ncomponents
        @test_throws Exception Tensor{Tuple{2,2}, Int, 3, 4}((1,2,3,4))   # bad ndims
        @test_throws Exception Tensor{Tuple{2,Symmetry{Int,2}}, Int, 3, 6}((1,2,3,4,5,6)) # bad Symmetry size
        @test_throws Exception Tensor{Tuple{Symmetry{2}}, Int, 1, 2}((1,2)) # bad Symmetry size
        @test_throws Exception Tensor{Tuple{2,Symmetry{2,2}}, Int, 3, 5}((1,2,3,4,5))     # bad ncomponents
        @test_throws Exception Tensor{Tuple{2,Symmetry{2,2}}, Int, 2, 6}((1,2,3,4,5,6))   # bad ndims
    end
    @testset "Outer constructors" begin
        data = (1,2,3,4)
        @testset "From Tuple" begin
            # same type
            @test (@inferred Tensor{Tuple{2,2}, Int, 2}(data))::Tensor{Tuple{2,2}, Int, 2, 4} |> Tuple == (1,2,3,4)
            @test (@inferred Tensor{Tuple{2,2}, Int}(data))::Tensor{Tuple{2,2}, Int, 2, 4} |> Tuple == (1,2,3,4)
            @test (@inferred Tensor{Tuple{2,2}}(data))::Tensor{Tuple{2,2}, Int, 2, 4} |> Tuple == (1,2,3,4)
            # different type
            @test (@inferred Tensor{Tuple{2,2}, Float64, 2}(data))::Tensor{Tuple{2,2}, Float64, 2, 4} |> Tuple == (1.0,2.0,3.0,4.0)
            @test (@inferred Tensor{Tuple{2,2}, Float64}(data))::Tensor{Tuple{2,2}, Float64, 2, 4} |> Tuple == (1.0,2.0,3.0,4.0)
            @test (@inferred Tensor{Tuple{2,2}}((1.0,2,3,4)))::Tensor{Tuple{2,2}, Float64, 2, 4} |> Tuple == (1.0,2.0,3.0,4.0)
        end
        @testset "From Vararg" begin
            # same type
            @test (@inferred Tensor{Tuple{2,2}, Int, 2}(data...))::Tensor{Tuple{2,2}, Int, 2, 4} |> Tuple == (1,2,3,4)
            @test (@inferred Tensor{Tuple{2,2}, Int}(data...))::Tensor{Tuple{2,2}, Int, 2, 4} |> Tuple == (1,2,3,4)
            @test (@inferred Tensor{Tuple{2,2}}(data...))::Tensor{Tuple{2,2}, Int, 2, 4} |> Tuple == (1,2,3,4)
            # different type
            @test (@inferred Tensor{Tuple{2,2}, Float64, 2}(data...))::Tensor{Tuple{2,2}, Float64, 2, 4} |> Tuple == (1.0,2.0,3.0,4.0)
            @test (@inferred Tensor{Tuple{2,2}, Float64}(data...))::Tensor{Tuple{2,2}, Float64, 2, 4} |> Tuple == (1.0,2.0,3.0,4.0)
            @test (@inferred Tensor{Tuple{2,2}}(1.0,2,3,4))::Tensor{Tuple{2,2}, Float64, 2, 4} |> Tuple == (1.0,2.0,3.0,4.0)
        end
        @testset "From Function" begin
            # same type
            @test (@inferred Tensor{Tuple{2,2}, Int, 2}((i,j) -> i))::Tensor{Tuple{2,2}, Int, 2, 4} |> Tuple == (1,2,1,2)
            @test (@inferred Tensor{Tuple{2,2}, Int}((i,j) -> i))::Tensor{Tuple{2,2}, Int, 2, 4} |> Tuple == (1,2,1,2)
            @test (@inferred Tensor{Tuple{2,2}}((i,j) -> i))::Tensor{Tuple{2,2}, Int, 2, 4} |> Tuple == (1,2,1,2)
            # different type
            @test (@inferred Tensor{Tuple{2,2}, Float64, 2}((i,j) -> i))::Tensor{Tuple{2,2}, Float64, 2, 4} |> Tuple == (1.0,2.0,1.0,2.0)
            @test (@inferred Tensor{Tuple{2,2}, Float64}((i,j) -> i))::Tensor{Tuple{2,2}, Float64, 2, 4} |> Tuple == (1.0,2.0,1.0,2.0)
        end
        @testset "From AbstractArray" begin
            A = [1 3; 2 4]
            # same type
            @test (@inferred Tensor{Tuple{2,2}, Int, 2}(A))::Tensor{Tuple{2,2}, Int, 2, 4} |> Tuple == (1,2,3,4)
            @test (@inferred Tensor{Tuple{2,2}, Int}(A))::Tensor{Tuple{2,2}, Int, 2, 4} |> Tuple == (1,2,3,4)
            @test (@inferred Tensor{Tuple{2,2}}(A))::Tensor{Tuple{2,2}, Int, 2, 4} |> Tuple == (1,2,3,4)
            # different type
            @test (@inferred Tensor{Tuple{2,2}, Float64, 2}(A))::Tensor{Tuple{2,2}, Float64, 2, 4} |> Tuple == (1.0,2.0,3.0,4.0)
            @test (@inferred Tensor{Tuple{2,2}, Float64}(A))::Tensor{Tuple{2,2}, Float64, 2, 4} |> Tuple == (1.0,2.0,3.0,4.0)
        end
        @testset "Macro" begin
            a = 2 # check escaping
            @test (@Vec [1,a,3])::Vec{3,Int} == [1,a,3]
            @test (@Vec ones(a))::Vec{a,Float64} == ones(a)
            @test (@Mat [1 a; 3 4])::Mat{2,2,Int} == [1 a; 3 4]
            @test (@Mat ones(a,2))::Mat{a,2,Float64} == ones(a,2)
            @test (@Tensor ones(a,3,3))::Tensor{Tuple{a,3,3},Float64} == ones(a,3,3)
        end
        @testset "zero" begin
            @test (@inferred zero(Tensor{Tuple{2,2}, Int, 2}))::Tensor{Tuple{2,2}, Int, 2, 4} |> Tuple == (0,0,0,0)
            @test (@inferred zero(Tensor{Tuple{2,2}, Int}))::Tensor{Tuple{2,2}, Int, 2, 4} |> Tuple == (0,0,0,0)
            @test (@inferred zero(Tensor{Tuple{2,2}}))::Tensor{Tuple{2,2}, Float64, 2, 4} |> Tuple == (0.0,0.0,0.0,0.0)
            for TensorType in (SecondOrderTensor, FourthOrderTensor,
                               SymmetricSecondOrderTensor, SymmetricFourthOrderTensor,
                               Vec)
                @test all(==(0), (@inferred zero(TensorType{2, Int}))::TensorType{2, Int})
                @test all(==(0), (@inferred zero(TensorType{2}))::TensorType{2, Float64})
            end
            @test all(==(0), (@inferred zero(Mat{2, 3, Int}))::Mat{2, 3, Int})
            @test all(==(0), (@inferred zero(Mat{2, 3}))::Mat{2, 3, Float64})
        end
        @testset "ones" begin
            @test (@inferred ones(Tensor{Tuple{2,2}, Int, 2}))::Tensor{Tuple{2,2}, Int, 2, 4} |> Tuple == (1,1,1,1)
            @test (@inferred ones(Tensor{Tuple{2,2}, Int}))::Tensor{Tuple{2,2}, Int, 2, 4} |> Tuple == (1,1,1,1)
            @test (@inferred ones(Tensor{Tuple{2,2}}))::Tensor{Tuple{2,2}, Float64, 2, 4} |> Tuple == (1.0,1.0,1.0,1.0)
            for TensorType in (SecondOrderTensor, FourthOrderTensor,
                               SymmetricSecondOrderTensor, SymmetricFourthOrderTensor,
                               Vec)
                @test all(==(1), (@inferred ones(TensorType{2, Int}))::TensorType{2, Int})
                @test all(==(1), (@inferred ones(TensorType{2}))::TensorType{2, Float64})
            end
            @test all(==(1), (@inferred ones(Mat{2, 3, Int}))::Mat{2, 3, Int})
            @test all(==(1), (@inferred ones(Mat{2, 3}))::Mat{2, 3, Float64})
        end
        @testset "rand/randn" begin
            for op in (rand, randn)
                @test (@inferred op(Tensor{Tuple{2,2}, Float32, 2}))::Tensor{Tuple{2,2}, Float32, 2, 4} |> unique |> length != 1
                @test (@inferred op(Tensor{Tuple{2,2}, Float32}))::Tensor{Tuple{2,2}, Float32, 2, 4} |> unique |> length != 1
                @test (@inferred op(Tensor{Tuple{2,2}}))::Tensor{Tuple{2,2}, Float64, 2, 4} |> unique |> length != 1
                for TensorType in (SecondOrderTensor, FourthOrderTensor,
                                   SymmetricSecondOrderTensor, SymmetricFourthOrderTensor,
                                   Vec)
                    @test (@inferred op(TensorType{2, Float32}))::TensorType{2, Float32} |> unique |> length != 1
                    @test (@inferred op(TensorType{2}))::TensorType{2, Float64} |> unique |> length != 1
                end
                @test (@inferred op(Mat{2, 3, Float32}))::Mat{2, 3, Float32} |> unique |> length != 1
                @test (@inferred op(Mat{2, 3}))::Mat{2, 3, Float64} |> unique |> length != 1
            end
        end
        @testset "aliases" begin
            for T in (Float32, Float64), dim in 1:3
                for TT in (SecondOrderTensor{dim, T},
                           FourthOrderTensor{dim, T},
                           SymmetricSecondOrderTensor{dim, T},
                           SymmetricFourthOrderTensor{dim, T},
                           Mat{dim, dim, T},
                           Vec{dim, T})
                    data = ntuple(i -> T(1), Tensorial.ncomponents(Space(TT)))
                    @test (@inferred TT(data))::TT |> Tuple == data
                    @test (@inferred TT(data...))::TT |> Tuple == data
                end
            end
            for T in (Float32, Float64)
                x = rand(Vec{3, T})
                @test Vec(x) ≈ x
            end
        end
        @testset "Identity tensor" begin
            # second order tensor
            for TensorType in (SecondOrderTensor, SymmetricSecondOrderTensor)
                for dim in 1:3
                    v = rand(Vec{dim})
                    @test (@inferred one(TensorType{dim, Int}))::TensorType{dim, Int} * v ≈ v
                    @test (@inferred one(TensorType{dim}))::TensorType{dim, Float64} * v ≈ v
                end
            end
            # fourth order tensor
            for dim in 1:3
                I = one(SecondOrderTensor{dim})
                A = rand(SecondOrderTensor{dim})
                As = rand(SymmetricSecondOrderTensor{dim})
                II = (@inferred one(FourthOrderTensor{dim, Float32}))::FourthOrderTensor{dim, Float32}
                II = (@inferred one(FourthOrderTensor{dim}))::FourthOrderTensor{dim, Float64}
                @test (II ⊡₂ A)::SecondOrderTensor{dim} ≈ A
                @test (II ⊡₂ As)::SecondOrderTensor{dim} ≈ As
                IIs = (@inferred one(SymmetricFourthOrderTensor{dim, Float32}))::SymmetricFourthOrderTensor{dim, Float32}
                IIs = (@inferred one(SymmetricFourthOrderTensor{dim}))::SymmetricFourthOrderTensor{dim, Float64}
                @test (IIs ⊡₂ A)::SymmetricSecondOrderTensor{dim} ≈ (A+A')/2
                @test (IIs ⊡₂ As)::SymmetricSecondOrderTensor{dim} ≈ As
            end
            # higher order tensor
            for T in (Float32, Float64)
                for (TT, val) in ((Tensor{NTuple{2, @Symmetry{2,2,2}}}, Val(3)),
                                  (Tensor{Tuple{2,@Symmetry{2,2}, 2,@Symmetry{2,2}}}, Val(3)),
                                  (Tensor{Tuple{@Symmetry{2,2},2, @Symmetry{2,2},2}}, Val(3)),
                                  (Tensor{NTuple{2, @Symmetry{2,2,2,2}}}, Val(4)),
                                  (Tensor{Tuple{@Symmetry{2,2},@Symmetry{2,2}, @Symmetry{2,2},@Symmetry{2,2}}}, Val(4)))
                    @test eltype(one(TT)) == eltype(one(TT{Float64}))
                    @test eltype(one(TT{T})) == T
                    I = one(TT{T})
                    A = rand(TT{T})
                    @test contract(A, I, val) ≈ contract(I, A, val) ≈ A
                    @test contract(I, I, val) ≈ contract(I, contract(I, I, val), val) ≈ I
                    @test inv(I) ≈ I
                end
            end
        end
        @testset "UniformScaling" begin
            for T in (Float32, Float64)
                @test (@inferred Mat{2, 2}(I))::Mat{2, 2, Bool} == Matrix(I, 2, 2)
                @test (@inferred Mat{2, 2}(2I))::Mat{2, 2, Int} == Matrix(2I, 2, 2)
                @test (@inferred Mat{2, 2, T}(2I))::Mat{2, 2, T} == Matrix(2I, 2, 2)
                @test (@inferred Mat{2, 3, T}(2I))::Mat{2, 3, T} == Matrix(2I, 2, 3)
            end
        end
        @testset "Levi-Civita tensor" begin
            for N in 2:4
                ϵ = (@inferred levicivita(Val(N)))::Tensor{NTuple{N, N}, Int}
                for I in CartesianIndices(ϵ)
                    @test ϵ[I] == Combinatorics.levicivita(collect(Tuple(I)))
                end
            end
        end
    end
end

@testset "Symmetric tensors" begin
    x = rand(Tensor{Tuple{2, @Symmetry{2,2}, 3}})
    @test Tensorial.ncomponents(x) == 18
    for i in axes(x, 1), l in axes(x, 4)
        for j in axes(x, 2), k in axes(x, 3)
            @test x[i,j,k,l] == x[i,k,j,l]
        end
    end
    x = rand(Tensor{Tuple{2, @Symmetry{3,3,3}}})
    @test Tensorial.ncomponents(x) == 2 * Tensorial.ncomponents(@Symmetry{3,3,3}())
    for i in axes(x, 1)
        for j in axes(x, 2), k in axes(x, 3), l in axes(x, 4)
            @test x[i,j,k,l] == x[i,k,j,l] == x[i,j,l,k] == x[i,l,k,j] == x[i,k,l,j] == x[i,l,j,k]
        end
    end
    # totally symmetric third-order tensor
    # https://arxiv.org/pdf/2009.10752.pdf
    x = rand(Tensor{Tuple{@Symmetry{3,3,3}}})
    A = zeros(3,3,3)
    for I in CartesianIndices(x)
        i, j, k = Tuple(I)
        A[i,j,k] = (x[i,j,k] + x[j,k,i] + x[k,i,j] + x[j,i,k] + x[k,j,i] + x[i,k,j]) / 6
    end
    @test x ≈ A
end

@testset "Indices" begin
    x = rand(Tensor{Tuple{2, @Symmetry{2,2}, 3, @Symmetry{2,2}}})
    n = Tensorial.ncomponents(x)
    @test (@inferred Tensorial.tupleindices_tensor(x))::SArray{Tuple{size(x)...}, Int} |> unique == 1:n
    inds = (@inferred Tensorial.tensorindices_tuple(x))::SVector{n, Int}
    @test x[inds] == unique(x[inds])
    dups = (@inferred Tensorial.nduplicates_tuple(x))::SVector{n, Int}
    for i in eachindex(inds)
        v = x[inds[i]]
        @test count(==(v), x) == dups[i]
    end
end

@testset "Conversion" begin
    @testset "Tensor -> Tensor" begin
        A = Tensor{Tuple{3,3}}(1:9...)
        S = Tensor{Tuple{@Symmetry{3,3}}}(1:6...)
        v = Vec(1:3...)
        for T in (Float32, Float64)
            # convert eltype
            Adata = map(T, (1:9...,))
            @test (@inferred convert(Tensor{Tuple{3,3}, T}, A))::Tensor{Tuple{3,3}, T} |> Tuple == Adata
            @test (@inferred convert(SecondOrderTensor{3, T}, A))::Tensor{Tuple{3,3}, T} |> Tuple == Adata
            @test (@inferred convert(Mat{3, 3, T}, A))::Tensor{Tuple{3,3}, T} |> Tuple == Adata
            Sdata = map(T, (1:6...,))
            @test (@inferred convert(Tensor{Tuple{@Symmetry{3,3}}, T}, S))::Tensor{Tuple{@Symmetry{3,3}}, T} |> Tuple == Sdata
            @test (@inferred convert(SymmetricSecondOrderTensor{3, T}, S))::Tensor{Tuple{@Symmetry{3,3}}, T} |> Tuple == Sdata
            vdata = map(T, (1:3...,))
            @test (@inferred convert(Vec{3, T}, v))::Vec{3, T} |> Tuple == vdata
            # convert symmetric tensor to tensor
            @test (@inferred convert(Tensor{Tuple{3, 3}, T}, S))::Tensor{Tuple{3, 3}, T} == Array(S)
            @test (@inferred convert(Mat{3, 3, T}, S))::Tensor{Tuple{3, 3}, T} == Array(S)
            # convert tensor to symmetric tensor
            @test (@inferred convert(Tensor{Tuple{@Symmetry{3, 3}}, T}, A+A')) == Array(A+A')
            @test_throws InexactError convert(Tensor{Tuple{@Symmetry{3, 3}}, T}, A)
        end
        # Float64 -> Float32
        x = rand(Mat{3,3,Float64})
        (@inferred convert(SymmetricSecondOrderTensor{3,Float32}, x+x'))::SymmetricSecondOrderTensor{3,Float32}
    end
    @testset "AbstractArray -> Tensor" begin
        A = [1 3; 2 4]
        v = [1, 2]
        for T in (Float32, Float64)
            Adata = map(T, (1,2,3,4))
            @test (@inferred convert(Tensor{Tuple{2,2}, T}, A))::Tensor{Tuple{2,2}, T} |> Tuple == Adata
            @test (@inferred convert(SecondOrderTensor{2, T}, A))::Tensor{Tuple{2,2}, T} |> Tuple == Adata
            @test (@inferred convert(Mat{2, 2, T}, A))::Tensor{Tuple{2,2}, T} |> Tuple == Adata
            @test_throws InexactError convert(Tensor{Tuple{@Symmetry{2,2}}, T}, A)
            @test_throws InexactError convert(SymmetricSecondOrderTensor{2, T}, A)
            vdata = map(T, (1,2))
            @test (@inferred convert(Vec{2, T}, v))::Vec{2, T} |> Tuple == vdata
        end
    end
    @testset "Tuple -> Tensor" begin
        for T in (Float32, Float64)
            Adata = map(T, (1,2,3,4))
            @test (@inferred convert(Vec{4, T}, Adata))::Vec{4, T} |> Tuple == Adata
            @test (@inferred convert(Vec{4}, Adata))::Vec{4, T} |> Tuple == Adata
            @test (@inferred convert(Mat{2, 2, T}, Adata))::Mat{2, 2, T} |> Tuple == Adata
            @test (@inferred convert(Mat{2, 2}, Adata))::Mat{2, 2, T} |> Tuple == Adata
        end
    end
end

@testset "Promotion" begin
    @test (@inferred promote_rule(Vec{2, Float64}, Vec{2, Float32})) === Vec{2, Float64}
    @test (@inferred promote_rule(Mat{2, 2, Int, 4}, Mat{2, 2, Float32, 4})) === Mat{2, 2, Float32, 4}
    x = Vec(1,2,3)
    for T in (Float32, Float64)
        res = (@inferred Tensorial.promote_elements(x, 3, rand(Mat{3,3,T})))
        @test promote_type(map(eltype, res)...) == T
    end
end

@testset "Tensor misc" begin
    # size
    TT = Tensor{Tuple{2, 3}}
    @test (@inferred size(TT))::Tuple{Int, Int} == (2,3)
    TT = Tensor{Tuple{2, 3, @Symmetry{3, 3}}}
    @test (@inferred size(TT))::Tuple{Int, Int, Int, Int} == (2,3,3,3)
    # macro
    @test @Tensor({Tuple{3,3}}) == Tensor{Tuple{3,3},T,2,9} where {T}
    @test @Tensor({Tuple{3,3},Float64}) == Tensor{Tuple{3,3},Float64,2,9}
    @test @Tensor({Tuple{@Symmetry{3,3},2}}) == Tensor{Tuple{Symmetry{Tuple{3,3}},2},T,3,12} where {T}
    @test @Tensor({Tuple{@Symmetry{3,3},2},Int}) == Tensor{Tuple{Symmetry{Tuple{3,3}},2},Int,3,12}
end
