function check_value_and_type(x, y, z)
    @test x ≈ y ≈ z
    @test typeof(x) == typeof(y)
end

@testset "Einstein summation" begin
    @testset "no dummy indices" begin
        x = rand(Tensorial.tensortype(Space(4, Symmetry(2, 2))))

        # check type instability
        f(x) = @einsum (1,2,3) -> x[3,2,1]
        @inferred(f(x))

        # permutedims
        check_value_and_type((@einsum (1,2,3) -> x[1,2,3]), permutedims(x, Val((1,2,3))), (@tensor t[1,2,3] := Array(x)[1,2,3]))
        check_value_and_type((@einsum (1,3,2) -> x[1,2,3]), permutedims(x, Val((1,3,2))), (@tensor t[1,3,2] := Array(x)[1,2,3]))
        check_value_and_type((@einsum (2,1,3) -> x[1,2,3]), permutedims(x, Val((2,1,3))), (@tensor t[2,1,3] := Array(x)[1,2,3]))
        check_value_and_type((@einsum (2,3,1) -> x[1,2,3]), permutedims(x, Val((2,3,1))), (@tensor t[2,3,1] := Array(x)[1,2,3]))
        check_value_and_type((@einsum (3,1,2) -> x[1,2,3]), permutedims(x, Val((3,1,2))), (@tensor t[3,1,2] := Array(x)[1,2,3]))
        check_value_and_type((@einsum (3,2,1) -> x[1,2,3]), permutedims(x, Val((3,2,1))), (@tensor t[3,2,1] := Array(x)[1,2,3]))

        v1 = rand(Vec{3})
        v2 = rand(Vec{3})
        check_value_and_type((@einsum (i,j) -> v1[i] * v2[j]), v1 ⊗ v2, (@tensor t[i,j] := Array(v1)[i] * Array(v2)[j]))
        check_value_and_type((@einsum (j,i) -> v1[i] * v2[j]), v2 ⊗ v1, (@tensor t[j,i] := Array(v1)[i] * Array(v2)[j]))
        check_value_and_type((@einsum v1[i] * v2[j]), v1 ⊗ v2, (@tensor t[i,j] := Array(v1)[i] * Array(v2)[j]))

        S1 = rand(SymmetricSecondOrderTensor{3})
        S2 = rand(SecondOrderTensor{3})
        check_value_and_type((@einsum (i,j,k,l) -> S1[i,j] * S2[k,l]), S1 ⊗ S2, (@tensor t[i,j,k,l] := Array(S1)[i,j] * Array(S2)[k,l]))
        check_value_and_type((@einsum (i,k,j,l) -> S1[i,j] * S2[k,l]), permutedims(S1 ⊗ S2, Val((1,3,2,4))), (@tensor t[i,k,j,l] := Array(S1)[i,j] * Array(S2)[k,l]))
        check_value_and_type((@einsum S1[i,j] * S2[k,l]), S1 ⊗ S2, (@tensor t[i,j,k,l] := Array(S1)[i,j] * Array(S2)[k,l]))
    end
    @testset "no free indices" begin
        v1 = rand(Vec{3})
        v2 = rand(Vec{3})
        check_value_and_type((@einsum () -> v1[i] * v2[i]), v1 ⋅ v2, only(@tensor t[] := Array(v1)[i] * Array(v2)[i]))
        check_value_and_type((@einsum v1[i] * v2[i]), v1 ⋅ v2, only(@tensor t[] := Array(v1)[i] * Array(v2)[i]))

        S1 = rand(SymmetricSecondOrderTensor{3})
        S2 = rand(SecondOrderTensor{3})
        check_value_and_type((@einsum () -> S1[i,j] * S2[i,j]), S1 ⊡ S2, only(@tensor t[] := Array(S1)[i,j] * Array(S2)[i,j]))
        check_value_and_type((@einsum () -> S1[i,j] * S2[j,i]), S1 ⊡ S2', only(@tensor t[] := Array(S1)[i,j] * Array(S2)[j,i]))
        check_value_and_type((@einsum S1[i,j] * S2[i,j]), S1 ⊡ S2, only(@tensor t[] := Array(S1)[i,j] * Array(S2)[i,j]))
        check_value_and_type((@einsum S1[i,j] * S2[j,i]), S1 ⊡ S2', only(@tensor t[] := Array(S1)[i,j] * Array(S2)[j,i]))
        check_value_and_type((@einsum S1[i,i]), tr(S1), only(@tensor t[] := Array(S1)[i,i]))
        check_value_and_type((@einsum S1[i,i]/3), mean(S1), only(@tensor t[] := Array(S1)[i,i]/3))
        check_value_and_type((@einsum S1[i,i]/S2[j,j]), tr(S1)/tr(S2), tr(Array(S1))/tr(Array(S2))) # not allowed in TensorOperations
    end
    @testset "mixed" begin
        S1 = rand(SymmetricSecondOrderTensor{3})
        S2 = rand(SecondOrderTensor{3})
        check_value_and_type((@einsum (i,j) -> S1[i,k] * S2[k,j]), S1 ⋅ S2, (@tensor t[i,j] := Array(S1)[i,k] * Array(S2)[k,j]))
        check_value_and_type((@einsum (i,j) -> S1[i,k] * S2[j,k]), S1 ⋅ S2', (@tensor t[i,j] := Array(S1)[i,k] * Array(S2)[j,k]))
        check_value_and_type((@einsum S1[i,k] * S2[k,j]), S1 ⋅ S2, (@tensor t[i,j] := Array(S1)[i,k] * Array(S2)[k,j]))
        check_value_and_type((@einsum (i,j) -> S1[i,k] * S2[j,k] + S1[j,k] * S2[i,k]), S1 ⋅ S2' + S2 ⋅ S1', (@tensor t[i,j] := Array(S1)[i,k] * Array(S2)[j,k] + Array(S1)[j,k] * Array(S2)[i,k]))
        check_value_and_type((@einsum (i,j) -> S1[i,k] * (S2[j,k] + S1[k,j] * S2[i,i])), S1 ⋅ (S2' + S1 * tr(S2)), (@tensor t[i,j] := Array(S1)[i,k] * (Array(S2)[j,k] + Array(S1)[k,j] * Array(S2)[i,i])))

        S3 = rand(Tensor{Tuple{@Symmetry{3,3,3}}})
        v1 = rand(Vec{3})
        check_value_and_type((@einsum (i,j) -> S3[j,k,i] * v1[k]), permutedims(S3, Val((3,1,2))) ⋅ v1, (@tensor t[i,j] := Array(S3)[j,k,i] * Array(v1)[k]))
        check_value_and_type((@einsum (j) -> v1[i] * S3[j,k,i] * v1[k]), (S3 ⋅ v1) ⋅ v1, (@tensor t[j] := Array(v1)[i] * Array(S3)[j,k,i] * Array(v1)[k]))
        check_value_and_type((@einsum v1[i] * S3[j,k,i] * v1[k]), (S3 ⋅ v1) ⋅ v1, (@tensor t[j] := Array(v1)[i] * Array(S3)[j,k,i] * Array(v1)[k]))

        # unary operator (#201)
        a = Vec(1,0,0)
        b = Vec(0,1,0)
        check_value_and_type((@einsum -a[μ]*a[v] + b[μ]*b[v]), -a ⊗ a + b ⊗ b, (@tensor t[μ,v] := -Array(a)[μ] * Array(a)[v] + Array(b)[μ] * Array(b)[v]))
        check_value_and_type((@einsum b[μ]*b[v] + -a[μ]*a[v]), b ⊗ b + -a ⊗ a, (@tensor t[μ,v] := Array(b)[μ] * Array(b)[v] + -Array(a)[μ] * Array(a)[v]))
    end
    @testset "type annotation" begin
        A = rand(SecondOrderTensor{4})
        B = rand(Tensor{Tuple{4,@Symmetry{4,4}}})
        ans = @einsum A[σp,σ]*A[μp,μ]*A[νp,ν]*B[σp,μp,νp]
        @test (@einsum Tensor{Tuple{4,@Symmetry{4,4}}, Float64} A[σp,σ]*A[μp,μ]*A[νp,ν]*B[σp,μp,νp])::Tensor{Tuple{4,@Symmetry{4,4}}, Float64} ≈ ans
        @test (@einsum Tensor{Tuple{4,@Symmetry{4,4}}, Float32} A[σp,σ]*A[μp,μ]*A[νp,ν]*B[σp,μp,νp])::Tensor{Tuple{4,@Symmetry{4,4}}, Float32} ≈ ans
    end
    @testset "errors" begin
        S1 = rand(SymmetricSecondOrderTensor{3})
        v = rand(Vec{2})
        @test_throws Exception (@einsum (i) -> S1[i,j] * v[j])
        # @test_throws Exception (@einsum (i,j) -> S1[i,j] + S1[i,j])
        # @test_throws Exception (@einsum (i,j) -> S1[i,j] - S1[i,j])
        # @test_throws Exception (@einsum (i,j) -> S1[i,j] * S1[i,j])
        # @test_throws Exception (@einsum (k) -> S1[i,j] * S1[i,j])
        # @test_throws Exception (@einsum (j) -> S1[i,i] * S1[i,j])
        # @test_throws Exception (@einsum S1[i,i] * S1[i,j])
    end
end
