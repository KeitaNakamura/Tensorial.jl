@testset "Broadcast" begin
    for T in (Float32, Float64)
        for TensorType in (Vec{3}, Mat{3,3})
            x = rand(TensorType{T})
            ys = [rand(TensorType{T}) for i in 1:5]
            @test (@inferred x .+ ys)::Vector{<: TensorType{T}} ≈ map(y -> x + y, ys)
            if TensorType <: Vec
                @test (@inferred x .+ (1,2,3))::Vec{3, T} ≈ map(+, x, (1,2,3))
            else
                @test_throws Exception x .+ ntuple(identity, Val(Tensorial.ncomponents(TensorType)))
            end
        end
    end
end
