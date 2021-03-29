@testset "Broadcast" begin
    for T in (Float32, Float64)
        for TensorType in (Vec{3}, Mat{3,3})
            x = rand(TensorType{T})
            ys = [rand(TensorType{T}) for i in 1:5]
            for op in (+, -, *, /)
                if (op == +) || (op == -)
                    @test (@inferred broadcast(op, x, ys))::Vector{<: TensorType{T}} ≈ map(y -> op(x, y), ys)
                    @test (@. op(op(x, ys), op(x, x)))::Vector{<: TensorType{T}} ≈ map(y -> op(op(x, y), op(x, x)), ys)
                end
                if TensorType <: Vec
                    @test (@inferred broadcast(op, x, (1,2,3)))::Vec{3, T} ≈ map(op, x, (1,2,3))
                else
                    @test_throws Exception broadcast(op, x, ntuple(identity, Val(Tensorial.ncomponents(TensorType))))
                end
            end
        end
    end
end
