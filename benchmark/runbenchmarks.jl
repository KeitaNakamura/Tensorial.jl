using Tensorial
using BenchmarkTools
using InteractiveUtils # for `versioninfo`
using Markdown

const run_array = true

const suite = BenchmarkGroup()
suite["Array"] = BenchmarkGroup()
suite["Tensor"] = BenchmarkGroup()

const tensors = let
    dict = Dict{Tuple{Space, DataType}, Tensor}()
    for S in (Space(3),
              Space(3,3),
              Space(Symmetry(3,3)),
              Space(3,3,3),
              Space(Symmetry(3,3,3)),
              Space(3,3,3,3),
              Space(Symmetry(3,3),Symmetry(3,3)))
        for T in (Float32, Float64)
            dict[(S, T)] = rand(Tensorial.tensortype(S){T})
        end
    end
    dict
end

include("ops.jl")

tune!(suite)
const results = run(suite)

let path = "../docs/src/Benchmarks.md"

    path = joinpath(dirname(@__FILE__), path)

    open(path, "w") do file

        function printheader(head)
            println(file, "| **$(head)** | | | |")
        end
        function printrow(op, tt, ta)
            pretty = (t) -> BenchmarkTools.prettytime(BenchmarkTools.time(minimum(t)))
            speedup = (ta, tt) -> round(10*BenchmarkTools.time(minimum(ta))/BenchmarkTools.time(minimum(tt)))/10
            println(file, "| `$(op)` | $(pretty(tt)) | $(pretty(ta)) | ×$(speedup(ta, tt)) |")
        end

        print(file,
              """
              # Benchmarks

              ```julia
              a = rand(Vec{3})
              A = rand(SecondOrderTensor{3})
              S = rand(SymmetricSecondOrderTensor{3})
              B = rand(Tensor{Tuple{3,3,3}})
              AA = rand(FourthOrderTensor{3})
              SS = rand(SymmetricFourthOrderTensor{3})
              ```

              | Operation  | `Tensor` | `Array` | speed-up |
              |:-----------|---------:|--------:|---------:|
              """)

        printheader("Single contraction")
        printrow("a ⋅ a", results["Tensor"]["Space(3) ⋅ Space(3)"],
                          results["Array" ]["Space(3) ⋅ Space(3)"])
        printrow("A ⋅ a", results["Tensor"]["Space(3,3) ⋅ Space(3)"],
                          results["Array" ]["Space(3,3) ⋅ Space(3)"])
        printrow("S ⋅ a", results["Tensor"]["Space(Symmetry(3,3)) ⋅ Space(3)"],
                          results["Array" ]["Space(Symmetry(3,3)) ⋅ Space(3)"])

        printheader("Double contraction")
        printrow("A ⊡ A", results["Tensor"]["Space(3,3) ⊡ Space(3,3)"],
                          results["Array" ]["Space(3,3) ⊡ Space(3,3)"])
        printrow("S ⊡ S", results["Tensor"]["Space(Symmetry(3,3)) ⊡ Space(Symmetry(3,3))"],
                          results["Array" ]["Space(Symmetry(3,3)) ⊡ Space(Symmetry(3,3))"])
        printrow("B ⊡ A", results["Tensor"]["Space(3,3,3) ⊡ Space(3,3)"],
                          results["Array" ]["Space(3,3,3) ⊡ Space(3,3)"])
        printrow("AA ⊡ A", results["Tensor"]["Space(3,3,3,3) ⊡ Space(3,3)"],
                           results["Array" ]["Space(3,3,3,3) ⊡ Space(3,3)"])
        printrow("SS ⊡ S", results["Tensor"]["Space(Symmetry(3,3),Symmetry(3,3)) ⊡ Space(Symmetry(3,3))"],
                           results["Array" ]["Space(Symmetry(3,3),Symmetry(3,3)) ⊡ Space(Symmetry(3,3))"])

        printheader("Tensor product")
        printrow("a ⊗ a", results["Tensor"]["Space(3) ⊗ Space(3)"],
                          results["Array" ]["Space(3) ⊗ Space(3)"])

        printheader("Determinant and Inverse")
        printrow("det(A)", results["Tensor"]["det(Space(3,3))"],
                           results["Array" ]["det(Space(3,3))"])
        printrow("det(S)", results["Tensor"]["det(Space(Symmetry(3,3)))"],
                           results["Array" ]["det(Space(Symmetry(3,3)))"])
        printrow("inv(A)", results["Tensor"]["inv(Space(3,3))"],
                           results["Array" ]["inv(Space(3,3))"])
        printrow("inv(S)", results["Tensor"]["inv(Space(Symmetry(3,3)))"],
                           results["Array" ]["inv(Space(Symmetry(3,3)))"])

        # print versioninfo
        println(file,
                """

                The benchmarks are generated by
                [`runbenchmarks.jl`](https://github.com/KeitaNakamura/Tensorial.jl/blob/master/benchmark/runbenchmarks.jl)
                on the following system:

                ```julia
                julia> versioninfo()
                """)
        versioninfo(file)
        println(file, "```")
    end

    # display results
    open(path, "r") do file
        display(Markdown.parse(file))
    end
end
