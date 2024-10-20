using Tensorial
using BenchmarkTools
using InteractiveUtils # for `versioninfo`
using Markdown

using Tensorial: Space

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
        function printrow(op, tt, ta, ts)
            pretty = (t) -> BenchmarkTools.prettytime(BenchmarkTools.time(minimum(t)))
            speedup = (ta, tt) -> round(BenchmarkTools.time(minimum(ta))/BenchmarkTools.time(minimum(tt)); sigdigits=2)
            println(file, "| `$(op)` | $(pretty(tt)) | $(pretty(ta)) | ×$(speedup(ta, tt)) | $(pretty(ts)) | ×$(speedup(ts, tt))")
        end

        print(file,
              """
              # Benchmarks

              The performance for some typical operators is summarized below.
              For fourth-order tensors, both `Array` and `SArray` use the classical [Voigt form](https://en.wikipedia.org/wiki/Voigt_notation)
              to correctly handle symmetries.
              The benchmakrs show that `Tensor` offers performance comparable to `SArray` without the hassle of using the Voigt form.

              ```julia
              a = rand(Vec{3})
              A = rand(SecondOrderTensor{3})
              S = rand(SymmetricSecondOrderTensor{3})
              AA = rand(FourthOrderTensor{3})
              SS = rand(SymmetricFourthOrderTensor{3})
              ```

              | Operation  | `Tensor` | `Array` | Speedup | `SArray` | Speedup |
              |:-----------|---------:|--------:|--------:|---------:|--------:|
              """)

        printheader("Single contraction")
        printrow("a ⋅ a", results["Tensor"]["Space(3) ⋅ Space(3)"],
                          results["Array" ]["Space(3) ⋅ Space(3)"],
                          results["SArray"]["Space(3) ⋅ Space(3)"])
        printrow("A ⋅ a", results["Tensor"]["Space(3,3) ⋅ Space(3)"],
                          results["Array" ]["Space(3,3) ⋅ Space(3)"],
                          results["SArray"]["Space(3,3) ⋅ Space(3)"])
        printrow("S ⋅ a", results["Tensor"]["Space(Symmetry(3,3)) ⋅ Space(3)"],
                          results["Array" ]["Space(Symmetry(3,3)) ⋅ Space(3)"],
                          results["SArray"]["Space(Symmetry(3,3)) ⋅ Space(3)"])

        printheader("Double contraction")
        printrow("A ⊡ A", results["Tensor"]["Space(3,3) ⊡ Space(3,3)"],
                          results["Array" ]["Space(3,3) ⊡ Space(3,3)"],
                          results["SArray"]["Space(3,3) ⊡ Space(3,3)"])
        printrow("S ⊡ S", results["Tensor"]["Space(Symmetry(3,3)) ⊡ Space(Symmetry(3,3))"],
                          results["Array" ]["Space(Symmetry(3,3)) ⊡ Space(Symmetry(3,3))"],
                          results["SArray"]["Space(Symmetry(3,3)) ⊡ Space(Symmetry(3,3))"])
        printrow("AA ⊡ A", results["Tensor"]["Space(3,3,3,3) ⊡ Space(3,3)"],
                           results["Array" ]["Space(3,3,3,3) ⊡ Space(3,3)"],
                           results["SArray"]["Space(3,3,3,3) ⊡ Space(3,3)"])
        printrow("SS ⊡ S", results["Tensor"]["Space(Symmetry(3,3),Symmetry(3,3)) ⊡ Space(Symmetry(3,3))"],
                           results["Array" ]["Space(Symmetry(3,3),Symmetry(3,3)) ⊡ Space(Symmetry(3,3))"],
                           results["SArray"]["Space(Symmetry(3,3),Symmetry(3,3)) ⊡ Space(Symmetry(3,3))"])

        printheader("Tensor product")
        printrow("a ⊗ a", results["Tensor"]["Space(3) ⊗ Space(3)"],
                          results["Array" ]["Space(3) ⊗ Space(3)"],
                          results["SArray"]["Space(3) ⊗ Space(3)"])

        printheader("Cross product")
        printrow("a × a", results["Tensor"]["Space(3) ⊗ Space(3)"],
                          results["Array" ]["Space(3) ⊗ Space(3)"],
                          results["SArray"]["Space(3) ⊗ Space(3)"])

        printheader("Determinant")
        printrow("det(A)", results["Tensor"]["det(Space(3,3))"],
                           results["Array" ]["det(Space(3,3))"],
                           results["SArray"]["det(Space(3,3))"])
        printrow("det(S)", results["Tensor"]["det(Space(Symmetry(3,3)))"],
                           results["Array" ]["det(Space(Symmetry(3,3)))"],
                           results["SArray"]["det(Space(Symmetry(3,3)))"])

        printheader("Inverse")
        printrow("inv(A)", results["Tensor"]["inv(Space(3,3))"],
                           results["Array" ]["inv(Space(3,3))"],
                           results["SArray"]["inv(Space(3,3))"])
        printrow("inv(S)", results["Tensor"]["inv(Space(Symmetry(3,3)))"],
                           results["Array" ]["inv(Space(Symmetry(3,3)))"],
                           results["SArray"]["inv(Space(Symmetry(3,3)))"])
        printrow("inv(AA)", results["Tensor"]["inv(Space(3,3,3,3))"],
                            results["Array" ]["inv(Space(3,3,3,3))"],
                            results["SArray"]["inv(Space(3,3,3,3))"])
        printrow("inv(SS)", results["Tensor"]["inv(Space(Symmetry(3,3),Symmetry(3,3)))"],
                            results["Array" ]["inv(Space(Symmetry(3,3),Symmetry(3,3)))"],
                            results["SArray"]["inv(Space(Symmetry(3,3),Symmetry(3,3)))"])

        # print versioninfo
        print(file,
              """

              The benchmarks are generated by
              [`runbenchmarks.jl`](https://github.com/KeitaNakamura/Tensorial.jl/blob/master/benchmark/runbenchmarks.jl)
              on the following system:

              ```julia
              julia> versioninfo()
              $((x = IOBuffer(); versioninfo(x); String(take!(x))))
              """)
        println(file, "```")
    end

    # display results
    open(path, "r") do file
        display(Markdown.parse(file))
    end
end
