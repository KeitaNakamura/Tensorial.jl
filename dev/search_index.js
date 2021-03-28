var documenterSearchIndex = {"docs":
[{"location":"Quaternion/","page":"Quaternion","title":"Quaternion","text":"DocTestSetup = :(using Tensorial)","category":"page"},{"location":"Quaternion/#Quaternion","page":"Quaternion","title":"Quaternion","text":"","category":"section"},{"location":"Quaternion/","page":"Quaternion","title":"Quaternion","text":"Order = [:type, :function]\nPages = [\"Quaternion.md\"]","category":"page"},{"location":"Quaternion/","page":"Quaternion","title":"Quaternion","text":"Modules = [Tensorial]\nOrder   = [:type, :function]\nPages   = [\"quaternion.jl\"]","category":"page"},{"location":"Quaternion/#Tensorial.Quaternion","page":"Quaternion","title":"Tensorial.Quaternion","text":"Quaternion represents q_1 + q_2 bmi + q_3 bmj + q_4 bmk. The salar part and vector part can be accessed by q.scalar and q.vector, respectively.\n\nnote: Note\nQuaternion is experimental and could change or disappear in future versions of Tensorial.\n\n\n\n\n\n","category":"type"},{"location":"Quaternion/#Tensorial.quaternion-Tuple{Real, Vec{3, T} where T}","page":"Quaternion","title":"Tensorial.quaternion","text":"quaternion(θ, x::Vec; [normalize = true, degree = false])\nquaternion(T, θ, x::Vec; [normalize = true, degree = false])\n\nConstruct Quaternion from angle θ and direction x. The constructed quaternion is normalized such as norm(q) ≈ 1 by default.\n\njulia> q = quaternion(π/4, Vec(0,0,1))\n0.9238795325112867 + 0.0𝙞 + 0.0𝙟 + 0.3826834323650898𝙠\n\njulia> v = rand(Vec{3})\n3-element Tensor{Tuple{3},Float64,1,3}:\n 0.5908446386657102\n 0.7667970365022592\n 0.5662374165061859\n\njulia> (q * v / q).vector ≈ rotmatz(π/4) ⋅ v\ntrue\n\n\n\n\n\n","category":"method"},{"location":"Quaternion/#Tensorial.rotate-Tuple{Vec{3, T} where T, Quaternion}","page":"Quaternion","title":"Tensorial.rotate","text":"rotate(x::Vec, q::Quaternion)\n\nRotate x by quaternion q.\n\nExamples\n\njulia> v = Vec(1.0, 0.0, 0.0)\n3-element Tensor{Tuple{3},Float64,1,3}:\n 1.0\n 0.0\n 0.0\n\njulia> rotate(v, quaternion(π/4, Vec(0,0,1)))\n3-element Tensor{Tuple{3},Float64,1,3}:\n 0.7071067811865475\n 0.7071067811865476\n 0.0\n\n\n\n\n\n","category":"method"},{"location":"Tensor Type/#Tensor-type","page":"Tensor type","title":"Tensor type","text":"","category":"section"},{"location":"Tensor Type/#Type-parameters","page":"Tensor type","title":"Type parameters","text":"","category":"section"},{"location":"Tensor Type/","page":"Tensor type","title":"Tensor type","text":"All tensors are represented by a type Tensor{S, T, N, L} where each type parameter represents following:","category":"page"},{"location":"Tensor Type/","page":"Tensor type","title":"Tensor type","text":"S: The size of Tensors which is specified by using Tuple (e.g., 3x2 tensor becomes Tensor{Tuple{3,2}}).\nT: The type of element which must be T <: Real.\nN: The number of dimensions (the order of tensor).\nL: The number of independent components.","category":"page"},{"location":"Tensor Type/","page":"Tensor type","title":"Tensor type","text":"Basically, the type parameters N and L do not need to be specified for constructing tensors because it can be inferred from the size of tensor S.","category":"page"},{"location":"Tensor Type/#Symmetry","page":"Tensor type","title":"Symmetry","text":"","category":"section"},{"location":"Tensor Type/","page":"Tensor type","title":"Tensor type","text":"If possible, specifying the symmetry of the tensor is good for performance since Tensorial.jl provides the optimal computations. The symmetries can be applied using Symmetry in type parameter S (e.g., Symmetry{Tuple{3,3}}). @Symmetry macro can omit Tuple like @Symmetry{2,2}. The following are examples to specify symmetries:","category":"page"},{"location":"Tensor Type/","page":"Tensor type","title":"Tensor type","text":"A_(ij) with 3x3: Tensor{Tuple{@Symmetry{3,3}}}\nA_(ij)k with 3x3x2: Tensor{Tuple{@Symmetry{3,3}, 2}}\nA_(ijk) with 3x3x3: Tensor{Tuple{@Symmetry{3,3,3}}}\nA_(ij)(kl) with 3x3x3x3: Tensor{Tuple{@Symmetry{3,3}, @Symmetry{3,3}}}","category":"page"},{"location":"Tensor Type/","page":"Tensor type","title":"Tensor type","text":"where the bracket () in indices denotes the symmetry.","category":"page"},{"location":"Cheat Sheet/#Cheat-Sheet","page":"Cheat Sheet","title":"Cheat Sheet","text":"","category":"section"},{"location":"Cheat Sheet/#Constructors","page":"Cheat Sheet","title":"Constructors","text":"","category":"section"},{"location":"Cheat Sheet/","page":"Cheat Sheet","title":"Cheat Sheet","text":"# identity tensors\none(Tensor{Tuple{3,3}})            == Matrix(I,3,3) # second-order identity tensor\none(Tensor{Tuple{@Symmetry{3,3}}}) == Matrix(I,3,3) # symmetric second-order identity tensor\nI  = one(Tensor{NTuple{4,3}})               # fourth-order identity tensor\nIs = one(Tensor{NTuple{2, @Symmetry{3,3}}}) # symmetric fourth-order identity tensor\n\n# zero tensors\nzero(Tensor{Tuple{2,3}}) == zeros(2, 3)\nzero(Tensor{Tuple{@Symmetry{3,3}}}) == zeros(3, 3)\n\n# random tensors\nrand(Tensor{Tuple{2,3}})\nrandn(Tensor{Tuple{2,3}})\n\n# from arrays\nTensor{Tuple{2,2}}([1 2; 3 4]) == [1 2; 3 4]\nTensor{Tuple{@Symmetry{2,2}}}([1 2; 3 4]) == [1 3; 3 4] # lower triangular part is used\n\n# from functions\nTensor{Tuple{2,2}}((i,j) -> i == j ? 1 : 0) == one(Tensor{Tuple{2,2}})\nTensor{Tuple{@Symmetry{2,2}}}((i,j) -> i == j ? 1 : 0) == one(Tensor{Tuple{@Symmetry{2,2}}})\n\n# macros (same interface as StaticArrays.jl)\n@Vec [1,2,3]\n@Vec rand(4)\n@Mat [1 2\n      3 4]\n@Mat rand(4,4)\n@Tensor rand(2,2,2)","category":"page"},{"location":"Cheat Sheet/#Tensor-Operations","page":"Cheat Sheet","title":"Tensor Operations","text":"","category":"section"},{"location":"Cheat Sheet/","page":"Cheat Sheet","title":"Cheat Sheet","text":"# 2nd-order vs. 2nd-order\nx = rand(Tensor{Tuple{2,2}})\ny = rand(Tensor{Tuple{@Symmetry{2,2}}})\nx ⊗ y isa Tensor{Tuple{2,2,@Symmetry{2,2}}} # tensor product\nx ⋅ y isa Tensor{Tuple{2,2}}                # single contraction (x_ij * y_jk)\nx ⊡ y isa Real                              # double contraction (x_ij * y_ij)\n\n# 3rd-order vs. 1st-order\nA = rand(Tensor{Tuple{@Symmetry{2,2},2}})\nv = rand(Vec{2})\nA ⊗ v isa Tensor{Tuple{@Symmetry{2,2},2,2}} # A_ijk * v_l\nA ⋅ v isa Tensor{Tuple{@Symmetry{2,2}}}     # A_ijk * v_k\nA ⊡ v # error\n\n# 4th-order vs. 2nd-order\nII = one(SymmetricFourthOrderTensor{2}) # equal to one(Tensor{Tuple{@Symmetry{2,2}, @Symmetry{2,2}}})\nA = rand(Tensor{Tuple{2,2}})\nS = rand(Tensor{Tuple{@Symmetry{2,2}}})\nII ⊡ A == (A + A') / 2 == symmetric(A) # symmetrizing A, resulting in Tensor{Tuple{@Symmetry{2,2}}}\nII ⊡ S == S\n\n# contraction\nx = rand(Tensor{Tuple{2,2,2}})\ny = rand(Tensor{Tuple{2,@Symmetry{2,2}}})\ncontraction(x, y, Val(1)) isa Tensor{Tuple{2,2, @Symmetry{2,2}}}     # single contraction (== ⋅)\ncontraction(x, y, Val(2)) isa Tensor{Tuple{2,2}}                     # double contraction (== ⊡)\ncontraction(x, y, Val(3)) isa Real                                   # triple contraction (x_ijk * y_ijk)\ncontraction(x, y, Val(0)) isa Tensor{Tuple{2,2,2,2, @Symmetry{2,2}}} # tensor product (== ⊗)\n\n# norm/tr/mean/vol/dev\nx = rand(SecondOrderTensor{3}) # equal to rand(Tensor{Tuple{3,3}})\nv = rand(Vec{3})\nnorm(v)\ntr(x)\nmean(x) == tr(x) / 3 # useful for computing mean stress\nvol(x) + dev(x) == x # decomposition into volumetric part and deviatoric part\n\n# det/inv for 2nd-order tensor\nA = rand(SecondOrderTensor{3})          # equal to one(Tensor{Tuple{3,3}})\nS = rand(SymmetricSecondOrderTensor{3}) # equal to one(Tensor{Tuple{@Symmetry{3,3}}})\ndet(A); det(S)\ninv(A) ⋅ A ≈ one(A)\ninv(S) ⋅ S ≈ one(S)\n\n# inv for 4th-order tensor\nAA = rand(FourthOrderTensor{3})          # equal to one(Tensor{Tuple{3,3,3,3}})\nSS = rand(SymmetricFourthOrderTensor{3}) # equal to one(Tensor{Tuple{@Symmetry{3,3}, @Symmetry{3,3}}})\ninv(AA) ⊡ AA ≈ one(AA)\ninv(SS) ⊡ SS ≈ one(SS)","category":"page"},{"location":"Cheat Sheet/#Automatic-differentiation","page":"Cheat Sheet","title":"Automatic differentiation","text":"","category":"section"},{"location":"Cheat Sheet/","page":"Cheat Sheet","title":"Cheat Sheet","text":"# Real -> Real\ngradient(x -> 2x^2 + x + 3, 3) == (x = 3; 4x + 1)\ngradient(x -> 2.0, 3) == 0.0\n\n# Real -> Tensor\ngradient(x -> Tensor{Tuple{2,2}}((i,j) -> i*x^2), 3) == (x = 3; Tensor{Tuple{2,2}}((i,j) -> 2i*x))\ngradient(x -> one(Tensor{Tuple{2,2}}), 3) == zero(Tensor{Tuple{2,2}})\n\n# Tensor -> Real\ngradient(tr, rand(Tensor{Tuple{3,3}})) == one(Tensor{Tuple{3,3}})\n\n# Tensor -> Tensor\nA = rand(Tensor{Tuple{3,3}})\nD  = gradient(dev, A)            # deviatoric projection tensor\nDs = gradient(dev, symmetric(A)) # symmetric deviatoric projection tensor\nA ⊡ D  ≈ dev(A)\nA ⊡ Ds ≈ symmetric(dev(A))\ngradient(identity, A)  == one(FourthOrderTensor{3})          # 4th-order identity tensor\ngradient(symmetric, A) == one(SymmetricFourthOrderTensor{3}) # symmetric 4th-order identity tensor","category":"page"},{"location":"Cheat Sheet/#Aliases","page":"Cheat Sheet","title":"Aliases","text":"","category":"section"},{"location":"Cheat Sheet/","page":"Cheat Sheet","title":"Cheat Sheet","text":"const Vec{dim, T} = Tensor{Tuple{dim}, T, 1, dim}\nconst Mat{m, n, T, L} = Tensor{Tuple{m, n}, T, 2, L}\nconst SecondOrderTensor{dim, T, L} = Tensor{NTuple{2, dim}, T, 2, L}\nconst FourthOrderTensor{dim, T, L} = Tensor{NTuple{4, dim}, T, 4, L}\nconst SymmetricSecondOrderTensor{dim, T, L} = Tensor{Tuple{@Symmetry{dim, dim}}, T, 2, L}\nconst SymmetricFourthOrderTensor{dim, T, L} = Tensor{NTuple{2, @Symmetry{dim, dim}}, T, 4, L}","category":"page"},{"location":"Continuum Mechanics/","page":"Continuum Mechanics","title":"Continuum Mechanics","text":"DocTestSetup = :(using Tensorial)","category":"page"},{"location":"Continuum Mechanics/#Continuum-Mechanics","page":"Continuum Mechanics","title":"Continuum Mechanics","text":"","category":"section"},{"location":"Continuum Mechanics/","page":"Continuum Mechanics","title":"Continuum Mechanics","text":"Tensor operations for continuum mechanics.","category":"page"},{"location":"Continuum Mechanics/","page":"Continuum Mechanics","title":"Continuum Mechanics","text":"mean(::Tensorial.AbstractSquareTensor)","category":"page"},{"location":"Continuum Mechanics/#Statistics.mean-Tuple{Union{AbstractSymmetricSecondOrderTensor{dim, T}, AbstractTensor{Tuple{dim, dim}, T, 2}} where {dim, T}}","page":"Continuum Mechanics","title":"Statistics.mean","text":"mean(::AbstractSecondOrderTensor)\nmean(::AbstractSymmetricSecondOrderTensor)\n\nCompute the mean value of diagonal entries of a square tensor.\n\nExamples\n\njulia> x = rand(Mat{3,3})\n3×3 Tensor{Tuple{3,3},Float64,2,9}:\n 0.590845  0.460085  0.200586\n 0.766797  0.794026  0.298614\n 0.566237  0.854147  0.246837\n\njulia> mean(x)\n0.5439025118691712\n\n\n\n\n\n","category":"method"},{"location":"Continuum Mechanics/#Deviatoric–volumetric-additive-split","page":"Continuum Mechanics","title":"Deviatoric–volumetric additive split","text":"","category":"section"},{"location":"Continuum Mechanics/","page":"Continuum Mechanics","title":"Continuum Mechanics","text":"vol\ndev","category":"page"},{"location":"Continuum Mechanics/#Tensorial.vol","page":"Continuum Mechanics","title":"Tensorial.vol","text":"vol(::AbstractSecondOrderTensor{3})\nvol(::AbstractSymmetricSecondOrderTensor{3})\n\nCompute the volumetric part of a square tensor. Support only for tensors in 3D.\n\nExamples\n\njulia> x = rand(Mat{3,3})\n3×3 Tensor{Tuple{3,3},Float64,2,9}:\n 0.590845  0.460085  0.200586\n 0.766797  0.794026  0.298614\n 0.566237  0.854147  0.246837\n\njulia> vol(x)\n3×3 Tensor{Tuple{3,3},Float64,2,9}:\n 0.543903  0.0       0.0\n 0.0       0.543903  0.0\n 0.0       0.0       0.543903\n\njulia> vol(x) + dev(x) ≈ x\ntrue\n\n\n\n\n\n","category":"function"},{"location":"Continuum Mechanics/#Tensorial.dev","page":"Continuum Mechanics","title":"Tensorial.dev","text":"dev(::AbstractSecondOrderTensor{3})\ndev(::AbstractSymmetricSecondOrderTensor{3})\n\nCompute the deviatoric part of a square tensor. Support only for tensors in 3D.\n\nExamples\n\njulia> x = rand(Mat{3,3})\n3×3 Tensor{Tuple{3,3},Float64,2,9}:\n 0.590845  0.460085  0.200586\n 0.766797  0.794026  0.298614\n 0.566237  0.854147  0.246837\n\njulia> dev(x)\n3×3 Tensor{Tuple{3,3},Float64,2,9}:\n 0.0469421  0.460085   0.200586\n 0.766797   0.250123   0.298614\n 0.566237   0.854147  -0.297065\n\njulia> tr(dev(x))\n0.0\n\n\n\n\n\n","category":"function"},{"location":"Continuum Mechanics/#Stress-invariants","page":"Continuum Mechanics","title":"Stress invariants","text":"","category":"section"},{"location":"Continuum Mechanics/","page":"Continuum Mechanics","title":"Continuum Mechanics","text":"stress_invariants\ndeviatoric_stress_invariants","category":"page"},{"location":"Continuum Mechanics/#Tensorial.stress_invariants","page":"Continuum Mechanics","title":"Tensorial.stress_invariants","text":"stress_invariants(::AbstractSecondOrderTensor{3})\nstress_invariants(::AbstractSymmetricSecondOrderTensor{3})\n\nReturn NamedTuple storing stress invariants.\n\nbeginaligned\nI_1(bmsigma) = mathrmtr(bmsigma) \nI_2(bmsigma) = frac12 (mathrmtr(bmsigma)^2 - mathrmtr(bmsigma^2)) \nI_3(bmsigma) = det(bmsigma)\nendaligned\n\njulia> σ = rand(SymmetricSecondOrderTensor{3})\n3×3 Tensor{Tuple{Symmetry{Tuple{3,3}}},Float64,2,6}:\n 0.590845  0.766797  0.566237\n 0.766797  0.460085  0.794026\n 0.566237  0.794026  0.854147\n\njulia> I₁, I₂, I₃ = stress_invariants(σ)\n(I1 = 1.9050765715072775, I2 = -0.3695921176777066, I3 = -0.10054272199258936)\n\n\n\n\n\n","category":"function"},{"location":"Continuum Mechanics/#Tensorial.deviatoric_stress_invariants","page":"Continuum Mechanics","title":"Tensorial.deviatoric_stress_invariants","text":"deviatoric_stress_invariants(::AbstractSecondOrderTensor{3})\ndeviatoric_stress_invariants(::AbstractSymmetricSecondOrderTensor{3})\n\nReturn NamedTuple storing deviatoric stress invariants.\n\nbeginaligned\nJ_1(bmsigma) = mathrmtr(mathrmdev(bmsigma)) = 0 \nJ_2(bmsigma) = -frac12 mathrmtr(mathrmdev(bmsigma)^2) \nJ_3(bmsigma) = frac13 mathrmtr(mathrmdev(bmsigma)^3)\nendaligned\n\njulia> σ = rand(SymmetricSecondOrderTensor{3})\n3×3 Tensor{Tuple{Symmetry{Tuple{3,3}}},Float64,2,6}:\n 0.590845  0.766797  0.566237\n 0.766797  0.460085  0.794026\n 0.566237  0.794026  0.854147\n\njulia> J₁, J₂, J₃ = deviatoric_stress_invariants(σ)\n(J1 = 0.0, J2 = 1.5793643654463476, J3 = 0.6463152097154271)\n\n\n\n\n\n","category":"function"},{"location":"Tensor Inversion/","page":"Tensor Inversion","title":"Tensor Inversion","text":"DocTestSetup = :(using Tensorial)","category":"page"},{"location":"Tensor Inversion/#Tensor-Inversion","page":"Tensor Inversion","title":"Tensor Inversion","text":"","category":"section"},{"location":"Tensor Inversion/","page":"Tensor Inversion","title":"Tensor Inversion","text":"Inversion of 2nd and 4th order tensors are supported.","category":"page"},{"location":"Tensor Inversion/","page":"Tensor Inversion","title":"Tensor Inversion","text":"Order = [:function]\nPages = [\"Tensor Inversion.md\"]","category":"page"},{"location":"Tensor Inversion/","page":"Tensor Inversion","title":"Tensor Inversion","text":"Modules = [Tensorial]\nOrder   = [:function]\nPages   = [\"inv.jl\"]","category":"page"},{"location":"Tensor Inversion/#Base.inv","page":"Tensor Inversion","title":"Base.inv","text":"inv(::AbstractSecondOrderTensor)\ninv(::AbstractSymmetricSecondOrderTensor)\ninv(::AbstractFourthOrderTensor)\ninv(::AbstractSymmetricFourthOrderTensor)\n\nCompute the inverse of a tensor.\n\nExamples\n\njulia> x = rand(SecondOrderTensor{3})\n3×3 Tensor{Tuple{3,3},Float64,2,9}:\n 0.590845  0.460085  0.200586\n 0.766797  0.794026  0.298614\n 0.566237  0.854147  0.246837\n\njulia> inv(x)\n3×3 Tensor{Tuple{3,3},Float64,2,9}:\n  19.7146   -19.2802    7.30384\n   6.73809  -10.7687    7.55198\n -68.541     81.4917  -38.8361\n\njulia> x ⋅ inv(x) ≈ one(I)\ntrue\n\n\n\n\n\n","category":"function"},{"location":"Voigt form/","page":"Voigt Form","title":"Voigt Form","text":"DocTestSetup = :(using Tensorial)","category":"page"},{"location":"Voigt form/#Voigt-Form","page":"Voigt Form","title":"Voigt Form","text":"","category":"section"},{"location":"Voigt form/","page":"Voigt Form","title":"Voigt Form","text":"Order = [:function]\nPages = [\"Voigt form.md\"]","category":"page"},{"location":"Voigt form/","page":"Voigt Form","title":"Voigt Form","text":"Modules = [Tensorial]\nOrder   = [:function]\nPages   = [\"voigt.jl\"]","category":"page"},{"location":"Voigt form/#Tensorial.frommandel-Union{Tuple{T}, Tuple{Type{var\"#s46\"} where var\"#s46\"<:Union{SymmetricFourthOrderTensor{dim, T, L} where {dim, T, L}, SymmetricSecondOrderTensor{dim, T, L} where {dim, T, L}}, AbstractArray{T, N} where N}} where T","page":"Voigt Form","title":"Tensorial.frommandel","text":"frommandel(S::Type{<: Union{SymmetricSecondOrderTensor, SymmetricFourthOrderTensor}}, A::AbstractArray{T})\n\nCreate a tensor of type S from Mandel form. This is equivalent to fromvoigt(S, A, offdiagscale = √2).\n\nSee also fromvoigt.\n\n\n\n\n\n","category":"method"},{"location":"Voigt form/#Tensorial.fromvoigt","page":"Voigt Form","title":"Tensorial.fromvoigt","text":"fromvoigt(S::Type{<: Union{SecondOrderTensor, FourthOrderTensor}}, A::AbstractArray{T})\nfromvoigt(S::Type{<: Union{SymmetricSecondOrderTensor, SymmetricFourthOrderTensor}}, A::AbstractArray{T}; [offdiagscale])\n\nConverts an array A stored in Voigt format to a Tensor of type S.\n\nKeyword arguments:\n\noffdiagscale: determines the scaling factor for the offdiagonal elements.\norder: vector of cartesian indices (Tuple{Int, Int}) determining the Voigt order. The default order is [(1,1), (2,2), (3,3), (2,3), (1,3), (1,2), (3,2), (3,1), (2,1)]\n\nSee also tovoigt.\n\njulia> fromvoigt(Mat{3,3}, 1.0:1.0:9.0)\n3×3 Tensor{Tuple{3,3},Float64,2,9}:\n 1.0  6.0  5.0\n 9.0  2.0  4.0\n 8.0  7.0  3.0\n\njulia> fromvoigt(SymmetricSecondOrderTensor{3},\n                 1.0:1.0:6.0,\n                 offdiagscale = 2.0,\n                 order = [(1,1), (2,2), (3,3), (1,2), (1,3), (2,3)])\n3×3 Tensor{Tuple{Symmetry{Tuple{3,3}}},Float64,2,6}:\n 1.0  2.0  2.5\n 2.0  2.0  3.0\n 2.5  3.0  3.0\n\n\n\n\n\n","category":"function"},{"location":"Voigt form/#Tensorial.tomandel-Tuple{Union{SymmetricFourthOrderTensor{dim, T, L} where {dim, T, L}, SymmetricSecondOrderTensor{dim, T, L} where {dim, T, L}}}","page":"Voigt Form","title":"Tensorial.tomandel","text":"tomandel(A::Union{SymmetricSecondOrderTensor, SymmetricFourthOrderTensor})\n\nConvert a tensor to Mandel form which is equivalent to tovoigt(A, offdiagscale = √2).\n\nSee also tovoigt.\n\n\n\n\n\n","category":"method"},{"location":"Voigt form/#Tensorial.tovoigt","page":"Voigt Form","title":"Tensorial.tovoigt","text":"tovoigt(A::Union{SecondOrderTensor, FourthOrderTensor}; [order])\ntovoigt(A::Union{SymmetricSecondOrderTensor, SymmetricFourthOrderTensor}; [order, offdiagonal])\n\nConvert a tensor to Voigt form.\n\nKeyword arguments:\n\noffdiagscale: determines the scaling factor for the offdiagonal elements.\norder: vector of cartesian indices (Tuple{Int, Int}) determining the Voigt order. The default order is [(1,1), (2,2), (3,3), (2,3), (1,3), (1,2), (3,2), (3,1), (2,1)]\n\nSee also fromvoigt.\n\njulia> x = Mat{3,3}(1:9...)\n3×3 Tensor{Tuple{3,3},Int64,2,9}:\n 1  4  7\n 2  5  8\n 3  6  9\n\njulia> tovoigt(x)\n9-element StaticArrays.SArray{Tuple{9},Int64,1,9} with indices SOneTo(9):\n 1\n 5\n 9\n 8\n 7\n 4\n 6\n 3\n 2\n\njulia> x = SymmetricSecondOrderTensor{3}(1:6...)\n3×3 Tensor{Tuple{Symmetry{Tuple{3,3}}},Int64,2,6}:\n 1  2  3\n 2  4  5\n 3  5  6\n\njulia> tovoigt(x; offdiagscale = 2,\n                  order = [(1,1), (2,2), (3,3), (1,2), (1,3), (2,3)])\n6-element StaticArrays.SArray{Tuple{6},Int64,1,6} with indices SOneTo(6):\n  1\n  4\n  6\n  4\n  6\n 10\n\n\n\n\n\n","category":"function"},{"location":"Benchmarks/#Benchmarks","page":"Benchmarks","title":"Benchmarks","text":"","category":"section"},{"location":"Benchmarks/","page":"Benchmarks","title":"Benchmarks","text":"a = rand(Vec{3})\nA = rand(SecondOrderTensor{3})\nS = rand(SymmetricSecondOrderTensor{3})\nB = rand(Tensor{Tuple{3,3,3}})\nAA = rand(FourthOrderTensor{3})\nSS = rand(SymmetricFourthOrderTensor{3})","category":"page"},{"location":"Benchmarks/","page":"Benchmarks","title":"Benchmarks","text":"Operation Tensor Array speed-up\nSingle contraction   \na ⋅ a 2.000 ns 16.433 ns ×8.2\nA ⋅ a 2.400 ns 87.708 ns ×36.5\nS ⋅ a 2.800 ns 87.710 ns ×31.3\nDouble contraction   \nA ⊡ A 5.600 ns 18.054 ns ×3.2\nS ⊡ S 3.700 ns 18.054 ns ×4.9\nB ⊡ A 6.000 ns 221.760 ns ×37.0\nAA ⊡ A 8.809 ns 242.893 ns ×27.6\nSS ⊡ S 6.100 ns 242.821 ns ×39.8\nTensor product   \na ⊗ a 3.200 ns 61.327 ns ×19.2\nCross product   \na × a 3.200 ns 61.327 ns ×19.2\nDeterminant   \ndet(A) 2.400 ns 290.717 ns ×121.1\ndet(S) 2.800 ns 300.000 ns ×107.1\nInverse   \ninv(A) 11.411 ns 748.485 ns ×65.6\ninv(S) 8.208 ns 800.000 ns ×97.5\ninv(AA) 1.370 μs 6.280 μs ×4.6\ninv(SS) 506.736 ns 6.200 μs ×12.2","category":"page"},{"location":"Benchmarks/","page":"Benchmarks","title":"Benchmarks","text":"The benchmarks are generated by runbenchmarks.jl on the following system:","category":"page"},{"location":"Benchmarks/","page":"Benchmarks","title":"Benchmarks","text":"julia> versioninfo()\nJulia Version 1.6.0\nCommit f9720dc2eb (2021-03-24 12:55 UTC)\nPlatform Info:\n  OS: Linux (x86_64-pc-linux-gnu)\n  CPU: Intel(R) Xeon(R) Platinum 8171M CPU @ 2.60GHz\n  WORD_SIZE: 64\n  LIBM: libopenlibm\n  LLVM: libLLVM-11.0.1 (ORCJIT, skylake-avx512)\n","category":"page"},{"location":"Tensor Operations/","page":"Tensor Operations","title":"Tensor Operations","text":"DocTestSetup = :(using Tensorial)","category":"page"},{"location":"Tensor Operations/#Tensor-Operations","page":"Tensor Operations","title":"Tensor Operations","text":"","category":"section"},{"location":"Tensor Operations/","page":"Tensor Operations","title":"Tensor Operations","text":"Order = [:function]\nPages = [\"Tensor Operations.md\"]","category":"page"},{"location":"Tensor Operations/","page":"Tensor Operations","title":"Tensor Operations","text":"Modules = [Tensorial]\nOrder   = [:function]\nPages   = [\"ops.jl\"]","category":"page"},{"location":"Tensor Operations/#LinearAlgebra.cross-Union{Tuple{T2}, Tuple{T1}, Tuple{Vec{1, T1}, Vec{1, T2}}} where {T1, T2}","page":"Tensor Operations","title":"LinearAlgebra.cross","text":"cross(x::Vec{3}, y::Vec{3}) -> Vec{3}\ncross(x::Vec{2}, y::Vec{2}) -> Vec{3}\ncross(x::Vec{1}, y::Vec{1}) -> Vec{3}\nx × y\n\nCompute the cross product between two vectors. The vectors are expanded to 3D frist for dimensions 1 and 2. The infix operator × (written \\times) can also be used. x × y (where × can be typed by \\times<tab>) is a synonym for cross(x, y).\n\njulia> x = rand(Vec{3})\n3-element Tensor{Tuple{3},Float64,1,3}:\n 0.5908446386657102\n 0.7667970365022592\n 0.5662374165061859\n\njulia> y = rand(Vec{3})\n3-element Tensor{Tuple{3},Float64,1,3}:\n 0.4600853424625171\n 0.7940257103317943\n 0.8541465903790502\n\njulia> x × y\n3-element Tensor{Tuple{3},Float64,1,3}:\n  0.20535000738340053\n -0.24415039787171888\n  0.11635375677388776\n\n\n\n\n\n","category":"method"},{"location":"Tensor Operations/#LinearAlgebra.dot-Tuple{AbstractTensor, AbstractTensor}","page":"Tensor Operations","title":"LinearAlgebra.dot","text":"dot(x::AbstractTensor, y::AbstractTensor)\nx ⋅ y\n\nCompute dot product such as a = x_i y_i. This is equivalent to contraction(::AbstractTensor, ::AbstractTensor, Val(1)). x ⋅ y (where ⋅ can be typed by \\cdot<tab>) is a synonym for dot(x, y).\n\nExamples\n\njulia> x = rand(Vec{3})\n3-element Tensor{Tuple{3},Float64,1,3}:\n 0.5908446386657102\n 0.7667970365022592\n 0.5662374165061859\n\njulia> y = rand(Vec{3})\n3-element Tensor{Tuple{3},Float64,1,3}:\n 0.4600853424625171\n 0.7940257103317943\n 0.8541465903790502\n\njulia> a = x ⋅ y\n1.3643452781654772\n\n\n\n\n\n","category":"method"},{"location":"Tensor Operations/#LinearAlgebra.norm-Tuple{AbstractTensor}","page":"Tensor Operations","title":"LinearAlgebra.norm","text":"norm(::AbstractTensor)\n\nCompute norm of a tensor.\n\nExamples\n\njulia> x = rand(Mat{3, 3})\n3×3 Tensor{Tuple{3,3},Float64,2,9}:\n 0.590845  0.460085  0.200586\n 0.766797  0.794026  0.298614\n 0.566237  0.854147  0.246837\n\njulia> norm(x)\n1.7377443667834922\n\n\n\n\n\n","category":"method"},{"location":"Tensor Operations/#LinearAlgebra.tr-Union{Tuple{Union{AbstractSymmetricSecondOrderTensor{dim, T}, AbstractTensor{Tuple{dim, dim}, T, 2}} where T}, Tuple{dim}} where dim","page":"Tensor Operations","title":"LinearAlgebra.tr","text":"tr(::AbstractSecondOrderTensor)\ntr(::AbstractSymmetricSecondOrderTensor)\n\nCompute the trace of a square tensor.\n\nExamples\n\njulia> x = rand(Mat{3,3})\n3×3 Tensor{Tuple{3,3},Float64,2,9}:\n 0.590845  0.460085  0.200586\n 0.766797  0.794026  0.298614\n 0.566237  0.854147  0.246837\n\njulia> tr(x)\n1.6317075356075135\n\n\n\n\n\n","category":"method"},{"location":"Tensor Operations/#Tensorial.contraction-Union{Tuple{N}, Tuple{AbstractTensor, AbstractTensor, Val{N}}} where N","page":"Tensor Operations","title":"Tensorial.contraction","text":"contraction(::AbstractTensor, ::AbstractTensor, ::Val{N})\n\nConduct contraction of N inner indices. For example, N=2 contraction for third-order tensors A_ij = B_ikl C_klj can be computed in Tensorial.jl as\n\njulia> B = rand(Tensor{Tuple{3,3,3}});\n\njulia> C = rand(Tensor{Tuple{3,3,3}});\n\njulia> A = contraction(B, C, Val(2))\n3×3 Tensor{Tuple{3,3},Float64,2,9}:\n 1.36912   1.86751  1.32531\n 1.61744   2.34426  1.94101\n 0.929252  1.89656  1.79015\n\nFollowing symbols are also available for specific contractions:\n\nx ⊗ y (where ⊗ can be typed by \\otimes<tab>): contraction(x, y, Val(0))\nx ⋅ y (where ⋅ can be typed by \\cdot<tab>): contraction(x, y, Val(1))\nx ⊡ y (where ⊡ can be typed by \\boxdot<tab>): contraction(x, y, Val(2))\n\n\n\n\n\n","category":"method"},{"location":"Tensor Operations/#Tensorial.otimes-Tuple{AbstractTensor, AbstractTensor}","page":"Tensor Operations","title":"Tensorial.otimes","text":"otimes(x::AbstractTensor, y::AbstractTensor)\nx ⊗ y\n\nCompute tensor product such as A_ij = x_i y_j. x ⊗ y (where ⊗ can be typed by \\otimes<tab>) is a synonym for otimes(x, y).\n\nExamples\n\njulia> x = rand(Vec{3})\n3-element Tensor{Tuple{3},Float64,1,3}:\n 0.5908446386657102\n 0.7667970365022592\n 0.5662374165061859\n\njulia> y = rand(Vec{3})\n3-element Tensor{Tuple{3},Float64,1,3}:\n 0.4600853424625171\n 0.7940257103317943\n 0.8541465903790502\n\njulia> A = x ⊗ y\n3×3 Tensor{Tuple{3,3},Float64,2,9}:\n 0.271839  0.469146  0.504668\n 0.352792  0.608857  0.654957\n 0.260518  0.449607  0.48365\n\n\n\n\n\n","category":"method"},{"location":"Tensor Operations/#Tensorial.rotate-Tuple{Vec{dim, T} where {dim, T}, Tensor{Tuple{dim, dim}, T, 2, L} where {dim, T, L}}","page":"Tensor Operations","title":"Tensorial.rotate","text":"rotate(x::Vec, R::SecondOrderTensor)\nrotate(x::SecondOrderTensor, R::SecondOrderTensor)\nrotate(x::SymmetricSecondOrderTensor, R::SecondOrderTensor)\n\nRotate x by rotation matrix R. This function can hold the symmetry of SymmetricSecondOrderTensor.\n\nExamples\n\njulia> A = rand(SymmetricSecondOrderTensor{3})\n3×3 Tensor{Tuple{Symmetry{Tuple{3,3}}},Float64,2,6}:\n 0.590845  0.766797  0.566237\n 0.766797  0.460085  0.794026\n 0.566237  0.794026  0.854147\n\njulia> R = rotmatz(π/4)\n3×3 Tensor{Tuple{3,3},Float64,2,9}:\n 0.707107  -0.707107  0.0\n 0.707107   0.707107  0.0\n 0.0        0.0       1.0\n\njulia> rotate(A, R)\n3×3 Tensor{Tuple{Symmetry{Tuple{3,3}}},Float64,2,6}:\n -0.241332   0.0653796  -0.161071\n  0.0653796  1.29226     0.961851\n -0.161071   0.961851    0.854147\n\njulia> R ⋅ A ⋅ R'\n3×3 Tensor{Tuple{3,3},Float64,2,9}:\n -0.241332   0.0653796  -0.161071\n  0.0653796  1.29226     0.961851\n -0.161071   0.961851    0.854147\n\n\n\n\n\n","category":"method"},{"location":"Tensor Operations/#Tensorial.rotmat-Tuple{Real, Vec{3, T} where T}","page":"Tensor Operations","title":"Tensorial.rotmat","text":"rotmat(θ, n::Vec; degree::Bool = false)\n\nConstruct rotation matrix from angle θ and direction n.\n\njulia> x = Vec(1.0, 0.0, 0.0)\n3-element Tensor{Tuple{3},Float64,1,3}:\n 1.0\n 0.0\n 0.0\n\njulia> n = Vec(0.0, 0.0, 1.0)\n3-element Tensor{Tuple{3},Float64,1,3}:\n 0.0\n 0.0\n 1.0\n\njulia> rotmat(π/2, n) ⋅ x\n3-element Tensor{Tuple{3},Float64,1,3}:\n 1.1102230246251565e-16\n 1.0\n 0.0\n\n\n\n\n\n","category":"method"},{"location":"Tensor Operations/#Tensorial.rotmat-Tuple{Real}","page":"Tensor Operations","title":"Tensorial.rotmat","text":"rotmat(θ::Real; degree::Bool = false)\n\nConstruct 2D rotation matrix.\n\nExamples\n\njulia> rotmat(30, degree = true)\n2×2 Tensor{Tuple{2,2},Float64,2,4}:\n 0.866025  -0.5\n 0.5        0.866025\n\n\n\n\n\n","category":"method"},{"location":"Tensor Operations/#Tensorial.rotmat-Tuple{Vec{3, T} where T}","page":"Tensor Operations","title":"Tensorial.rotmat","text":"rotmat(θ::Vec{3}; sequence::Symbol, degree::Bool = false)\nrotmatx(θ::Real)\nrotmaty(θ::Real)\nrotmatz(θ::Real)\n\nConvert Euler angles to rotation matrix. Use 3 characters belonging to the set (X, Y, Z) for intrinsic rotations, or (x, y, z) for extrinsic rotations.\n\nExamples\n\njulia> α, β, γ = rand(Vec{3});\n\njulia> rotmat(Vec(α,β,γ), sequence = :XYZ) ≈ rotmatx(α) ⋅ rotmaty(β) ⋅ rotmatz(γ)\ntrue\n\njulia> rotmat(Vec(α,β,γ), sequence = :xyz) ≈ rotmatz(γ) ⋅ rotmaty(β) ⋅ rotmatx(α)\ntrue\n\njulia> rotmat(Vec(α,β,γ), sequence = :XYZ) ≈ rotmat(Vec(γ,β,α), sequence = :zyx)\ntrue\n\n\n\n\n\n","category":"method"},{"location":"Tensor Operations/#Tensorial.rotmat-Union{Tuple{Pair{Vec{dim, T}, Vec{dim, T}}}, Tuple{T}, Tuple{dim}} where {dim, T}","page":"Tensor Operations","title":"Tensorial.rotmat","text":"rotmat(a => b)\n\nConstruct rotation matrix rotating vector a to b. The norms of two vectors must be the same.\n\nExamples\n\njulia> a = normalize(rand(Vec{3}))\n3-element Tensor{Tuple{3},Float64,1,3}:\n 0.526847334217759\n 0.683741457787621\n 0.5049054419691867\n\njulia> b = normalize(rand(Vec{3}))\n3-element Tensor{Tuple{3},Float64,1,3}:\n 0.36698690362212083\n 0.6333543148133657\n 0.6813097125956302\n\njulia> R = rotmat(a => b)\n3×3 Tensor{Tuple{3,3},Float64,2,9}:\n -0.594528   0.597477   0.538106\n  0.597477  -0.119597   0.792917\n  0.538106   0.792917  -0.285875\n\njulia> R ⋅ a ≈ b\ntrue\n\n\n\n\n\n","category":"method"},{"location":"Tensor Operations/#Tensorial.skew-Tuple{AbstractTensor{Tuple{dim, dim}, T, 2} where {dim, T}}","page":"Tensor Operations","title":"Tensorial.skew","text":"skew(::AbstractSecondOrderTensor)\nskew(::AbstractSymmetricSecondOrderTensor)\n\nCompute skew-symmetric (anti-symmetric) part of a second order tensor.\n\n\n\n\n\n","category":"method"},{"location":"Tensor Operations/#Tensorial.skew-Tuple{Vec{3, T} where T}","page":"Tensor Operations","title":"Tensorial.skew","text":"skew(ω::Vec{3})\n\nConstruct a skew-symmetric (anti-symmetric) tensor W from a vector ω as\n\nbmomega = beginBmatrix\n    omega_1 \n    omega_2 \n    omega_3\nendBmatrix quad\nbmW = beginbmatrix\n     0          -omega_3   omega_2 \n     omega_3  0           -omega_1 \n    -omega_2   omega_1   0\nendbmatrix\n\nExamples\n\njulia> skew(Vec(1,2,3))\n3×3 Tensor{Tuple{3,3},Int64,2,9}:\n  0  -3   2\n  3   0  -1\n -2   1   0\n\n\n\n\n\n","category":"method"},{"location":"#Tensorial","page":"Home","title":"Tensorial","text":"","category":"section"},{"location":"#Introduction","page":"Home","title":"Introduction","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Tensorial provides useful tensor operations (e.g., contraction; tensor product, ⊗; inv; etc.) written in the Julia programming language. The library supports arbitrary size of non-symmetric and symmetric tensors, where symmetries should be specified to avoid wasteful duplicate computations. The way to give a size of the tensor is similar to StaticArrays.jl, and symmetries of tensors can be specified by using @Symmetry. For example, symmetric fourth-order tensor (symmetrizing tensor) is represented in this library as Tensor{Tuple{@Symmetry{3,3}, @Symmetry{3,3}}}. Any tensors can also be used in provided automatic differentiation functions.","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"pkg> add Tensorial","category":"page"},{"location":"#Other-tensor-packages","page":"Home","title":"Other tensor packages","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Einsum.jl\nTensorOprations.jl\nTensors.jl","category":"page"},{"location":"#Inspiration","page":"Home","title":"Inspiration","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"StaticArrays.jl\nTensors.jl","category":"page"}]
}
