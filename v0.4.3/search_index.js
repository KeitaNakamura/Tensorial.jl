var documenterSearchIndex = {"docs":
[{"location":"Tensor Type/#Tensor-type","page":"Tensor type","title":"Tensor type","text":"","category":"section"},{"location":"Tensor Type/#Type-parameters","page":"Tensor type","title":"Type parameters","text":"","category":"section"},{"location":"Tensor Type/","page":"Tensor type","title":"Tensor type","text":"All tensors are represented by a type Tensor{S, T, N, L} where each type parameter represents following:","category":"page"},{"location":"Tensor Type/","page":"Tensor type","title":"Tensor type","text":"S: The size of Tensors which is specified by using Tuple (e.g., 3x2 tensor becomes Tensor{Tuple{3,2}}).\nT: The type of element which must be T <: Real.\nN: The number of dimensions (the order of tensor).\nL: The number of independent components.","category":"page"},{"location":"Tensor Type/","page":"Tensor type","title":"Tensor type","text":"Basically, the type parameters N and L do not need to be specified for constructing tensors because it can be inferred from the size of tensor S.","category":"page"},{"location":"Tensor Type/#Symmetry","page":"Tensor type","title":"Symmetry","text":"","category":"section"},{"location":"Tensor Type/","page":"Tensor type","title":"Tensor type","text":"If possible, specifying the symmetry of the tensor is good for performance since Tensorial.jl provides the optimal computations. The symmetries can be applied using Symmetry in type parameter S (e.g., Symmetry{Tuple{3,3}}). @Symmetry macro can omit Tuple like @Symmetry{2,2}. The following are examples to specify symmetries:","category":"page"},{"location":"Tensor Type/","page":"Tensor type","title":"Tensor type","text":"A_(ij) with 3x3: Tensor{Tuple{@Symmetry{3,3}}}\nA_(ij)k with 3x3x2: Tensor{Tuple{@Symmetry{3,3}, 2}}\nA_(ijk) with 3x3x3: Tensor{Tuple{@Symmetry{3,3,3}}}\nA_(ij)(kl) with 3x3x3x3: Tensor{Tuple{@Symmetry{3,3}, @Symmetry{3,3}}}","category":"page"},{"location":"Tensor Type/","page":"Tensor type","title":"Tensor type","text":"where the bracket () in indices denotes the symmetry.","category":"page"},{"location":"Cheat Sheet/#Cheat-Sheet","page":"Cheat Sheet","title":"Cheat Sheet","text":"","category":"section"},{"location":"Cheat Sheet/#Constructors","page":"Cheat Sheet","title":"Constructors","text":"","category":"section"},{"location":"Cheat Sheet/","page":"Cheat Sheet","title":"Cheat Sheet","text":"# identity tensors\none(Tensor{Tuple{3,3}})            == Matrix(I,3,3) # second-order identity tensor\none(Tensor{Tuple{@Symmetry{3,3}}}) == Matrix(I,3,3) # symmetric second-order identity tensor\nI  = one(Tensor{NTuple{4,3}})               # fourth-order identity tensor\nIs = one(Tensor{NTuple{2, @Symmetry{3,3}}}) # symmetric fourth-order identity tensor\n\n# zero tensors\nzero(Tensor{Tuple{2,3}}) == zeros(2, 3)\nzero(Tensor{Tuple{@Symmetry{3,3}}}) == zeros(3, 3)\n\n# random tensors\nrand(Tensor{Tuple{2,3}})\nrandn(Tensor{Tuple{2,3}})\n\n# from arrays\nTensor{Tuple{2,2}}([1 2; 3 4]) == [1 2; 3 4]\nTensor{Tuple{@Symmetry{2,2}}}([1 2; 3 4]) == [1 3; 3 4] # lower triangular part is used\n\n# from functions\nTensor{Tuple{2,2}}((i,j) -> i == j ? 1 : 0) == one(Tensor{Tuple{2,2}})\nTensor{Tuple{@Symmetry{2,2}}}((i,j) -> i == j ? 1 : 0) == one(Tensor{Tuple{@Symmetry{2,2}}})\n\n# macros (same interface as StaticArrays.jl)\n@Vec [1,2,3]\n@Vec rand(4)\n@Mat [1 2\n      3 4]\n@Mat rand(4,4)\n@Tensor rand(2,2,2)","category":"page"},{"location":"Cheat Sheet/#Tensor-Operations","page":"Cheat Sheet","title":"Tensor Operations","text":"","category":"section"},{"location":"Cheat Sheet/","page":"Cheat Sheet","title":"Cheat Sheet","text":"# 2nd-order vs. 2nd-order\nx = rand(Tensor{Tuple{2,2}})\ny = rand(Tensor{Tuple{@Symmetry{2,2}}})\nx ⊗ y isa Tensor{Tuple{2,2,@Symmetry{2,2}}} # tensor product\nx ⋅ y isa Tensor{Tuple{2,2}}                # single contraction (x_ij * y_jk)\nx ⊡ y isa Real                              # double contraction (x_ij * y_ij)\n\n# 3rd-order vs. 1st-order\nA = rand(Tensor{Tuple{@Symmetry{2,2},2}})\nv = rand(Vec{2})\nA ⊗ v isa Tensor{Tuple{@Symmetry{2,2},2,2}} # A_ijk * v_l\nA ⋅ v isa Tensor{Tuple{@Symmetry{2,2}}}     # A_ijk * v_k\nA ⊡ v # error\n\n# 4th-order vs. 2nd-order\nII = one(SymmetricFourthOrderTensor{2}) # equal to one(Tensor{Tuple{@Symmetry{2,2}, @Symmetry{2,2}}})\nA = rand(Tensor{Tuple{2,2}})\nS = rand(Tensor{Tuple{@Symmetry{2,2}}})\nII ⊡ A == (A + A') / 2 == symmetric(A) # symmetrizing A, resulting in Tensor{Tuple{@Symmetry{2,2}}}\nII ⊡ S == S\n\n# contraction\nx = rand(Tensor{Tuple{2,2,2}})\ny = rand(Tensor{Tuple{2,@Symmetry{2,2}}})\ncontraction(x, y, Val(1)) isa Tensor{Tuple{2,2, @Symmetry{2,2}}}     # single contraction (== ⋅)\ncontraction(x, y, Val(2)) isa Tensor{Tuple{2,2}}                     # double contraction (== ⊡)\ncontraction(x, y, Val(3)) isa Real                                   # triple contraction (x_ijk * y_ijk)\ncontraction(x, y, Val(0)) isa Tensor{Tuple{2,2,2,2, @Symmetry{2,2}}} # tensor product (== ⊗)\n\n# norm/tr/mean/vol/dev\nx = rand(SecondOrderTensor{3}) # equal to rand(Tensor{Tuple{3,3}})\nv = rand(Vec{3})\nnorm(v)\ntr(x)\nmean(x) == tr(x) / 3 # useful for computing mean stress\nvol(x) + dev(x) == x # decomposition into volumetric part and deviatoric part\n\n# det/inv for 2nd-order tensor\nA = rand(SecondOrderTensor{3})          # equal to one(Tensor{Tuple{3,3}})\nS = rand(SymmetricSecondOrderTensor{3}) # equal to one(Tensor{Tuple{@Symmetry{3,3}}})\ndet(A); det(S)\ninv(A) ⋅ A ≈ one(A)\ninv(S) ⋅ S ≈ one(S)\n\n# inv for 4th-order tensor\nAA = rand(FourthOrderTensor{3})          # equal to one(Tensor{Tuple{3,3,3,3}})\nSS = rand(SymmetricFourthOrderTensor{3}) # equal to one(Tensor{Tuple{@Symmetry{3,3}, @Symmetry{3,3}}})\ninv(AA) ⊡ AA ≈ one(AA)\ninv(SS) ⊡ SS ≈ one(SS)","category":"page"},{"location":"Cheat Sheet/#Automatic-differentiation","page":"Cheat Sheet","title":"Automatic differentiation","text":"","category":"section"},{"location":"Cheat Sheet/","page":"Cheat Sheet","title":"Cheat Sheet","text":"# Real -> Real\ngradient(x -> 2x^2 + x + 3, 3) == (x = 3; 4x + 1)\ngradient(x -> 2.0, 3) == 0.0\n\n# Real -> Tensor\ngradient(x -> Tensor{Tuple{2,2}}((i,j) -> i*x^2), 3) == (x = 3; Tensor{Tuple{2,2}}((i,j) -> 2i*x))\ngradient(x -> one(Tensor{Tuple{2,2}}), 3) == zero(Tensor{Tuple{2,2}})\n\n# Tensor -> Real\ngradient(tr, rand(Tensor{Tuple{3,3}})) == one(Tensor{Tuple{3,3}})\n\n# Tensor -> Tensor\nA = rand(Tensor{Tuple{3,3}})\nD  = gradient(dev, A)            # deviatoric projection tensor\nDs = gradient(dev, symmetric(A)) # symmetric deviatoric projection tensor\nA ⊡ D  ≈ dev(A)\nA ⊡ Ds ≈ symmetric(dev(A))\ngradient(identity, A)  == one(FourthOrderTensor{3})          # 4th-order identity tensor\ngradient(symmetric, A) == one(SymmetricFourthOrderTensor{3}) # symmetric 4th-order identity tensor","category":"page"},{"location":"Cheat Sheet/#Aliases","page":"Cheat Sheet","title":"Aliases","text":"","category":"section"},{"location":"Cheat Sheet/","page":"Cheat Sheet","title":"Cheat Sheet","text":"const Vec{dim, T} = Tensor{Tuple{dim}, T, 1, dim}\nconst Mat{m, n, T, L} = Tensor{Tuple{m, n}, T, 2, L}\nconst SecondOrderTensor{dim, T, L} = Tensor{NTuple{2, dim}, T, 2, L}\nconst FourthOrderTensor{dim, T, L} = Tensor{NTuple{4, dim}, T, 4, L}\nconst SymmetricSecondOrderTensor{dim, T, L} = Tensor{Tuple{@Symmetry{dim, dim}}, T, 2, L}\nconst SymmetricFourthOrderTensor{dim, T, L} = Tensor{NTuple{2, @Symmetry{dim, dim}}, T, 4, L}","category":"page"},{"location":"Benchmarks/#Benchmarks","page":"Benchmarks","title":"Benchmarks","text":"","category":"section"},{"location":"Benchmarks/","page":"Benchmarks","title":"Benchmarks","text":"a = rand(Vec{3})\nA = rand(SecondOrderTensor{3})\nS = rand(SymmetricSecondOrderTensor{3})\nB = rand(Tensor{Tuple{3,3,3}})\nAA = rand(FourthOrderTensor{3})\nSS = rand(SymmetricFourthOrderTensor{3})","category":"page"},{"location":"Benchmarks/","page":"Benchmarks","title":"Benchmarks","text":"Operation Tensor Array speed-up\nSingle contraction   \na ⋅ a 1.899 ns 11.011 ns ×5.8\nA ⋅ a 3.600 ns 68.285 ns ×19.0\nS ⋅ a 3.200 ns 68.313 ns ×21.3\nDouble contraction   \nA ⊡ A 3.900 ns 14.227 ns ×3.6\nS ⊡ S 3.100 ns 13.927 ns ×4.5\nB ⊡ A 8.208 ns 242.145 ns ×29.5\nAA ⊡ A 9.810 ns 267.861 ns ×27.3\nSS ⊡ S 6.400 ns 269.167 ns ×42.1\nTensor product   \na ⊗ a 3.900 ns 48.938 ns ×12.5\nCross product   \na × a 3.900 ns 48.938 ns ×12.5\nDeterminant and Inverse   \ndet(A) 2.800 ns 235.662 ns ×84.2\ndet(S) 3.100 ns 232.051 ns ×74.9\ninv(A) 15.631 ns 578.698 ns ×37.0\ninv(S) 10.911 ns 562.500 ns ×51.6","category":"page"},{"location":"Benchmarks/","page":"Benchmarks","title":"Benchmarks","text":"The benchmarks are generated by runbenchmarks.jl on the following system:","category":"page"},{"location":"Benchmarks/","page":"Benchmarks","title":"Benchmarks","text":"julia> versioninfo()\nJulia Version 1.5.3\nCommit 788b2c77c1 (2020-11-09 13:37 UTC)\nPlatform Info:\n  OS: Linux (x86_64-pc-linux-gnu)\n  CPU: Intel(R) Xeon(R) CPU E5-2673 v4 @ 2.30GHz\n  WORD_SIZE: 64\n  LIBM: libopenlibm\n  LLVM: libLLVM-9.0.1 (ORCJIT, broadwell)\n\n","category":"page"},{"location":"Tensor Operations/","page":"Tensor Operations","title":"Tensor Operations","text":"DocTestSetup = :(using Tensorial)","category":"page"},{"location":"Tensor Operations/#Tensor-Operations","page":"Tensor Operations","title":"Tensor Operations","text":"","category":"section"},{"location":"Tensor Operations/","page":"Tensor Operations","title":"Tensor Operations","text":"Order = [:function]\nPages = [\"Tensor Operations.md\"]","category":"page"},{"location":"Tensor Operations/","page":"Tensor Operations","title":"Tensor Operations","text":"Modules = [Tensorial]\nOrder   = [:function]\nPages   = [\"ops.jl\"]","category":"page"},{"location":"Tensor Operations/#Base.inv-Tuple{Union{AbstractTensor{Tuple{Symmetry{Tuple{1,1}}},T,2}, AbstractTensor{Tuple{1,1},T,2}} where T}","page":"Tensor Operations","title":"Base.inv","text":"inv(::AbstractSecondOrderTensor)\ninv(::AbstractSymmetricSecondOrderTensor)\ninv(::AbstractFourthOrderTensor)\ninv(::AbstractSymmetricFourthOrderTensor)\n\nCompute the inverse of a tensor.\n\nExamples\n\njulia> x = rand(SecondOrderTensor{3})\n3×3 Tensor{Tuple{3,3},Float64,2,9}:\n 0.590845  0.460085  0.200586\n 0.766797  0.794026  0.298614\n 0.566237  0.854147  0.246837\n\njulia> inv(x)\n3×3 Tensor{Tuple{3,3},Float64,2,9}:\n  19.7146   -19.2802    7.30384\n   6.73809  -10.7687    7.55198\n -68.541     81.4917  -38.8361\n\njulia> x ⋅ inv(x) ≈ one(I)\ntrue\n\n\n\n\n\n","category":"method"},{"location":"Tensor Operations/#LinearAlgebra.cross-Union{Tuple{T2}, Tuple{T1}, Tuple{Tensor{Tuple{1},T1,1,1},Tensor{Tuple{1},T2,1,1}}} where T2 where T1","page":"Tensor Operations","title":"LinearAlgebra.cross","text":"cross(x::Vec{3}, y::Vec{3}) -> Vec{3}\ncross(x::Vec{2}, y::Vec{2}) -> Vec{3}\ncross(x::Vec{1}, y::Vec{1}) -> Vec{3}\nx × y\n\nCompute the cross product between two vectors. The vectors are expanded to 3D frist for dimensions 1 and 2. The infix operator × (written \\times) can also be used. x × y (where × can be typed by \\times<tab>) is a synonym for cross(x, y).\n\njulia> x = rand(Vec{3})\n3-element Tensor{Tuple{3},Float64,1,3}:\n 0.5908446386657102\n 0.7667970365022592\n 0.5662374165061859\n\njulia> y = rand(Vec{3})\n3-element Tensor{Tuple{3},Float64,1,3}:\n 0.4600853424625171\n 0.7940257103317943\n 0.8541465903790502\n\njulia> x × y\n3-element Tensor{Tuple{3},Float64,1,3}:\n  0.20535000738340053\n -0.24415039787171888\n  0.11635375677388776\n\n\n\n\n\n","category":"method"},{"location":"Tensor Operations/#LinearAlgebra.dot-Tuple{AbstractTensor,AbstractTensor}","page":"Tensor Operations","title":"LinearAlgebra.dot","text":"dot(x::AbstractTensor, y::AbstractTensor)\nx ⋅ y\n\nCompute dot product such as a = x_i y_i. This is equivalent to contraction(::AbstractTensor, ::AbstractTensor, Val(1)). x ⋅ y (where ⋅ can be typed by \\cdot<tab>) is a synonym for dot(x, y).\n\nExamples\n\njulia> x = rand(Vec{3})\n3-element Tensor{Tuple{3},Float64,1,3}:\n 0.5908446386657102\n 0.7667970365022592\n 0.5662374165061859\n\njulia> y = rand(Vec{3})\n3-element Tensor{Tuple{3},Float64,1,3}:\n 0.4600853424625171\n 0.7940257103317943\n 0.8541465903790502\n\njulia> a = x ⋅ y\n1.3643452781654772\n\n\n\n\n\n","category":"method"},{"location":"Tensor Operations/#LinearAlgebra.norm-Tuple{AbstractTensor}","page":"Tensor Operations","title":"LinearAlgebra.norm","text":"norm(::AbstractTensor)\n\nCompute norm of a tensor.\n\nExamples\n\njulia> x = rand(Mat{3, 3})\n3×3 Tensor{Tuple{3,3},Float64,2,9}:\n 0.590845  0.460085  0.200586\n 0.766797  0.794026  0.298614\n 0.566237  0.854147  0.246837\n\njulia> norm(x)\n1.7377443667834922\n\n\n\n\n\n","category":"method"},{"location":"Tensor Operations/#LinearAlgebra.tr-Union{Tuple{Union{AbstractTensor{Tuple{Symmetry{Tuple{dim,dim}}},T,2}, AbstractTensor{Tuple{dim,dim},T,2}} where T}, Tuple{dim}} where dim","page":"Tensor Operations","title":"LinearAlgebra.tr","text":"tr(::AbstractSecondOrderTensor)\ntr(::AbstractSymmetricSecondOrderTensor)\n\nCompute the trace of a square tensor.\n\nExamples\n\njulia> x = rand(Mat{3,3})\n3×3 Tensor{Tuple{3,3},Float64,2,9}:\n 0.590845  0.460085  0.200586\n 0.766797  0.794026  0.298614\n 0.566237  0.854147  0.246837\n\njulia> tr(x)\n1.6317075356075135\n\n\n\n\n\n","category":"method"},{"location":"Tensor Operations/#Statistics.mean-Union{Tuple{Union{AbstractTensor{Tuple{Symmetry{Tuple{dim,dim}}},T,2}, AbstractTensor{Tuple{dim,dim},T,2}} where T}, Tuple{dim}} where dim","page":"Tensor Operations","title":"Statistics.mean","text":"mean(::AbstractSecondOrderTensor)\nmean(::AbstractSymmetricSecondOrderTensor)\n\nCompute the mean value of diagonal entries of a square tensor.\n\nExamples\n\njulia> x = rand(Mat{3,3})\n3×3 Tensor{Tuple{3,3},Float64,2,9}:\n 0.590845  0.460085  0.200586\n 0.766797  0.794026  0.298614\n 0.566237  0.854147  0.246837\n\njulia> mean(x)\n0.5439025118691712\n\n\n\n\n\n","category":"method"},{"location":"Tensor Operations/#Tensorial.contraction-Union{Tuple{N}, Tuple{AbstractTensor,AbstractTensor,Val{N}}} where N","page":"Tensor Operations","title":"Tensorial.contraction","text":"contraction(::AbstractTensor, ::AbstractTensor, ::Val{N})\n\nConduct contraction of N inner indices. For example, N=2 contraction for third-order tensors A_ij = B_ikl C_klj can be computed in Tensorial.jl as\n\njulia> B = rand(Tensor{Tuple{3,3,3}});\n\njulia> C = rand(Tensor{Tuple{3,3,3}});\n\njulia> A = contraction(B, C, Val(2))\n3×3 Tensor{Tuple{3,3},Float64,2,9}:\n 1.36912   1.86751  1.32531\n 1.61744   2.34426  1.94101\n 0.929252  1.89656  1.79015\n\nFollowing symbols are also available for specific contractions:\n\nx ⊗ y (where ⊗ can be typed by \\otimes<tab>): contraction(x, y, Val(0))\nx ⋅ y (where ⋅ can be typed by \\cdot<tab>): contraction(x, y, Val(1))\nx ⊡ y (where ⊡ can be typed by \\boxdot<tab>): contraction(x, y, Val(2))\n\n\n\n\n\n","category":"method"},{"location":"Tensor Operations/#Tensorial.dev-Tuple{Union{AbstractTensor{Tuple{Symmetry{Tuple{3,3}}},T,2}, AbstractTensor{Tuple{3,3},T,2}} where T}","page":"Tensor Operations","title":"Tensorial.dev","text":"dev(::AbstractSecondOrderTensor{3})\ndev(::AbstractSymmetricSecondOrderTensor{3})\n\nCompute the deviatoric part of a square tensor. Support only for tensors in 3D.\n\nExamples\n\njulia> x = rand(Mat{3,3})\n3×3 Tensor{Tuple{3,3},Float64,2,9}:\n 0.590845  0.460085  0.200586\n 0.766797  0.794026  0.298614\n 0.566237  0.854147  0.246837\n\njulia> dev(x)\n3×3 Tensor{Tuple{3,3},Float64,2,9}:\n 0.0469421  0.460085   0.200586\n 0.766797   0.250123   0.298614\n 0.566237   0.854147  -0.297065\n\njulia> tr(dev(x))\n0.0\n\n\n\n\n\n","category":"method"},{"location":"Tensor Operations/#Tensorial.otimes-Tuple{AbstractTensor,AbstractTensor}","page":"Tensor Operations","title":"Tensorial.otimes","text":"otimes(x::AbstractTensor, y::AbstractTensor)\nx ⊗ y\n\nCompute tensor product such as A_ij = x_i y_j. x ⊗ y (where ⊗ can be typed by \\otimes<tab>) is a synonym for otimes(x, y).\n\nExamples\n\njulia> x = rand(Vec{3})\n3-element Tensor{Tuple{3},Float64,1,3}:\n 0.5908446386657102\n 0.7667970365022592\n 0.5662374165061859\n\njulia> y = rand(Vec{3})\n3-element Tensor{Tuple{3},Float64,1,3}:\n 0.4600853424625171\n 0.7940257103317943\n 0.8541465903790502\n\njulia> A = x ⊗ y\n3×3 Tensor{Tuple{3,3},Float64,2,9}:\n 0.271839  0.469146  0.504668\n 0.352792  0.608857  0.654957\n 0.260518  0.449607  0.48365\n\n\n\n\n\n","category":"method"},{"location":"Tensor Operations/#Tensorial.rotmat-Tuple{Real}","page":"Tensor Operations","title":"Tensorial.rotmat","text":"rotmat(θ::Real; degree::Bool = false)\n\nConstruct 2D rotation matrix.\n\nExamples\n\njulia> rotmat(30, degree = true)\n2×2 Tensor{Tuple{2,2},Float64,2,4}:\n 0.866025  -0.5\n 0.5        0.866025\n\n\n\n\n\n","category":"method"},{"location":"Tensor Operations/#Tensorial.rotmat-Tuple{Tensor{Tuple{3},T,1,3} where T}","page":"Tensor Operations","title":"Tensorial.rotmat","text":"rotmat(θ::Vec{3}; sequence::Symbol, degree::Bool = false)\nrotmatx(θ::Real)\nrotmaty(θ::Real)\nrotmatz(θ::Real)\n\nConvert Euler angles to rotation matrix. Use 3 characters belonging to the set (X, Y, Z) for intrinsic rotations, or (x, y, z) for extrinsic rotations.\n\nExamples\n\njulia> α, β, γ = rand(Vec{3});\n\njulia> rotmat(Vec(α,β,γ), sequence = :XYZ) ≈ rotmatx(α) ⋅ rotmaty(β) ⋅ rotmatz(γ)\ntrue\n\njulia> rotmat(Vec(α,β,γ), sequence = :xyz) ≈ rotmatz(γ) ⋅ rotmaty(β) ⋅ rotmatx(α)\ntrue\n\njulia> rotmat(Vec(α,β,γ), sequence = :XYZ) ≈ rotmat(Vec(γ,β,α), sequence = :zyx)\ntrue\n\n\n\n\n\n","category":"method"},{"location":"Tensor Operations/#Tensorial.rotmat-Union{Tuple{Pair{Tensor{Tuple{dim},T,1,dim},Tensor{Tuple{dim},T,1,dim}}}, Tuple{T}, Tuple{dim}} where T where dim","page":"Tensor Operations","title":"Tensorial.rotmat","text":"rotmat(a => b)\n\nConstruct rotation matrix rotating vector a to b. The norms of two vectors must be the same.\n\nExamples\n\njulia> a = rand(Vec{3}); a /= norm(a);\n\njulia> b = rand(Vec{3}); b /= norm(b);\n\njulia> R = rotmat(a => b);\n\njulia> R ⋅ a ≈ b\ntrue\n\n\n\n\n\n","category":"method"},{"location":"Tensor Operations/#Tensorial.skew-Tuple{AbstractTensor{Tuple{dim,dim},T,2} where T where dim}","page":"Tensor Operations","title":"Tensorial.skew","text":"skew(::AbstractSecondOrderTensor)\nskew(::AbstractSymmetricSecondOrderTensor)\n\nCompute skew-symmetric (anti-symmetric) part of a second order tensor.\n\n\n\n\n\n","category":"method"},{"location":"Tensor Operations/#Tensorial.vol-Tuple{Union{AbstractTensor{Tuple{Symmetry{Tuple{3,3}}},T,2}, AbstractTensor{Tuple{3,3},T,2}} where T}","page":"Tensor Operations","title":"Tensorial.vol","text":"vol(::AbstractSecondOrderTensor{3})\nvol(::AbstractSymmetricSecondOrderTensor{3})\n\nCompute the volumetric part of a square tensor. Support only for tensors in 3D.\n\nExamples\n\njulia> x = rand(Mat{3,3})\n3×3 Tensor{Tuple{3,3},Float64,2,9}:\n 0.590845  0.460085  0.200586\n 0.766797  0.794026  0.298614\n 0.566237  0.854147  0.246837\n\njulia> vol(x)\n3×3 Tensor{Tuple{3,3},Float64,2,9}:\n 0.543903  0.0       0.0\n 0.0       0.543903  0.0\n 0.0       0.0       0.543903\n\njulia> vol(x) + dev(x) ≈ x\ntrue\n\n\n\n\n\n","category":"method"},{"location":"#Tensorial","page":"Home","title":"Tensorial","text":"","category":"section"},{"location":"#Introduction","page":"Home","title":"Introduction","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Tensorial provides useful tensor operations (e.g., contraction; tensor product, ⊗; inv; etc.) written in the Julia programming language. The library supports arbitrary size of non-symmetric and symmetric tensors, where symmetries should be specified to avoid wasteful duplicate computations. The way to give a size of the tensor is similar to StaticArrays.jl, and symmetries of tensors can be specified by using @Symmetry. For example, symmetric fourth-order tensor (symmetrizing tensor) is represented in this library as Tensor{Tuple{@Symmetry{3,3}, @Symmetry{3,3}}}. Any tensors can also be used in provided automatic differentiation functions.","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"pkg> add Tensorial","category":"page"},{"location":"#Other-tensor-packages","page":"Home","title":"Other tensor packages","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Einsum.jl\nTensorOprations.jl\nTensors.jl","category":"page"},{"location":"#Inspiration","page":"Home","title":"Inspiration","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"StaticArrays.jl\nTensors.jl","category":"page"}]
}
