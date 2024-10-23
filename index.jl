### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ 45f57670-911f-11ef-0bbd-0f33f6b482c5
begin
	using AllocCheck: @check_allocs, check_allocs
	using BenchmarkTools: @benchmark, @btime, @belapsed, @ballocated
	using DispatchDoctor: @stable
	using JET: @report_opt, @test_opt
	using LinearAlgebra: dot, mul!
	using PlutoTeachingTools: TwoColumn, ThreeColumn
	using PlutoUI: TableOfContents, with_terminal
	using ProfileCanvas: @profview, @profview_allocs
	using ProgressLogging: @progress
	using Test: @testset, @test, @test_broken, @inferred
end

# ╔═╡ 8880768e-9696-40db-b118-712ab96ce684
md"""
# Writing fast Julia

_It's easy but not obvious_

Guillaume Dalle (EPFL)
"""

# ╔═╡ 8929cf7a-5edd-4c42-bb9f-1aa6f16c14bd
TableOfContents(; depth=2)

# ╔═╡ 4bc7a2da-396f-4707-a5d4-d0623f9379a8
md"""
## Getting started
"""

# ╔═╡ 2b831cb3-5edc-48b8-af26-6d459eb0df91
TwoColumn(
md"""
!!! info "Link to this notebook"
	<https://gdalle.github.io/JuliaOptimizationDays2024-FastJulia/>
""",
md"""
To run it yourself (on Julia 1.10):
```julia$
using Pkg; Pkg.add("Pluto")
using Pluto; Pluto.run()
```
"""
)

# ╔═╡ d33deafa-e7db-4831-9725-d5ba16c7d7a4
md"""
## Writing good Julia
"""

# ╔═╡ a912431e-5b43-460c-8808-ff80b0c95a29
md"""
Check out our blog on best practices for development:

> <https://modernjuliaworkflows.org/>
"""

# ╔═╡ e6469d22-66c5-4029-ae4d-7c13999e1d0b
md"""
# Step 1: start from working code
"""

# ╔═╡ f3f232ed-de69-445c-bd99-9640423247dd
md"""
## Toy example

"""

# ╔═╡ d460ae66-599e-4918-9b5f-ac757bb70308
md"""
Linear regression is an optimization problem:
```math
\min_x f(x) = \lVert Ax - b \rVert^2
```
Its gradient is given by
```math
\nabla f(x) = 2A^\top(Ax-b)
```
Let's say you want to use gradient descent to find the optimal $x$.
"""

# ╔═╡ b825390a-d412-4b05-9fcd-90113b7a115c
md"""
## First implementation
"""

# ╔═╡ 3cbb6036-3f2c-4167-a304-6e87460ec182
begin
	m, n = 10, 20;
	A, b = randn(m, n), randn(m);
	x0 = zeros(n)
end;

# ╔═╡ a4d57bc8-a6c6-4d83-9ed2-509b1a55fb4f
"""
	transp(A::Matrix)

Compute the transpose of `A`.
"""
function transp(A::Matrix)
	m, n = size(A)
	At = Matrix(undef, n, m)  # this is bad
	for i in 1:m, j in 1:n
		At[j, i] = A[i, j]
	end
	return At
end

# ╔═╡ c9c46791-c617-494a-addb-ba600b726a31
"""
	matvec(A::Matrix, v::Vector)

Compute the product `A * v`.
"""
function matvec(A::Matrix, v::Vector)
	r = [
		sum(A[i, :] .* v)  # this is bad
		for i in 1:size(A, 1)
	]
	return r
end

# ╔═╡ 4cbb9054-1d69-4c2a-836b-5cbb55e275ae
function f(x)
	residual = matvec(A, x) - b
	return sum(abs2, residual)
end

# ╔═╡ 5fe48521-88d6-45a3-b68c-00de7f33d356
function ∇f(x::AbstractVector)
	residual = matvec(A, x) - b
	At = transp(A)
	halfgrad = matvec(At, residual)
	grad = 2 .* halfgrad
	return grad
end

# ╔═╡ 53fadf3b-5e70-4fec-b09e-06bfa01d14f6
function descent(; iterations=10^3, step=1e-3)
	x = copy(x0)
	for t in 1:iterations
		x -= step * ∇f(x)
	end
	return x
end

# ╔═╡ 086ec55d-7847-4228-88dd-2aefb602f357
md"""
## Correctness test
"""

# ╔═╡ 3b4e33c7-06b9-43a3-8d18-ab4937febfb6
md"""
Does the code run without errors?
"""

# ╔═╡ bf4d7b63-eda5-4822-a250-8efcf0a07ac7
descent()

# ╔═╡ b2057009-2bf6-4022-ae7b-87f18533ba7e
md"""
Does the code return something coherent?
"""

# ╔═╡ 0aa0d45d-1e81-44c1-95e2-04630a161d92
with_terminal() do
	@testset verbose = true "Gradient descent" begin
		@testset "Gradient correctness" begin
			h = rand(length(x0))  # random direction
			ε = 1e-4
			@test dot(∇f(x0), h) ≈ (f(x0 .+ ε .* h) - f(x0)) ./ ε rtol=1e-2
			@test_broken dot(∇f(x0), h) ≈ (f(x0 .+ ε .* h) - f(x0)) ./ ε rtol=1e-6
		end
		@testset "Objective decrease" begin
			xf = descent()
			@test f(xf) < f(x0)
		end
	end
end

# ╔═╡ 7f2a042b-17bd-4b85-9ece-67595201974f
md"""
# Step 2: measure performance
"""

# ╔═╡ 68febfe2-c4c3-4f47-a177-8de00d857de9
md"""
## Logging
"""

# ╔═╡ 342e0e16-684c-4f52-b14c-41897e41d785
TwoColumn(
md"""
Main tools:

- [`Logging`](https://docs.julialang.org/en/v1/stdlib/Logging/) (standard library)
- [ProgressLogging.jl](https://github.com/JuliaLogging/ProgressLogging.jl)
""",
md"""
Also relevant:

- [ProgressMeter.jl](https://github.com/timholy/ProgressMeter.jl)
"""
)

# ╔═╡ f69c073c-8b20-4de6-af8c-062fc8162213
md"""
You can use logging macros to display messages and variable values during execution, or save those to a file.
"""

# ╔═╡ a22a5d64-04e5-49c8-9702-971a520f4b93
let
	x = 3
	@debug "For internal use" x
	@info "It's all good" x - 1
	@warn "Be careful" x - 2
	@error "Something went wrong" x - 3
end

# ╔═╡ c9da130d-f9b8-4a56-afee-c07c0b5f5d81
md"""
Progress meters are useful to keep track of long computations.
"""

# ╔═╡ deeeff7d-973b-4509-b34b-8ac10af0a1f7
@progress name="My loop:" for i in 1:100
	sleep(0.02)
end

# ╔═╡ cd3ff817-6f3f-4097-9c11-7d60d40bfb65
md"""
## Benchmarking
"""

# ╔═╡ f0993738-ec87-4c44-8416-e0765c4b57fa
TwoColumn(
md"""
Main tools:

- [BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl)
- [Chairmarks.jl](https://github.com/LilithHafner/Chairmarks.jl)
""",
md"""
Also relevant:

- [TimerOutputs.jl](https://github.com/KristofferC/TimerOutputs.jl)
"""
)

# ╔═╡ b86386e2-be66-44b3-b7f2-2a28a0221c03
md"""
The built-in `@time` macro measures how long your code takes and how much it allocates.
It is biased by just-in-time compilation and global variables.
"""

# ╔═╡ c2427919-5d38-4bf1-a93c-96797b124679
let
	myadd(x, y) = x .= x .+ y
	x, y = rand(3), rand(3)
	@time myadd(x, y)
	@time myadd(x, y)
end

# ╔═╡ 03839f29-d658-4d47-b362-34f759b7e4f1
md"""
BenchmarkTools.jl takes care of running the code several times. You can interpolate global variables with a `$` to remove their overhead, or define a setup phase.
"""

# ╔═╡ 1791d65a-7007-4776-af30-dc21fe8238d1
let
	x, y = rand(3), rand(3)
	@benchmark $x .= $x .+ $y
end

# ╔═╡ a7e75d45-d575-4ad3-925f-dc947d3eed94
md"""
# Step 3: detect problems
"""

# ╔═╡ d28cf02e-d872-479b-bff1-e209c083e52a
md"""
Back to our gradient descent!
"""

# ╔═╡ f05dbd8e-5784-4420-9ffb-7edde1676ccd
@benchmark descent()

# ╔═╡ 8f38ce07-28c6-426a-a11f-85cfff449d06
md"""
Why is this so slow? Why does it allocate so much? Let's find out.
"""

# ╔═╡ 9d618baa-1a94-4f3c-b116-8e5391b93a29
md"""
## Profiling
"""

# ╔═╡ 43a27706-e046-472c-b02e-fc0c35199041
TwoColumn(
md"""
Main tools

- [VSCode profiler](https://www.julia-vscode.org/docs/stable/userguide/profiler/)
- [ProfileView.jl](https://github.com/timholy/ProfileView.jl)
""",
md"""
Also relevant:

- [`Profile`](https://docs.julialang.org/en/v1/stdlib/Profile/) (standard library)
- [ProfileSVG.jl](https://github.com/kimikage/ProfileSVG.jl)
- [PProf.jl](https://github.com/JuliaPerf/PProf.jl)
"""
)

# ╔═╡ 3a5a0a42-2c82-4a82-96a0-13bdc31b90dd
md"""
The VSCode extension and various other packages define a `@profview` macro, which shows you how much time is spent in each function of the call stack.

Each tile of the resulting "flame graph" has a width proportional to its duration. Colors have special meaning:
- blue $\implies$ everything is fine
- red $\implies$ "runtime dispatch", a sign of bad type inference
- yellow $\implies$ "garbage collection", a sign of memory allocations
- gray $\implies$ compilation overhead
"""

# ╔═╡ c3af7531-fa47-4efc-a50c-8281fbd31a84
@profview descent(; iterations=10^4)

# ╔═╡ be53e4a1-a055-4c52-a6b9-7c93950707ae
md"""
You can do the same with allocations instead of time.
"""

# ╔═╡ 2a56e88a-7395-486a-a893-5660e1a5da28
@profview_allocs descent(; iterations=10^4)

# ╔═╡ 41462465-fcda-4708-9437-010ffbb5f9b1
md"""
## Superficial introspection
"""

# ╔═╡ 5b5bf5e4-46e8-40a9-8c88-5146456be0fb
md"""
Main tools:

- [`Test.@inferred`](https://docs.julialang.org/en/v1/stdlib/Test/#Test.@inferred)
- [`InteractiveUtils.@code_warntype`](https://docs.julialang.org/en/v1/manual/performance-tips/#man-code-warntype)
"""

# ╔═╡ 54816659-b221-4f89-a628-75e67f888c66
md"""
`Test.@inferred` checks whether the output type of the function is correctly inferred.
"""

# ╔═╡ c7e170b4-7122-4a37-83f3-9e1118c7ded1
@inferred ∇f(x0)

# ╔═╡ fcb231aa-5da5-491b-abda-db60a234c9b6
md"""
`@code_warntype` lets you see what happens inside the function, but not inside other subfunctions. Its output is a bit hard to parse.
"""

# ╔═╡ c5e455a8-aead-4bd4-8b5c-f58ae8b0d8ad
with_terminal() do
	@code_warntype ∇f(x0)
end

# ╔═╡ 2a6fe319-437f-4aab-a36d-cecf95f97f08
md"""
## Deep introspection
"""

# ╔═╡ 56ddc048-c717-4736-a4ad-eaa0d7624c75
md"""
Main tools

- [Cthulhu.jl](https://github.com/JuliaDebug/Cthulhu.jl)
- [JET.jl](https://github.com/aviatesk/JET.jl)
"""

# ╔═╡ 885eefeb-1d47-4e43-8dce-f4195dd4bbcb
md"""
JET.jl recurses into every function call to track down any type inference problems.
Its output is often verbose and may need to be filtered for false positives.
"""

# ╔═╡ a198c3f6-d23e-4be7-8c84-010205765fd8
@report_opt ∇f(x0)

# ╔═╡ fd5743e4-06f4-4019-aec8-40a2142f5664
md"""
Cthulhu.jl allows you to perform this investigation interactively, descending into the call stack from your REPL with `Cthulhu.@descend`. It cannot be used inside Pluto, let's try a live demo in VSCode!
"""

# ╔═╡ c24785af-04ad-4c4f-be0d-74105b70cf00
md"""
# Step 4: learn the rules
"""

# ╔═╡ 6f3c631e-f411-47f1-9624-1588abc18d1a
md"""
Judging by the scary colors in the flame graph, these are the two principles you need to remember.
"""

# ╔═╡ 37015e7c-d90c-42b8-b93f-44b8bd8e0b55
md"""
!!! info "Enable type inference"
	The type of all variables must be _inferrable_ from the _type_ of function inputs (and not their _value_).
"""

# ╔═╡ 2d92f6be-c4b0-4cd3-b5ac-c3e2d09d99c8
md"""
!!! info "Reduce memory allocations"
	Memory must be _pre-allocated_ and _reused_ whenever possible.
"""

# ╔═╡ 832cde06-adb4-49c2-85ce-140ccc360f8e
md"""
## Premature optimization
"""

# ╔═╡ fbd5e4e9-cc1e-4080-b1cc-cbc065fb00c2
md"""
- Only optimize the parts of your code that need it
- Don't reinvent the wheel: if a built-in function or a good package exists, use that
"""

# ╔═╡ eccc9847-d571-4aee-906f-10ef896a10ce
md"""
## General advice
"""

# ╔═╡ ca9edc1e-7c67-4013-bdfe-78f4bf5a96fd
md"""
- Avoid [global variables](https://docs.julialang.org/en/v1/manual/performance-tips/#Avoid-global-variables)
- Put critical code [inside functions](https://docs.julialang.org/en/v1/manual/performance-tips/#Performance-critical-code-should-be-inside-a-function)
- Beware of [closures](https://docs.julialang.org/en/v1/manual/performance-tips/#man-performance-captured) (i.e. functions that return functions)
"""

# ╔═╡ 4d88c814-b82f-42f6-9542-ae26a8cd6554
md"""
## Enable type inference
"""

# ╔═╡ d49bbfc5-9779-44ee-ac8d-8b45ecb8a1d3
md"""
- Functions should [not change variable types](https://docs.julialang.org/en/v1/manual/performance-tips/#Avoid-changing-the-type-of-a-variable) and [always output the same type](https://docs.julialang.org/en/v1/manual/performance-tips/#Write-%22type-stable%22-functions)
- No abstract types in [container initializations](https://docs.julialang.org/en/v1/manual/performance-tips/#man-performance-abstract-container), [`struct` fields](https://docs.julialang.org/en/v1/manual/performance-tips/#Avoid-fields-with-abstract-type) and [`struct` field containers](https://docs.julialang.org/en/v1/manual/performance-tips/#Avoid-fields-with-abstract-containers)
- Leverage [multiple definitions](https://docs.julialang.org/en/v1/manual/performance-tips/#Break-functions-into-multiple-definitions) and [function barriers](https://docs.julialang.org/en/v1/manual/performance-tips/#kernel-functions)
- [Force specialization](https://docs.julialang.org/en/v1/manual/performance-tips/#Be-aware-of-when-Julia-avoids-specializing) if needed
"""

# ╔═╡ 37001dfc-f16a-4032-9b23-24b98d00fb4a
md"""
## Reduce memory allocations
"""

# ╔═╡ 2e984c6f-e3cc-4bbe-bc65-8610363e5074
md"""
- Fix type inference issues first, runtime dispatch can cause allocations
- Prefer mutating functions ([denoted with a bang `!`](https://docs.julialang.org/en/v1/manual/style-guide/#bang-convention)) and [pre-allocate](https://docs.julialang.org/en/v1/manual/performance-tips/#Pre-allocating-outputs) outputs
- Use [views instead of slices](https://docs.julialang.org/en/v1/manual/performance-tips/#man-performance-views) when you don't need copies: `view(A, :, 1)` instead of `A[:, 1]`
- [Fuse vectorized operations](https://docs.julialang.org/en/v1/manual/performance-tips/#More-dots:-Fuse-vectorized-operations) with dots: `x .+= y` instead of `x += y`
"""

# ╔═╡ 607898eb-0c37-4793-b16e-38ad5690430a
md"""
## What you shouldn't try at first
"""

# ╔═╡ 97e6fa54-ba76-4f5b-ab5c-3575e91e17d5
md"""
- Over-specialize types (e.g. force `Float64` everywhere)
- Use [magic macros](https://docs.julialang.org/en/v1/manual/performance-tips/#man-performance-annotations) like `@inbounds`
- Use parallelism when your sequential code is still unoptimized
"""

# ╔═╡ 1e469b8c-2f43-4435-a2d3-f17e4923cb35
md"""
## Additional tricks
"""

# ╔═╡ b876c1d1-1488-4e92-ab18-681a3c9be02b
md"""
- [Parallel computing](https://docs.julialang.org/en/v1/manual/parallel-computing/): multithreading, multiprocessing, GPUs
- [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl) for small vectors and matrices
"""

# ╔═╡ 35a9ba46-2b77-471a-8bea-50a23a83b2b2
md"""
# Step 5: fix the problems
"""

# ╔═╡ 74a7e718-baab-49eb-a80c-87f2b69881f8
md"""
Let's find increasingly clever fixes for your inefficient gradient descent.
"""

# ╔═╡ 840bbf2a-6f26-441d-970f-d841bee4bafb
md"""
## Version 2: type-stable, reduce allocations
"""

# ╔═╡ 6d23f9d0-13d1-4e1e-9a9b-d95976e1a16f
md"""
### New implementation
"""

# ╔═╡ 454cb370-eca6-4efe-894a-3cf6c9c4cf8d
function transp2(A::Matrix{T}) where {T}
	m, n = size(A)
	At = Matrix{T}(undef, n, m)  # initialize with the right type
	for i in 1:m, j in 1:n
		At[j, i] = A[i, j]
	end
	return At
end

# ╔═╡ 164a13b6-04fe-448a-aaa7-68c667c76ed1
function matvec2(A::Matrix{T1}, v::Vector{T2}) where {T1, T2}
	T = promote_type(T1, T2)  # initialize with the right promotion type
	m, n = size(A)
	r = zeros(T, m)
	for i in 1:m
		for j in 1:n  # explicit loop to avoid allocations in sum(A[i, :] .* v)
			r[i] += A[i, j] * v[j]
		end
	end
	return r
end

# ╔═╡ 1c762136-fc6c-4cde-9e9b-e82804a9914e
function f2(x::Vector, A::Matrix, b::Vector)
	return sum(abs2, matvec2(A, x) - b)
end

# ╔═╡ 13a1026e-ef9a-4fcd-b5bf-ea9617bea89c
function ∇f2(x::Vector, A::Matrix, b::Vector) # pass A and b, no global variables
	residual = matvec2(A, x) - b
	At = transp2(A)  # modify transposition
	halfgrad = matvec2(At, residual)
	grad = 2 .* halfgrad
	return grad
end

# ╔═╡ 8e818078-35db-43c9-a8b9-4db1f3f7fe6a
function descent2(
	x0::Vector, A::Matrix, b::Vector;
	iterations=10^3, step=1e-3
)
	x = copy(x0)
	for t in 1:iterations
		x -= step * ∇f2(x, A, b)
	end
	return x
end

# ╔═╡ a4b0ca7c-cb88-4b67-abec-085e54e23075
md"""
### Correctness test
"""

# ╔═╡ bd36f025-39c0-474c-9913-230a04ce5e05
with_terminal() do
	@testset verbose = true "Gradient descent (v2)" begin
		@testset "Gradient correctness" begin
			h = rand(length(x0))  # random direction
			ε = 1e-4
			@test dot(∇f2(x0, A, b), h) ≈ (f2(x0 .+ ε .* h, A, b) - f2(x0, A, b)) ./ ε rtol=1e-2
		end
		@testset "Objective decrease" begin
			xf = descent2(x0, A, b)
			@test f2(xf, A, b) < f2(x0, A, b)
		end
	end
end

# ╔═╡ 0c451b6e-d52a-4644-a107-2e72649549b4
md"""
### Performance checks
"""

# ╔═╡ 1f26ff5d-8d43-4e7b-8a03-4a9e14d23add
@inferred ∇f2(x0, A, b)

# ╔═╡ d30a6b32-1256-4f40-95cd-81f087b4ac48
with_terminal() do
	@code_warntype ∇f2(x0, A, b)
end

# ╔═╡ 645de3e5-a3bc-4485-aa6d-b0b6dcf6b5d4
@report_opt ∇f2(x0, A, b)

# ╔═╡ fbd95663-9a59-4f4f-b833-b255e5853df1
@benchmark descent2($x0, $A, $b)

# ╔═╡ 9fa55d9f-5713-4e21-a55f-0a1255024082
md"""
### Comparative benchmark
"""

# ╔═╡ 42bf06b6-8d37-41e5-b0dd-5a5742359aca
speedup2 = (
	@belapsed descent()
) / (
	@belapsed descent2($x0, $A, $b)
)

# ╔═╡ 914c014f-61b7-477d-9225-5c0509e23a14
memreduc2 = (
	@ballocated descent()
) / (
	@ballocated descent2($x0, $A, $b)
)

# ╔═╡ 20229658-1881-4837-9005-8c7aaeba08ff
md"""
## Version 3: do everything in-place
"""

# ╔═╡ 2051eeb4-503f-4122-84fa-8cedae32ed19
md"""
### New implementation
"""

# ╔═╡ 115e0c16-f2f0-44bc-8e8d-3abbaa15e087
function matvec3!(
	r::AbstractVector{T},  # in-place modification of result
	A::AbstractMatrix,
	v::AbstractVector
) where {T}
	for i in axes(A, 1)
		r[i] = zero(T)
		for j in axes(A, 2)  # explicit loop to avoid allocations with sum(row .* v)
			r[i] += A[i, j] * v[j]
		end
	end
	return r
end

# ╔═╡ c3695be2-e818-4c78-92a8-aaeae05a101b
f3(x, A, b) = f2(x, A, b)

# ╔═╡ c6ab6c57-55ab-4aea-88c7-b00325322744
function ∇f3!(grad, residual, x, A, b, At)
	matvec3!(residual, A, x)
	for i in eachindex(residual)
		residual[i] -= b[i]
	end
	matvec3!(grad, At, residual)
	for j in eachindex(grad)
		grad[j] /= 2
	end
	return grad
end

# ╔═╡ 985f4efc-1adb-42d7-b68f-ac5b2e2bcbb8
function descent3_aux!(
	x::Vector, grad::Vector, residual::Vector,
	A::Matrix, b::Vector, At::Matrix;
	iterations=10^3, step=1e-3
)
	# hot loop without allocations
	for t in 1:iterations
		∇f3!(grad, residual, x, A, b, At)
		for j in eachindex(x, grad)
			x[j] -= step * grad[j]
		end
	end
end

# ╔═╡ b8daadeb-a00d-445d-aed5-6830d9807cc4
function descent3(
	x0::Vector, A::Matrix, b::Vector;
	iterations=10^3, step=1e-3
)
	x = copy(x0)
	grad = similar(x0)
	residual = similar(x0, size(A, 1))
	At = transp2(A)
	descent3_aux!(x, grad, residual, A, b, At; iterations, step)
	return x
end

# ╔═╡ 7f8cd82c-c5de-451e-87e6-fb4d7445fa49
md"""
### Correctness test
"""

# ╔═╡ 9172b1cf-9bd3-4dcf-b358-56120f58e6a1
with_terminal() do
	@testset verbose = true "Gradient descent (v3)" begin
		@testset "Objective decrease" begin
			xf = descent3(x0, A, b)
			@test f3(xf, A, b) < f3(x0, A, b)
		end
	end
end

# ╔═╡ b19f7008-6983-411c-8618-5bf340ed406e
md"""
### Comparative benchmark
"""

# ╔═╡ 9da0174f-8dd1-43b6-b880-aceae8228260
speedup3 = (
	@belapsed descent()
) / (
	@belapsed descent3($x0, $A, $b)
)

# ╔═╡ 280a4599-d312-40f3-a70e-33bf9b7bc70c
memreduc3 = (
	@ballocated descent()
) / (
	@ballocated descent3($x0, $A, $b)
)

# ╔═╡ e177a30c-0864-431a-a377-5be878edd926
md"""
## Version 4: use `LinearAlgebra`
"""

# ╔═╡ d7d96b78-e2ad-45e8-b7e0-616286ed9119
md"""
### New implementation
"""

# ╔═╡ deeea0e8-2fa2-4c5f-9ed7-e135866a45ba
f4(x, A, b) = sum(abs2, A * x - b)

# ╔═╡ 3d69d888-8dd4-4f1e-a8db-b661462533dc
function ∇f4!(grad, residual, x, A, b)
	mul!(residual, A, x)  # mul! performs in-place multiplication
	residual .-= b
	mul!(grad, transpose(A), residual)  # transpose doesn't allocate
	grad ./= 2
	return grad
end

# ╔═╡ 88b798d1-e6cc-450d-b4cd-80dd3c457fba
function descent4_aux!(
	x::Vector, grad::Vector, residual::Vector,
	A::Matrix, b::Vector;
	iterations=10^3, step=1e-3
)
	for t in 1:iterations
		∇f4!(grad, residual, x, A, b)
		x .-= step .* grad
	end
end

# ╔═╡ 24d1418c-1a7e-4d40-b0cb-72f10cea28c4
function descent4(
	x0::Vector, A::Matrix, b::Vector;
	iterations=10^3, step=1e-3
)
	x = copy(x0)
	grad = similar(x0)
	residual = similar(x0, size(A, 1))
	descent4_aux!(x, grad, residual, A, b; iterations, step)
	return x
end

# ╔═╡ 1da169c5-fed0-467c-b7ef-93fa1fc6483a
md"""
### Correctness test
"""

# ╔═╡ fef23923-bafb-4a69-b899-0dbdb19cc4e2
with_terminal() do
	@testset verbose = true "Gradient descent (v4)" begin
		@testset "Objective decrease" begin
			xf = descent4(x0, A, b)
			@test f4(xf, A, b) < f4(x0, A, b)
		end
	end
end

# ╔═╡ c7d4ae32-9e99-493b-b506-5253b4e319f7
md"""
### Comparative benchmark
"""

# ╔═╡ fc8f45d8-e7d7-46ba-9d58-6d2df4dd2774
speedup4 = (
	@belapsed descent()
) / (
	@belapsed descent4($x0, $A, $b)
)

# ╔═╡ a5d0a68b-2394-4992-ac19-6ab17fec2016
memreduc4 = (
	@ballocated descent()
) / (
	@ballocated descent4($x0, $A, $b)
)

# ╔═╡ 7419cf16-5ac2-4bef-9191-f606366f43b8
md"""
# Step 6: prevent regressions
"""

# ╔═╡ e5b89edf-64df-4dec-b0f0-dea2e02f73cb
md"""
## Guarantee type stability
"""

# ╔═╡ 90cb9d1e-e4bd-4da2-b204-bb67d64127df
md"""
Main tools:

- [DispatchDoctor.jl](https://github.com/MilesCranmer/DispatchDoctor.jl)
- [JET.jl](https://github.com/aviatesk/JET.jl)
"""

# ╔═╡ d33777e8-f0f1-4dc8-b763-2a77b0902fe5
md"""
DispatchDoctor.jl allows you to annotate functions or entire modules with `@stable` to ensure that their return type is correctly inferred.
"""

# ╔═╡ d001b6ca-b0bf-4ce6-8222-3a508fcdcdcb
@stable function relu(x)
    if x > 0
        return x
    else
        return 0.0
    end
end

# ╔═╡ 6eb0451f-22b9-4dff-80f6-747d25f49378
relu(1.0)

# ╔═╡ 3e33019f-f7ae-4772-8552-c0c933bf79e7
relu(1)

# ╔═╡ 1204935c-45be-459c-9d24-d94302559b35
md"""
JET.jl can also be used in a test suite to verify that a certain combination of inputs does not cause type instabilities.
"""

# ╔═╡ d2ca9348-6f4e-4d85-b6a0-b7d82c1b03bc
md"""
## Guarantee allocation-free behavior
"""

# ╔═╡ 5a69b299-d1b5-42c3-b114-ae1cf217c2d6
md"""
Main tools:

- [AllocCheck.jl](https://github.com/JuliaLang/AllocCheck.jl)
- [BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl)
"""

# ╔═╡ d84f14d3-ff69-4631-9949-a8505f3035d0
md"""
BenchmarkTools.jl can count the number of heap allocations for a specific function call.
"""

# ╔═╡ e8906daa-44ed-46a1-b7c1-2678ba489917
let 
	bench = @benchmark descent4_aux!(x, grad, residual, $A, $b) setup = (
		x = copy(x0);
		grad = similar(x0);
		residual = similar(x0, m)
	)
	minimum(bench.allocs)
end

# ╔═╡ fba96b1f-3203-4694-8a46-bd07d5d36b64
md"""
AllocCheck.jl can provide a formal guarantee that no allocations are possible for a combination of input types. Like DispatchDoctor.jl, it also provides a macro to annotate functions. Beware of false positives though: just because an allocation is possible does not mean it happens.
"""

# ╔═╡ ce8c5e70-672c-44c9-bc0c-3dfcfcbe9831
let
	V, M = Vector{Float64}, Matrix{Float64}
	check_allocs(descent3_aux!, (V, V, V, M, V, M))  # empty = good
end

# ╔═╡ 38a60ea9-33ce-412b-b425-77f4b5228957
let
	V, M = Vector{Float64}, Matrix{Float64}
	check_allocs(descent4_aux!, (V, V, V, M, V))  # not empty = possible allocations (or false alarms)
end

# ╔═╡ e226d9fd-2b0e-44d2-99c5-c0e763565f38
md"""
## Continuous benchmarking
"""

# ╔═╡ c3d15da8-3362-41c3-b005-58626900fccc
md"""
Main tools:

- [PkgBenchmark.jl](https://github.com/JuliaCI/PkgBenchmark.jl)
- [AirSpeedVelocity.jl](https://github.com/MilesCranmer/AirspeedVelocity.jl)
"""

# ╔═╡ 96ba05b8-6fe9-497d-a4b0-f2e0454d3059
md"""
Running some benchmarks as part of continuous integration is a good idea for package developers to avoid accidental regressions.
"""

# ╔═╡ 7f2a51ed-dc4d-46ab-99f2-44b0e92a83f5
md"""
# Step 7: go further
"""

# ╔═╡ 81a7761e-d63d-4075-9c86-44fc5c11064a
md"""
## Understand hardware
"""

# ╔═╡ b389c3f8-f578-41b0-81ce-a48bebb837fd
md"""
A very good blog post on how your computer computes:

> <https://viralinstruction.com/posts/hardware/>
"""

# ╔═╡ 2ac3c90f-9d9b-42bb-990a-44ece3604e8f
md"""
## Reduce latency
"""

# ╔═╡ 1e778715-a988-412e-b3bd-876cffd2b64b
md"""
We only focused on speeding up functions _once they have been compiled_. But sometimes, compilation time itself is the problem.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
AllocCheck = "9b6a8646-10ed-4001-bbdc-1d2f46dfbb1a"
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
DispatchDoctor = "8d63f2c5-f18a-4cf2-ba9d-b3f60fc568c8"
JET = "c3a54625-cd67-489e-a8e7-0a5a0ff4e31b"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
ProfileCanvas = "efd6af41-a80b-495e-886c-e51b0c7d77a3"
ProgressLogging = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[compat]
AllocCheck = "~0.2.0"
BenchmarkTools = "~1.5.0"
DispatchDoctor = "~0.4.15"
JET = "~0.9.12"
PlutoTeachingTools = "~0.3.1"
PlutoUI = "~0.7.60"
ProfileCanvas = "~0.1.6"
ProgressLogging = "~0.1.4"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.5"
manifest_format = "2.0"
project_hash = "2e4fa690b979e4ba982c72ee7d176262fd6b00c2"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.AllocCheck]]
deps = ["ExprTools", "GPUCompiler", "LLVM", "MacroTools"]
git-tree-sha1 = "4ca13ce2695b68fa558cc02769e834c5ee278e1d"
uuid = "9b6a8646-10ed-4001-bbdc-1d2f46dfbb1a"
version = "0.2.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "f1dff6729bc61f4d49e140da1af55dcd1ac97b2f"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.5.0"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "7eee164f122511d3e4e1ebadb7956939ea7e1c77"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.3.6"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DispatchDoctor]]
deps = ["MacroTools", "Preferences"]
git-tree-sha1 = "19eec8ee5cde1c10aa5e5e0f25d10a3b4c8f8f12"
uuid = "8d63f2c5-f18a-4cf2-ba9d-b3f60fc568c8"
version = "0.4.15"

    [deps.DispatchDoctor.extensions]
    DispatchDoctorChainRulesCoreExt = "ChainRulesCore"
    DispatchDoctorEnzymeCoreExt = "EnzymeCore"

    [deps.DispatchDoctor.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.GPUCompiler]]
deps = ["ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "PrecompileTools", "Preferences", "Scratch", "Serialization", "TOML", "TimerOutputs", "UUIDs"]
git-tree-sha1 = "1d6f290a5eb1201cd63574fbc4440c788d5cb38f"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "0.27.8"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.JET]]
deps = ["CodeTracking", "InteractiveUtils", "JuliaInterpreter", "LoweredCodeUtils", "MacroTools", "Pkg", "PrecompileTools", "Preferences", "Test"]
git-tree-sha1 = "5c5ac91e775b585864015c5c1703cee283071a47"
uuid = "c3a54625-cd67-489e-a8e7-0a5a0ff4e31b"
version = "0.9.12"

    [deps.JET.extensions]
    JETCthulhuExt = "Cthulhu"
    ReviseExt = "Revise"

    [deps.JET.weakdeps]
    Cthulhu = "f68482b8-f384-11e8-15f7-abe071a5a75f"
    Revise = "295af30f-e4ad-537b-8983-00126c2a3abe"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "be3dc50a92e5a386872a493a10050136d4703f9b"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.6.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "2984284a8abcfcc4784d95a9e2ea4e352dd8ede7"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.36"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Preferences", "Printf", "Unicode"]
git-tree-sha1 = "d422dfd9707bec6617335dc2ea3c5172a87d5908"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "9.1.3"

    [deps.LLVM.extensions]
    BFloat16sExt = "BFloat16s"

    [deps.LLVM.weakdeps]
    BFloat16s = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "05a8bd5a42309a9ec82f700876903abce1017dd3"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.34+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "ce5f5621cac23a86011836badfedf664a612cee4"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.5"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.6.4+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoweredCodeUtils]]
deps = ["JuliaInterpreter"]
git-tree-sha1 = "260dc274c1bc2cb839e758588c63d9c8b5e639d1"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "3.0.5"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.1.10"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+4"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.10.0"

[[deps.PlutoHooks]]
deps = ["InteractiveUtils", "Markdown", "UUIDs"]
git-tree-sha1 = "072cdf20c9b0507fdd977d7d246d90030609674b"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0774"
version = "0.0.5"

[[deps.PlutoLinks]]
deps = ["FileWatching", "InteractiveUtils", "Markdown", "PlutoHooks", "Revise", "UUIDs"]
git-tree-sha1 = "8f5fa7056e6dcfb23ac5211de38e6c03f6367794"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0420"
version = "0.1.6"

[[deps.PlutoTeachingTools]]
deps = ["Downloads", "HypertextLiteral", "Latexify", "Markdown", "PlutoLinks", "PlutoUI"]
git-tree-sha1 = "8252b5de1f81dc103eb0293523ddf917695adea1"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.3.1"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "eba4810d5e6a01f612b948c9fa94f905b49087b0"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.60"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.ProfileCanvas]]
deps = ["Base64", "JSON", "Pkg", "Profile", "REPL"]
git-tree-sha1 = "e42571ce9a614c2fbebcaa8aab23bbf8865c624e"
uuid = "efd6af41-a80b-495e-886c-e51b0c7d77a3"
version = "0.1.6"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "80d919dee55b9c50e8d9e2da5eeafff3fe58b539"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.4"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Revise]]
deps = ["CodeTracking", "Distributed", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "REPL", "Requires", "UUIDs", "Unicode"]
git-tree-sha1 = "7f4228017b83c66bd6aa4fddeb170ce487e53bc7"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.6.2"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.1+1"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "3a6f063d690135f5c1ba351412c82bae4d1402bf"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.25"

[[deps.Tricks]]
git-tree-sha1 = "7822b97e99a1672bfb1b49b668a6d46d58d8cbcb"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.9"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"
"""

# ╔═╡ Cell order:
# ╟─8880768e-9696-40db-b118-712ab96ce684
# ╠═45f57670-911f-11ef-0bbd-0f33f6b482c5
# ╠═8929cf7a-5edd-4c42-bb9f-1aa6f16c14bd
# ╟─4bc7a2da-396f-4707-a5d4-d0623f9379a8
# ╟─2b831cb3-5edc-48b8-af26-6d459eb0df91
# ╟─d33deafa-e7db-4831-9725-d5ba16c7d7a4
# ╟─a912431e-5b43-460c-8808-ff80b0c95a29
# ╟─e6469d22-66c5-4029-ae4d-7c13999e1d0b
# ╟─f3f232ed-de69-445c-bd99-9640423247dd
# ╟─d460ae66-599e-4918-9b5f-ac757bb70308
# ╟─b825390a-d412-4b05-9fcd-90113b7a115c
# ╠═3cbb6036-3f2c-4167-a304-6e87460ec182
# ╠═a4d57bc8-a6c6-4d83-9ed2-509b1a55fb4f
# ╠═c9c46791-c617-494a-addb-ba600b726a31
# ╠═4cbb9054-1d69-4c2a-836b-5cbb55e275ae
# ╠═5fe48521-88d6-45a3-b68c-00de7f33d356
# ╠═53fadf3b-5e70-4fec-b09e-06bfa01d14f6
# ╟─086ec55d-7847-4228-88dd-2aefb602f357
# ╟─3b4e33c7-06b9-43a3-8d18-ab4937febfb6
# ╠═bf4d7b63-eda5-4822-a250-8efcf0a07ac7
# ╟─b2057009-2bf6-4022-ae7b-87f18533ba7e
# ╠═0aa0d45d-1e81-44c1-95e2-04630a161d92
# ╟─7f2a042b-17bd-4b85-9ece-67595201974f
# ╟─68febfe2-c4c3-4f47-a177-8de00d857de9
# ╟─342e0e16-684c-4f52-b14c-41897e41d785
# ╟─f69c073c-8b20-4de6-af8c-062fc8162213
# ╠═a22a5d64-04e5-49c8-9702-971a520f4b93
# ╟─c9da130d-f9b8-4a56-afee-c07c0b5f5d81
# ╠═deeeff7d-973b-4509-b34b-8ac10af0a1f7
# ╟─cd3ff817-6f3f-4097-9c11-7d60d40bfb65
# ╟─f0993738-ec87-4c44-8416-e0765c4b57fa
# ╟─b86386e2-be66-44b3-b7f2-2a28a0221c03
# ╠═c2427919-5d38-4bf1-a93c-96797b124679
# ╟─03839f29-d658-4d47-b362-34f759b7e4f1
# ╠═1791d65a-7007-4776-af30-dc21fe8238d1
# ╟─a7e75d45-d575-4ad3-925f-dc947d3eed94
# ╟─d28cf02e-d872-479b-bff1-e209c083e52a
# ╟─f05dbd8e-5784-4420-9ffb-7edde1676ccd
# ╟─8f38ce07-28c6-426a-a11f-85cfff449d06
# ╟─9d618baa-1a94-4f3c-b116-8e5391b93a29
# ╟─43a27706-e046-472c-b02e-fc0c35199041
# ╟─3a5a0a42-2c82-4a82-96a0-13bdc31b90dd
# ╠═c3af7531-fa47-4efc-a50c-8281fbd31a84
# ╟─be53e4a1-a055-4c52-a6b9-7c93950707ae
# ╠═2a56e88a-7395-486a-a893-5660e1a5da28
# ╟─41462465-fcda-4708-9437-010ffbb5f9b1
# ╟─5b5bf5e4-46e8-40a9-8c88-5146456be0fb
# ╟─54816659-b221-4f89-a628-75e67f888c66
# ╠═c7e170b4-7122-4a37-83f3-9e1118c7ded1
# ╟─fcb231aa-5da5-491b-abda-db60a234c9b6
# ╠═c5e455a8-aead-4bd4-8b5c-f58ae8b0d8ad
# ╟─2a6fe319-437f-4aab-a36d-cecf95f97f08
# ╟─56ddc048-c717-4736-a4ad-eaa0d7624c75
# ╟─885eefeb-1d47-4e43-8dce-f4195dd4bbcb
# ╠═a198c3f6-d23e-4be7-8c84-010205765fd8
# ╟─fd5743e4-06f4-4019-aec8-40a2142f5664
# ╟─c24785af-04ad-4c4f-be0d-74105b70cf00
# ╟─6f3c631e-f411-47f1-9624-1588abc18d1a
# ╟─37015e7c-d90c-42b8-b93f-44b8bd8e0b55
# ╟─2d92f6be-c4b0-4cd3-b5ac-c3e2d09d99c8
# ╟─832cde06-adb4-49c2-85ce-140ccc360f8e
# ╟─fbd5e4e9-cc1e-4080-b1cc-cbc065fb00c2
# ╟─eccc9847-d571-4aee-906f-10ef896a10ce
# ╟─ca9edc1e-7c67-4013-bdfe-78f4bf5a96fd
# ╟─4d88c814-b82f-42f6-9542-ae26a8cd6554
# ╟─d49bbfc5-9779-44ee-ac8d-8b45ecb8a1d3
# ╟─37001dfc-f16a-4032-9b23-24b98d00fb4a
# ╟─2e984c6f-e3cc-4bbe-bc65-8610363e5074
# ╟─607898eb-0c37-4793-b16e-38ad5690430a
# ╟─97e6fa54-ba76-4f5b-ab5c-3575e91e17d5
# ╟─1e469b8c-2f43-4435-a2d3-f17e4923cb35
# ╟─b876c1d1-1488-4e92-ab18-681a3c9be02b
# ╟─35a9ba46-2b77-471a-8bea-50a23a83b2b2
# ╟─74a7e718-baab-49eb-a80c-87f2b69881f8
# ╟─840bbf2a-6f26-441d-970f-d841bee4bafb
# ╟─6d23f9d0-13d1-4e1e-9a9b-d95976e1a16f
# ╠═454cb370-eca6-4efe-894a-3cf6c9c4cf8d
# ╠═164a13b6-04fe-448a-aaa7-68c667c76ed1
# ╠═1c762136-fc6c-4cde-9e9b-e82804a9914e
# ╠═13a1026e-ef9a-4fcd-b5bf-ea9617bea89c
# ╠═8e818078-35db-43c9-a8b9-4db1f3f7fe6a
# ╟─a4b0ca7c-cb88-4b67-abec-085e54e23075
# ╠═bd36f025-39c0-474c-9913-230a04ce5e05
# ╟─0c451b6e-d52a-4644-a107-2e72649549b4
# ╠═1f26ff5d-8d43-4e7b-8a03-4a9e14d23add
# ╠═d30a6b32-1256-4f40-95cd-81f087b4ac48
# ╠═645de3e5-a3bc-4485-aa6d-b0b6dcf6b5d4
# ╠═fbd95663-9a59-4f4f-b833-b255e5853df1
# ╟─9fa55d9f-5713-4e21-a55f-0a1255024082
# ╠═42bf06b6-8d37-41e5-b0dd-5a5742359aca
# ╠═914c014f-61b7-477d-9225-5c0509e23a14
# ╟─20229658-1881-4837-9005-8c7aaeba08ff
# ╟─2051eeb4-503f-4122-84fa-8cedae32ed19
# ╠═115e0c16-f2f0-44bc-8e8d-3abbaa15e087
# ╠═c3695be2-e818-4c78-92a8-aaeae05a101b
# ╠═c6ab6c57-55ab-4aea-88c7-b00325322744
# ╠═b8daadeb-a00d-445d-aed5-6830d9807cc4
# ╠═985f4efc-1adb-42d7-b68f-ac5b2e2bcbb8
# ╟─7f8cd82c-c5de-451e-87e6-fb4d7445fa49
# ╠═9172b1cf-9bd3-4dcf-b358-56120f58e6a1
# ╟─b19f7008-6983-411c-8618-5bf340ed406e
# ╠═9da0174f-8dd1-43b6-b880-aceae8228260
# ╠═280a4599-d312-40f3-a70e-33bf9b7bc70c
# ╟─e177a30c-0864-431a-a377-5be878edd926
# ╟─d7d96b78-e2ad-45e8-b7e0-616286ed9119
# ╠═deeea0e8-2fa2-4c5f-9ed7-e135866a45ba
# ╠═3d69d888-8dd4-4f1e-a8db-b661462533dc
# ╠═24d1418c-1a7e-4d40-b0cb-72f10cea28c4
# ╠═88b798d1-e6cc-450d-b4cd-80dd3c457fba
# ╟─1da169c5-fed0-467c-b7ef-93fa1fc6483a
# ╠═fef23923-bafb-4a69-b899-0dbdb19cc4e2
# ╟─c7d4ae32-9e99-493b-b506-5253b4e319f7
# ╠═fc8f45d8-e7d7-46ba-9d58-6d2df4dd2774
# ╠═a5d0a68b-2394-4992-ac19-6ab17fec2016
# ╟─7419cf16-5ac2-4bef-9191-f606366f43b8
# ╟─e5b89edf-64df-4dec-b0f0-dea2e02f73cb
# ╟─90cb9d1e-e4bd-4da2-b204-bb67d64127df
# ╟─d33777e8-f0f1-4dc8-b763-2a77b0902fe5
# ╠═d001b6ca-b0bf-4ce6-8222-3a508fcdcdcb
# ╠═6eb0451f-22b9-4dff-80f6-747d25f49378
# ╠═3e33019f-f7ae-4772-8552-c0c933bf79e7
# ╟─1204935c-45be-459c-9d24-d94302559b35
# ╟─d2ca9348-6f4e-4d85-b6a0-b7d82c1b03bc
# ╟─5a69b299-d1b5-42c3-b114-ae1cf217c2d6
# ╟─d84f14d3-ff69-4631-9949-a8505f3035d0
# ╠═e8906daa-44ed-46a1-b7c1-2678ba489917
# ╟─fba96b1f-3203-4694-8a46-bd07d5d36b64
# ╠═ce8c5e70-672c-44c9-bc0c-3dfcfcbe9831
# ╠═38a60ea9-33ce-412b-b425-77f4b5228957
# ╟─e226d9fd-2b0e-44d2-99c5-c0e763565f38
# ╟─c3d15da8-3362-41c3-b005-58626900fccc
# ╟─96ba05b8-6fe9-497d-a4b0-f2e0454d3059
# ╟─7f2a51ed-dc4d-46ab-99f2-44b0e92a83f5
# ╟─81a7761e-d63d-4075-9c86-44fc5c11064a
# ╟─b389c3f8-f578-41b0-81ce-a48bebb837fd
# ╟─2ac3c90f-9d9b-42bb-990a-44ece3604e8f
# ╟─1e778715-a988-412e-b3bd-876cffd2b64b
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
