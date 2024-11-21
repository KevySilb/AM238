using GLMakie
using QuadGK
using StaticArrays
using Statistics

one(x::Float64) = (x ≤ 1.0 && x ≥ -1.0) ? (exp(-x))/(exp(1) - exp(-1)) : 0.0
function q1plotPDF()
    xs = LinRange(-1.1, 1.1, 1000)
    fig = Figure();display(fig)
    ax = Axis(fig[1, 1],
        title = "PDF of ξ")
    lines!(ax, xs, one.(xs))
    save("PDFone.png", fig)
end


# weight function
μ(x) = exp(-x) * ^((exp(1.0) - ^(exp(1.0), -1.0)), -1.0)

# define an integral using gauss-kronrod quadrature rule
integ(x::Function, sup::SVector{2}) = quadgk(x, sup[1], sup[2]; atol=1e-8, rtol=1e-8)[1]

# the support of the weight function
sup = SVector{2}(-1.0, 1.0)

function stieltjes(μ::Function, N::Int64, sup::SVector{2})
    # μ: weight function defining the inner product
    # N: number of orthogonal polynomials to compute
    # sup: support (integration bounds) of the weight function

    M = N + 2  # Extend size to accommodate buffer
    n = 2      # Starting index for the recursion

    # Initialize orthogonal polynomials (πn) as functions
    π = Vector{Function}(undef, M)
    π[n-1] = x -> 0.0 * x^0.0  # π₀(x) = 0
    π[n] = x -> 1.0 * x^0.0    # π₁(x) = 1

    # Initialize coefficient vectors αn and βn
    α = Vector{Float64}(undef, M)
    β = Vector{Float64}(undef, M)
    β[n-1] = 0.0  # β₀ = 0
    β[n] = 0.0    # β₁ = 0

    # Compute the first α coefficient (α₂)
    # α₂ = ⟨xπ₁, π₁⟩ / ⟨π₁, π₁⟩
    α[n] = integ(x -> x * π[n](x) * π[n](x) * μ(x), sup) / integ(x -> π[n](x) * π[n](x) * μ(x), sup)
    # Compute the next orthogonal polynomial π₂
    # π₂(x) = (x - α₁)π₁(x) - β₁π₀(x)
    π[n+1] = x -> (x - α[n]) * π[n](x) - β[n] * π[n-1](x)

    for n in 3:M-1
        α[n] = integ(x -> x * π[n](x) * π[n](x) * μ(x), sup) / integ(x -> π[n](x) * π[n](x) * μ(x), sup)
        β[n] = integ(x -> π[n](x) * π[n](x) * μ(x), sup) / integ(x -> π[n-1](x) * π[n-1](x) * μ(x), sup)
        π[n+1] = π[n+1] = x -> (x - α[n]) * π[n](x) - β[n] * π[n-1](x)
    end
    return π
end

function validatepione(π::Vector{Function})
    π_1(x) = x + (2 / (exp(1)^2 - 1))
    xs = LinRange(-1, 1, 1000)
    fig = Figure()
    ax = Axis(fig[1, 1], title = "first orthogonal polynomial validation")
    lines!(ax, xs, π_1.(xs), label = "analytical")
    lines!(ax, xs, π[3].(xs), label = "numerical", linestyle = :dash, color = :red)
    Legend(fig[1, 2], ax)
    save("validationpione.png", fig)
end

function plotpolynomials(π::Vector{Function})
    M = size(π)[1]
    fig = Figure();display(fig)
    ax = Axis(fig[1, 1], title = "Plot of the set of orthogonal polynomials up to degree 6")
    xs = LinRange(-1.0, 1.0, 1000)
    for n in 1:M-1
        lines!(ax, xs, π[n+1].(xs), label="π_$(n-1)")
    end
    Legend(fig[1, 2], ax)
    save("plotpolynomials.png", fig)
end

function isdiagonal(π::Vector{Function}, sup::SVector{2})
    A = Matrix{Float64}(undef, 7, 7)
    for idx in CartesianIndices(A)
        (k, j) = idx.I
        ele = integ(x-> π[k+1](x) * π[j+1](x) * μ(x), sup)
        A[k, j] = ele < 1e-12 ? 1e-8 : ele
    end
    fig = Figure()
    ax = Axis(fig[1, 1], title = "heatmap of the diagona matrix")
    heatmap!(ax, log10.(A))
    ax.yreversed=true
    save("heatmapdiagonal.png", fig)
end

function cumsumtrap(f::Function, x::LinRange{Float64, Int64})
    y = f.(x)
    N = length(x)
    x1 = @view x[1:N-1]
    x2 = @view x[2:N]
    y1 = @view y[1:N-1]
    y2 = @view y[2:N]
    integral = cumsum(((x2.-x1).*(y1.+y2))./2.0)
    integral ./= integral[end]
    return [0; integral]
end

function sampleInverseCDF(x::Float64, points::Matrix{Float64})
    idx = findfirst(points[:, 1] .> x)
    if idx === nothing
        p1 = points[end-1, :]
        p2 = points[end, :]
    elseif idx == 1
        p1 = points[1, :]
        p2 = points[2, :]
    else
        p1 = points[idx-1, :]
        p2 = points[idx, :]
    end
    liy(x, p1, p2)        
end

function liy(x::Float64, p1::Vector{Float64}, p2::Vector{Float64})
    x1, y1 = p1
    x2, y2 = p2
    if isapprox(x1, x2, atol = 1e-12)
        return (y1 + y2) / 2.0
    end
    return y1 + (x - x1)*(y2 - y1)/(x2 - x1)
end


η(ξ) = (ξ - 1) / (2 + sin(2*ξ))
function question2a()
    r = LinRange(-1, 1, 1000)
    fig = Figure();display(fig)
    ax = Axis(fig[1, 1],
        title = "PDF of η(ξ(ω))")
    ys = cumsumtrap(μ, r)
    samples = Vector{Float64}(undef, 50000)
    for i in eachindex(samples)
        samples[i] = sampleInverseCDF(rand(), hcat(ys, r))
    end
    hist!(ax, η.(samples), bins = 80, normalization = :pdf)
    save("question2a.png", fig)
    samples
end

function question2b(πn::Vector{Function})
    a = Vector{Float64}(undef, 7)
    colors = Symbol[:red, :green, :blue, :yellow, :orange, :purple]
    # calculate coefficients
    for k in eachindex(a)
        a[k] = integ(x -> η(x) * πn[k+1](x) * μ(x), sup) / integ(x -> πn[k+1](x) * πn[k+1](x) * μ(x), sup)
    end
    
    fig = Figure();display(fig)
    ax = Axis(fig[1, 1], title="densities of ηM(ξ(ω)) for M = {1, 2, 4, 6}")
    r = LinRange(-1, 1, 1000)
    ys = hcat(cumsumtrap(μ, r), r)
    for M in SVector{4}(1, 2, 4, 6)
        ηM_samples = Vector{Float64}(undef, 50000)
        for l in eachindex(ηM_samples)
            η_sum = 0.0
            ξ = sampleInverseCDF(rand(), ys)
            for k in 1:M+1
                η_sum+=a[k]*πn[k+1](ξ)
            end
            ηM_samples[l] = η_sum
        end
        #density!(ax, ηM_samples, color = (colors[M], 0.3), label = "M = $M", strokecolor = colors[M], strokewidth = 3, strokearound = true)
        hist!(ax, ηM_samples, bins = 80, normalization = :pdf, color = (colors[M], 0.6), label = "M = $M")
    end
    Legend(fig[1, 2], ax)
    save("question2b.png", fig)
end


function question2c()
    r = LinRange(-1, 1, 1000)
    ys = hcat(cumsumtrap(μ, r), r)
    η_samples = Vector{Float64}(undef, 50000)
    for i in eachindex(η_samples)
        η_samples[i] = η(sampleInverseCDF(rand(), ys))
    end
    a = Vector{Float64}(undef, 7)
    for k in eachindex(a)
        a[k] = integ(x -> η(x) * πn[k+1](x) * μ(x), sup) / integ(x -> πn[k+1](x) * πn[k+1](x) * μ(x), sup)
    end
    η6_samples = Vector{Float64}(undef, 50000)
    for l in eachindex(η6_samples)
        η_sum = 0.0
        ξ = sampleInverseCDF(rand(), ys)
        for k in eachindex(a)
            η_sum+=a[k]*πn[k+1](ξ)
        end
        η6_samples[l] = η_sum
    end

    means = Vector{Float64}(undef, 1000)
    variances = Vector{Float64}(undef, 1000)
    for i in eachindex(means)
        sample = rand(η_samples, 1000)
        means[i] = mean(sample)
        variances[i] = var(sample)
    end
    η_mean = mean(means)
    η_var = mean(variances)

    means = Vector{Float64}(undef, 1000)
    variances = Vector{Float64}(undef, 1000)
    for i in eachindex(means)
        sample = rand(η6_samples, 1000)
        means[i] = mean(sample)
        variances[i] = var(sample)
    end
    η6_mean = mean(means)
    η6_var = mean(variances)
    fig = Figure();display(fig)
    ax = Axis(fig[1, 1],
        xticks = (1:2, ["η", "η₆"]),
    title = "mean and variance for η and η_6")
    barplot!(ax, [1, 1, 2, 2], [η_mean, η_var, η6_mean, η6_var],
        dodge = [1, 2, 1, 2],
        color = [1, 2, 1, 2])
    save("question2c.png", fig)
end
