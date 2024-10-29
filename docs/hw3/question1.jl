using GLMakie
using LinearAlgebra
function makefig1(τ::Float64)
    t = LinRange(0.0, 5.0, 5000)
    μ(t) = t*exp(sin(3*t))
    cov(t, s) = exp((-abs(t-s)) / (τ))
    μs = μ.(t)
    Σ = Matrix{Float64}(undef, length(t), length(t))
    for idx in CartesianIndices(Σ)
        Σ[idx] = cov(t[idx.I[1]], t[idx.I[2]])
    end
    A = cholesky(Σ).L
    fig = Figure()
    ax = Axis(fig[1, 1])
    for _ in 1:5
        Xi = μs .+ A*randn(length(t))
        lines!(t, Xi)
    end
    lines!(ax, t, μs, color = :red, label = "μ")
    lines!(ax, t, (μs .+ 2), color = :black, linestyle = :dash)
    lines!(ax, t, (μs .+ -2), color = :black, linestyle = :dash, label = "μ ± 2")
    Legend(fig[1, 2], ax)
    save("question1c_$τ.png", fig)
end
