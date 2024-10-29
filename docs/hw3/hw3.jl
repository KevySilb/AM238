using GLMakie
using Distributions
using LinearAlgebra
using KernelDensity
using SpecialFunctions

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
    for i in 1:5
        Xi = μs .+ A*randn(length(t))
        lines!(t, Xi)
    end
    lines!(ax, t, μs, color = :red, label = "μ")
    lines!(ax, t, (μs .+ 2), color = :black, linestyle = :dash)
    lines!(ax, t, (μs .+ -2), color = :black, linestyle = :dash, label = "μ ± 2")
    Legend(fig[1, 2], ax)
    save("question1c_$τ.png", fig)
end
makefig1(0.02);
makefig1(1.0);

function MRG32k3a(seed::Integer, N::Integer)
    m₁ = 2.0^32 - 209
    m₂ = 2.0^32 - 22853

    X = Vector{Float64}(undef, N+3)
    Y = Vector{Float64}(undef, N+3)

    for i = 1:3
        X[i] = Float64(seed)
        Y[i] = Float64(seed)
    end

    for j in 1:length(X) - 3
        X[j+3] = mod(1403580*X[j+1] - 810728*X[j], m₁)
        Y[j+3] = mod(527642*Y[j+2] - 1370589*Y[j], m₂)
    end
    
    transformX(Xk::Float64) = Xk ≥ 0 ? Xk : Xk - m₁*Xk
    transformY(Yk::Float64) = Yk ≥ 0 ? Yk : Yk - m₂*Yk
    
    Xhat = transformX.(X)
    Yhat = transformY.(Y)

    Uk = Vector{Float64}(undef, N)
    for i in eachindex(Xhat)
        if i > length(Xhat) - 3
            break
        end
        Uk[i] = mod(Xhat[i+3] - Yhat[i+3], m₁) / (m₁ + 1)
    end
    Uk
end

function question2()
    fig = Figure()
    ax = Axis(fig[1, 1],
        title = "Relative frequencies of Uⱼ for N = 10⁶",
        xlabel = "Uⱼ",
        ylabel = "frequency"
    )
    hist!(ax, MRG32k3a(111, 1000000), bins = 80)
    save("question2.png", fig)
end
question2();

p(x) = (2^(7/8) / gamma(1/4)) * exp(-2*x^4)
function question3c()
    x = LinRange(-2, 2, 1000)
    fig = Figure()
    ax = Axis(fig[1, 1], title = L"$p^*(x) = \frac{2^{\frac{7}{8}}}{\Gamma \left(\frac{1}{4} \right)} e^{-2x^4}$", 
              xlabel = L"$x$",
              ylabel = L"$p^*(x)$")
    lines!(ax, x, p.(x))
    save("question3c.png", fig)
end
question3c();

function question3partE()
    Δt = 1e-4
    ts = 0.0:Δt:5.0
    N = 5
    W = Normal(0, sqrt(Δt))
    procs = Matrix{Float64}(undef, length(ts), N)
    for i in 1:N
        procs[1, i] = rand() + 1.0
    end

    for i in 1:N
        for j in 2:length(ts)
            procs[j, i] = procs[j-1, i] - (procs[j-1, i]^3)*Δt + 0.5*rand(W)
        end
    end

    fig = Figure()
    ax = Axis(fig[1, 1],
        title = L"$X_{k+1} = X_k - X_k^3 \Delta t + \frac{1}{2} \Delta W_k$",
        xlabel = "x")
    for i in 1:N
        lines!(ax, ts, procs[:, i])
    end
    save("question3e.png", fig)
end
question3partE()

function question3partF()
    Δt = 1e-4
    ts = 0.0:Δt:5.0
    N = 100
    W = Normal(0, sqrt(Δt))
    procs = Matrix{Float64}(undef, length(ts), N)
    for i in 1:N
        procs[1, i] = rand() + 1.0
    end
    for i in 1:N
        for j in 2:length(ts)
            procs[j, i] = procs[j-1, i] - (procs[j-1, i]^3)*Δt + 0.5*rand(W)
        end
    end
    fig = Figure();display(fig)
    ax1 = Axis(fig[1, 1],
    title = "$N SDE paths")
    ax2 = Axis(fig[1, 2],
        title = "KDE Density")
    x = LinRange(-5, 5, 1000)
    xlims!(ax2, -5, 5)    
    for i in 1:N
        lines!(ax1, ts, procs[:, i])
    end
    d = kde(procs[1, :])
    vlinet = Observable(ts[1])
    kde_data = Observable((d.x, d.density))
    kde_line = lines!(ax2, [0.0], [0.0], color = :blue, label = "KDE")
    kde_plot = lift(kde_data) do (x, density)
        kde_line[1] = x
        kde_line[2] = density
    end
    vert = vlines!(ax1, vlinet, color = :red, label = "time")
    lines!(ax2, x, p.(x), color = :red, linestyle = :dash, label = L"$p^*(x)$")
    Legend(fig[2, 1], ax1, orientation = :horizontal)
    Legend(fig[2, 2], ax2, orientation = :horizontal)
    record(fig, "question3partF.mp4", 2:50:length(ts); framerate = 60) do k
        vlinet[] = ts[k]
        d = kde(procs[k, :])
        kde_data[] = (d.x, d.density)
    end
end
question3partF();
