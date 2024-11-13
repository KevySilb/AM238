using GLMakie
using Distributions

function partA(τ::Float64, σ::Float64)
    # define the length of subintervals
    Δt = 1e-4
    ts = 0.0:Δt:5.0
    # number of samples
    N = 5
    # Weiner processs
    W = Normal(0, sqrt(Δt))
    # initialize mesh
    X = Matrix{Float64}(undef, length(ts), N)
    Y = Matrix{Float64}(undef, length(ts), N)

    # apply initial conditions
    for i in 1:N
        X[1, i] = rand()
        Y[1, i] = rand()
    end
    # propagate the process
    for i in 1:N
        for j in 2:length(ts)
            ΔW = rand(W)
            X[j, i] = X[j-1, i] - (X[j-1, i]^3)*Δt -τ*Y[j-1, i]*Δt + σ*ΔW
            Y[j, i] = Y[j-1, i] - τ*Y[j-1, i]*Δt + σ*ΔW
        end
    end

    fig = Figure()
    ax = Axis(
        fig[1, 1],
        title = "σ = $σ τ = $τ",
        xlabel = "t",
        ylabel = L"$X_{n+1}$"
    )
    ylims!(ax, -1.0, 1.0)

    for i in 1:N
        lines!(ax, ts, X[:, i])
    end
    return fig
end

using KernelDensity

function simulateB(τ::Float64)
    σ = 0.1
    # define the length of subintervals
    Δt = 1e-4
    ts = 0.0:Δt:20.0
    # number of samples
    N = 1000
    # Weiner processs
    W = Normal(0, sqrt(Δt))
    # initialize mesh
    X = Matrix{Float64}(undef, length(ts), N)
    Y = Matrix{Float64}(undef, length(ts), N)

    # apply initial conditions
    for i in 1:N
        X[1, i] = rand()
        Y[1, i] = rand()
    end
    # propagate the process
    for i in 1:N
        for j in 2:length(ts)
            ΔW = rand(W)
            X[j, i] = X[j-1, i] - (X[j-1, i]^3)*Δt -τ*Y[j-1, i]*Δt + σ*ΔW
            Y[j, i] = Y[j-1, i] - τ*Y[j-1, i]*Δt + σ*ΔW
        end
    end
    return X, Y, length(ts)
end

function partB()
    X1, Y1, ts = simulateB(0.01)
    X2, Y2, _ = simulateB(1.0)
    X3, Y3, _ = simulateB(10.0)
    
    fig = Figure()
    ax = Axis(
        fig[1, 1]
    )
    ylims!(ax, 0.0, 3.0)
    xlims!(ax, -2.0, 2.0)

    d1 = kde(X1[1, :])
    d2 = kde(X2[1, :])
    d3 = kde(X3[1, :])    
    kde_data1 = Observable((d1.x, d1.density))
    kde_data2 = Observable((d2.x, d2.density))
    kde_data3 = Observable((d3.x, d3.density))
    
    kde_line1 = lines!(ax, [0.0], [0.0], color = :red, label = "τ = 0.01")
    kde_line2 = lines!(ax, [0.0], [0.0], color = :blue, label = "τ = 1.0")
    kde_line3 = lines!(ax, [0.0], [0.0], color = :green, label = "τ = 10.0")

    kde_plot1 = lift(kde_data1) do (x, density)
        kde_line1[1] = x
        kde_line1[2] = density
    end
    kde_plot2 = lift(kde_data2) do (x, density)
        kde_line2[1] = x
        kde_line2[2] = density
    end
    kde_plot3 = lift(kde_data3) do (x, density)
        kde_line3[1] = x
        kde_line3[2] = density
    end

    Legend(fig[1, 2], ax)
    record(fig, "partb.mp4", 2:400:ts; framerate = 30) do k
        d1 = kde(X1[k, :])
        d2 = kde(X2[k, :])
        d3 = kde(X3[k, :])
        kde_data1[] = (d1.x, d1.density)
        kde_data2[] = (d2.x, d2.density)
        kde_data3[] = (d3.x, d3.density)
    end
    return fig
end

