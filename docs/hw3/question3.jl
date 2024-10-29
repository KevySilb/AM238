using GLMakie
using Distributions
using KernelDensity
using SpecialFunctions
using QuadGK

p(x) = ((2*(2^(1.0/4.0)))/(gamma((1.0/4.0))))*exp(-2.0*x^4)

function question3c()
    x = LinRange(-2, 2, 1000)
    fig = Figure()
    ax = Axis(fig[1, 1], title = L"$p^*(x) = \frac{2(2)^\frac{1}{4}}{\Gamma\left(\frac{1}{4}\right)}e^{-2x^4}$", 
              xlabel = L"$x$",
              ylabel = L"$p^*(x)$")
    lines!(ax, x, p.(x))
    save("question3c.png", fig)
end

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

function question3partF()
    Δt = 1e-4
    ts = 0.0:Δt:5.0
    N = 1000
    W = Normal(0, sqrt(Δt))
    procs = Matrix{Float64}(undef, length(ts), N)
    println("initializing matrix with uniform random numbers in [1, 2]")
    for i in 1:N
        procs[1, i] = rand() + 1.0
    end
    println("simulating random process")
    for i in 1:N
        println("$N complete")
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
        lines!(ax1, ts, procs[:, i], linewidth = 1)
    end
    d = kde(procs[1, :])
    vlinet = Observable(ts[1])
    kde_data = Observable((d.x, d.density))
    kde_line = lines!(ax2, [0.0], [0.0], color = :blue, label = "KDE")
    
    kde_plot = lift(kde_data) do (x, density)
        kde_line[1] = x
        kde_line[2] = density
    end
    
    vlines!(ax1, vlinet, color = :red, label = "time")
    lines!(ax2, x, p.(x), color = :red, linestyle = :dash, label = L"$p^*(x)$")
    Legend(fig[2, 1], ax1, orientation = :horizontal)
    Legend(fig[2, 2], ax2, orientation = :horizontal)
    println("starting video rendering...")
    record(fig, "question3partF.mp4", 2:100:length(ts); framerate = 30) do k
        println("frame $k")
        vlinet[] = ts[k]
        d = kde(procs[k, :])
        kde_data[] = (d.x, d.density)
    end
    println("video rendered")
end

